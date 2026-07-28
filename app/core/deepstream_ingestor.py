from __future__ import annotations

import logging
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from app.core.raw_stream_router import RawStreamRouter
import cv2
import numpy as np
from app.core.frontend_frame_worker import FrontendFrameWorker
from app.core.router import TaskRouter
from app.core.source_registry import (
    RTSP,
    SOURCE_TYPES,
    SourceRecord,
    SourceRegistry,
    canonical_source_type,
)
from app.core.video_ingestor import VideoFileIngestor

LOGGER = logging.getLogger(__name__)


def _load_gstreamer() -> tuple[Any, Any]:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import GLib, Gst
    except (ImportError, ValueError) as exc:
        raise RuntimeError(
            "DeepStream ingestion requires Linux, GStreamer Python bindings, and "
            "the NVIDIA DeepStream plugins. Run the DeepStream Docker service."
        ) from exc
    Gst.init(None)
    return Gst, GLib


@dataclass(slots=True)
class DeepStreamSourceState:
    source_id: str
    source_uri: str
    display_uri: str
    source_type: str

    pipeline: Any
    source: Any
    decoded_sink: Any
    raw_sink: Any | None
    bus: Any

    bus_handler_id: int
    source_pad_handler_id: int
    decoded_sink_handler_id: int
    raw_sink_handler_id: int | None

    frame_width: int
    frame_height: int
    delivery_target_fps: float | None
    frontend_frame_index: int = -1
    codec: str | None = None
    next_frame_due_monotonic: float = 0.0
    latest_frame: np.ndarray | None = None
    latest_source_frame: np.ndarray | None = None
    latest_version: int = 0
    submitted_version: int = 0
    frame_index: int = -1
    source_time_seconds: float | None = None

    decoded_samples: int = 0
    raw_samples: int = 0
    raw_bytes: int = 0
    rate_limited_frames: int = 0
    received_frames: int = 0
    pre_submit_replacements: int = 0
    submitted_frames: int = 0

    last_frame_monotonic: float = 0.0
    last_raw_packet_monotonic: float = 0.0
    last_error: str | None = None
    warnings: int = 0
    source_frame_width: int = 0
    source_frame_height: int = 0


class DeepStreamIngestor:
    """Uses DeepStream/NVDEC for file and RTSP decoding, then feeds the router."""

    SCHEDULER_MAX_FPS = 240.0

    REQUIRED_ELEMENTS = (
        "rtspsrc",
        "rtph264depay",
        "rtph265depay",
        "h264parse",
        "h265parse",
        "tee",
        "queue",
        "nvv4l2decoder",
        "nvvideoconvert",
        "identity",
        "capsfilter",
        "appsink",
    )

    def __init__(
        self,
        *,
        registry: SourceRegistry,
        router: TaskRouter,
        project_root: Path,
        gpu_resize_enabled: bool = True,
        loop: bool = True,
        source_type_filter: str = RTSP,
        raw_stream_router: RawStreamRouter | None = None,
        frontend_frame_worker: FrontendFrameWorker | None = None,
        max_sources: int = 256,
        rtsp_enabled: bool = True,
        rtsp_transport: str = "tcp",
        rtsp_latency_ms: int = 500,
        rtsp_reconnect_seconds: float = 3.0,
        rtsp_stall_timeout_seconds: int = 30,
        skip_taskless_sources: bool = True,
        gst_loader: Callable[[], tuple[Any, Any]] = _load_gstreamer,
    ) -> None:
        if source_type_filter not in SOURCE_TYPES:
            raise ValueError(f"source_type_filter must be one of {sorted(SOURCE_TYPES)}")
        self.source_type_filter = source_type_filter
        self.max_sources = max(1, int(max_sources))
        self.registry = registry
        self.router = router
        self.project_root = project_root
        self.raw_stream_router = raw_stream_router
        self.frontend_frame_worker = frontend_frame_worker
        self.gpu_resize_enabled = bool(gpu_resize_enabled)
        # NOTE: GPU resize is now handled by the nvvideoconvert capsfilter
        # in the GStreamer pipeline.  The _gpu_resize_available flag is kept
        # only for status/diagnostics.
        self._gpu_resize_available = False
        self.loop = bool(loop)
        self.rtsp_enabled = bool(rtsp_enabled)
        self.skip_taskless_sources = bool(skip_taskless_sources)
        self.rtsp_transport = (
            rtsp_transport.strip().lower()
            if rtsp_transport.strip().lower() in {"tcp", "udp"}
            else "tcp"
        )
        self.rtsp_latency_ms = max(0, int(rtsp_latency_ms))
        self.rtsp_reconnect_seconds = max(0.5, float(rtsp_reconnect_seconds))
        self.rtsp_stall_timeout_seconds = max(
            0, int(rtsp_stall_timeout_seconds)
        )
        self.gst_loader = gst_loader

        self._gst: Any | None = None
        self._glib: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started = threading.Event()
        self._lock = threading.RLock()
        self._states: dict[str, DeepStreamSourceState] = {}
        self._closing_sources: set[str] = set()
        self._close_threads: dict[str, threading.Thread] = {}
        self._retry_after: dict[str, float] = {}
        self._failed_sources: set[str] = set()
        self._frame_sequences: dict[str, int] = {}
        self._loop_counts: dict[str, int] = {}
        self._gst_source_ids: dict[str, int] = {}
        self._round_sequence = 0
        self._rounds_submitted = 0
        self._frames_submitted = 0
        self._open_failures = 0
        self._reconnects = 0
        self._last_error: str | None = None

    @staticmethod
    def is_supported_source(record: SourceRecord) -> bool:
        return VideoFileIngestor.is_video_source(record)

    def _filter_by_type(self, records: list[SourceRecord]) -> list[SourceRecord]:
        return [
            r for r in records
            if canonical_source_type(
                r.source_uri,
                r.source_type,
            ) == self.source_type_filter
        ]

    def _resolve_uri(self, source_uri: str) -> str:
        if VideoFileIngestor.is_rtsp_uri(source_uri):
            return source_uri
        path = Path(source_uri).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve().as_uri()

    def _require_runtime(self) -> tuple[Any, Any]:
        if self._gst is None or self._glib is None:
            raise RuntimeError("DeepStream runtime is not initialized")
        return self._gst, self._glib

    @staticmethod
    def _set_if_supported(element: Any, name: str, value: Any) -> None:
        if element.find_property(name) is not None:
            element.set_property(name, value)

    @staticmethod
    def _safe_element_name(source_id: str) -> str:
        display_uri = VideoFileIngestor.redact_uri(source_id)
        return "".join(
            character if character.isalnum() else "_"
            for character in display_uri
        )

    def _gst_source_id(self, source_id: str) -> int:
        """Return a stable numeric ID so nvurisrcbin logs never show source -1."""
        with self._lock:
            assigned = self._gst_source_ids.get(source_id)
            if assigned is not None:
                return assigned

            used = set(self._gst_source_ids.values())
            trailing_number = re.search(r"(\d+)$", source_id)
            preferred = int(trailing_number.group(1)) if trailing_number else -1
            if preferred < 0 or preferred > 2_147_483_647 or preferred in used:
                preferred = 0
                while preferred in used:
                    preferred += 1
            self._gst_source_ids[source_id] = preferred
            return preferred

    def _make(self, factory: str, name: str) -> Any:
        Gst, _ = self._require_runtime()
        element = Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"Required GStreamer element is unavailable: {factory}")
        return element

    def _build_encoded_branches(
        self,
        *,
        pipeline: Any,
        source_id: str,
        safe_id: str,
        codec: str,
        decoded_sink: Any,
        gpu_convert: Any,
        bgrx_caps: Any,
        pacer: Any,
    ) -> dict[str, Any]:
        """Build one decode pipeline with two decoded branches.

        Branch 1 keeps the decoded camera resolution and delivers BGR frames to
 
        """
        Gst, _ = self._require_runtime()

        if codec == "h264":
            depay = self._make("rtph264depay", f"h264_depay_{safe_id}")
            parser = self._make("h264parse", f"h264_parser_{safe_id}")
            encoded_caps_value = (
                "video/x-h264,stream-format=byte-stream,alignment=au"
            )
        elif codec == "h265":
            depay = self._make("rtph265depay", f"h265_depay_{safe_id}")
            parser = self._make("h265parse", f"h265_parser_{safe_id}")
            encoded_caps_value = (
                "video/x-h265,stream-format=byte-stream,alignment=au"
            )
        else:
            raise ValueError(f"Unsupported codec: {codec}")

        self._set_if_supported(parser, "config-interval", -1)

        encoded_caps = self._make("capsfilter", f"encoded_caps_{safe_id}")
        encoded_caps.set_property(
            "caps",
            Gst.Caps.from_string(encoded_caps_value),
        )

        decode_queue = self._make("queue", f"decode_queue_{safe_id}")
        decoder = self._make("nvv4l2decoder", f"decoder_{safe_id}")
        decoded_tee = self._make("tee", f"decoded_tee_{safe_id}")

        # Original-resolution decoded branch -> FrontendFrameWorker.
        raw_queue = self._make("queue", f"raw_queue_{safe_id}")
        raw_convert = self._make("nvvideoconvert", f"raw_convert_{safe_id}")
        raw_caps = self._make("capsfilter", f"raw_caps_{safe_id}")
        raw_sink = self._make("appsink", f"raw_sink_{safe_id}")

        raw_caps.set_property(
            "caps",
            Gst.Caps.from_string("video/x-raw,format=BGRx"),
        )

        # Queue configuration. Each branch is independent and may drop old frames.
        for queue, max_buffers in (
            (decode_queue, 4),
            (raw_queue, 2),
        ):
            queue.set_property("leaky", 2)
            queue.set_property("max-size-buffers", max_buffers)
            queue.set_property("max-size-bytes", 0)
            queue.set_property("max-size-time", 0)

        raw_sink.set_property("emit-signals", True)
        raw_sink.set_property("sync", False)
        raw_sink.set_property("max-buffers", 1)
        raw_sink.set_property("drop", True)
        self._set_if_supported(raw_sink, "enable-last-sample", False)

        elements = (
            depay,
            parser,
            encoded_caps,
            decode_queue,
            decoder,
            decoded_tee,
            raw_queue,
            raw_convert,
            raw_caps,
            raw_sink,
        )
        for element in elements:
            pipeline.add(element)

        # RTP/encoded input -> one hardware decoder.
        if not depay.link(parser):
            raise RuntimeError("Could not link depayloader to parser")
        if not parser.link(encoded_caps):
            raise RuntimeError("Could not link parser to encoded caps")
        if not encoded_caps.link(decode_queue):
            raise RuntimeError("Could not link encoded caps to decode queue")
        if not decode_queue.link(decoder):
            raise RuntimeError("Could not link decode queue to NVDEC")
        if not decoder.link(decoded_tee):
            raise RuntimeError("Could not link NVDEC to decoded tee")

        # Original-resolution decoded branch -> RawStreamRouter.
        if not decoded_tee.link(raw_queue):
            raise RuntimeError("Could not link decoded tee to raw queue")
        if not raw_queue.link(raw_convert):
            raise RuntimeError("Could not link raw queue to raw converter")
        if not raw_convert.link(raw_caps):
            raise RuntimeError("Could not link raw converter to raw caps")
        if not raw_caps.link(raw_sink):
            raise RuntimeError("Could not link raw caps to raw appsink")

        # AI branch -> resize caps -> decoded_sink -> TaskRouter.
        if not decoded_tee.link(pacer):
            raise RuntimeError("Could not link decoded tee to AI pacer")
        if not pacer.link(gpu_convert):
            raise RuntimeError("Could not link AI pacer to nvvideoconvert")
        if not gpu_convert.link(bgrx_caps):
            raise RuntimeError("Could not link AI converter to 640x640 caps")
        if not bgrx_caps.link(decoded_sink):
            raise RuntimeError("Could not link AI caps to decoded appsink")

        raw_handler_id = raw_sink.connect(
            "new-sample",
            self._on_raw_sample,
            source_id,
        )

        # The elements are added while the parent pipeline is already running.
        for element in elements:
            if not element.sync_state_with_parent():
                raise RuntimeError(
                    f"Could not sync GStreamer element state: {element.get_name()}"
                )

        return {
            "depay": depay,
            "raw_sink": raw_sink,
            "raw_handler_id": raw_handler_id,
            "codec": codec,
        }

    def _on_rtsp_pad_added(
        self,
        _: Any,
        pad: Any,
        context: dict[str, Any],
    ) -> None:
        Gst, _ = self._require_runtime()

        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            return

        structure = caps.get_structure(0)
        media = structure.get_string("media")
        encoding_name = structure.get_string("encoding-name")

        if media != "video" or not encoding_name:
            return

        encoding_name = encoding_name.upper()
        if encoding_name == "H264":
            codec = "h264"
        elif encoding_name in {"H265", "HEVC"}:
            codec = "h265"
        else:
            LOGGER.warning(
                "Unsupported RTSP video codec source=%s codec=%s",
                VideoFileIngestor.redact_uri(context["source_id"]),
                encoding_name,
            )
            return

        if context["branch_created"]:
            return

        try:
            branch = self._build_encoded_branches(
                pipeline=context["pipeline"],
                source_id=context["source_id"],
                safe_id=context["safe_id"],
                codec=codec,
                decoded_sink=context["decoded_sink"],
                gpu_convert=context["gpu_convert"],
                bgrx_caps=context["bgrx_caps"],
                pacer=context["pacer"],
            )

            sink_pad = branch["depay"].get_static_pad("sink")
            if sink_pad is None:
                raise RuntimeError("Depayloader sink pad is unavailable")

            result = pad.link(sink_pad)
            if result != Gst.PadLinkReturn.OK:
                raise RuntimeError(
                    f"Could not link RTSP pad to {codec} depayloader: {result}"
                )

            context["branch_created"] = True
            context["codec"] = codec
            context["raw_sink"] = branch["raw_sink"]
            context["raw_handler_id"] = branch["raw_handler_id"]

            with self._lock:
                state = self._states.get(context["source_id"])
                if state is not None:
                    state.codec = codec
                    state.raw_sink = branch["raw_sink"]
                    state.raw_sink_handler_id = branch["raw_handler_id"]

            LOGGER.info(
                "DeepStream encoded branch ready source=%s codec=%s",
                VideoFileIngestor.redact_uri(context["source_id"]),
                codec,
            )
        except Exception as exc:
            self._mark_failed(
                context["source_id"],
                f"Could not create encoded RTSP branch: "
                f"{type(exc).__name__}: {exc}",
            )

    def _on_raw_sample(
        self,
        sink: Any,
        source_id: str,
    ) -> Any:
        """Deliver an original-resolution decoded BGR frame to RawStreamRouter."""
        Gst, _ = self._require_runtime()

        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        try:
            caps = sample.get_caps()
            if caps is None or caps.get_size() == 0:
                raise ValueError("Raw decoded sample has no caps")

            structure = caps.get_structure(0)
            width = int(structure.get_value("width"))
            height = int(structure.get_value("height"))
            pixel_format = str(structure.get_value("format"))

            buffer = sample.get_buffer()
            if buffer is None:
                raise ValueError("Raw decoded sample has no buffer")

            payload = buffer.extract_dup(0, buffer.get_size())
            frame = self._decode_cpu_sample(
                payload,
                width=width,
                height=height,
                pixel_format=pixel_format,
            )

            pts_ns = (
                None
                if buffer.pts == Gst.CLOCK_TIME_NONE
                else int(buffer.pts)
            )
            duration_ns = (
                None
                if buffer.duration == Gst.CLOCK_TIME_NONE
                else int(buffer.duration)
            )

           
            with self._lock:
                state = self._states.get(source_id)

                if state is None:
                    return Gst.FlowReturn.OK

                state.frontend_frame_index += 1
                frame_index = state.frontend_frame_index

                state.latest_source_frame = frame
                state.source_frame_width = width
                state.source_frame_height = height
                state.raw_samples += 1
                state.raw_bytes += int(frame.nbytes)
                state.last_raw_packet_monotonic = time.monotonic()

            source_time_seconds = (
                None
                if pts_ns is None
                else float(pts_ns) / float(Gst.SECOND)
            )

            if self.frontend_frame_worker is not None:
                self.frontend_frame_worker.submit_frame(
                    source_id=source_id,
                    frame=frame,
                    frame_index=frame_index,
                    source_time_seconds=source_time_seconds,
                )



        except Exception:
            LOGGER.exception(
                "Raw decoded-frame delivery failed source=%s",
                VideoFileIngestor.redact_uri(source_id),
            )
            return Gst.FlowReturn.ERROR

        return Gst.FlowReturn.OK

    def _redact_error(self, source_id: str, message: str) -> str:
        display_uri = VideoFileIngestor.redact_uri(source_id)
        with self._lock:
            state = self._states.get(source_id)
            if state is not None:
                display_uri = state.display_uri
        return message.replace(source_id, display_uri)

    def _mark_failed(
        self,
        source_id: str,
        message: str,
        *,
        expected: bool = False,
    ) -> None:
        safe_message = self._redact_error(source_id, message)
        display_uri = VideoFileIngestor.redact_uri(source_id)
        with self._lock:
            state = self._states.get(source_id)
            if state is not None:
                state.last_error = safe_message
                display_uri = state.display_uri
            self._last_error = f"{display_uri}: {safe_message}"
            self._failed_sources.add(source_id)
            self._retry_after[source_id] = (
                time.monotonic()
                if expected
                else time.monotonic() + self.rtsp_reconnect_seconds
            )
        log = LOGGER.info if expected else LOGGER.error
        log("DeepStream source %s: %s", display_uri, safe_message)

    def _on_bus_message(self, bus: Any, message: Any, source_id: str) -> None:
        Gst, _ = self._require_runtime()
        with self._lock:
            state = self._states.get(source_id)
            if state is None or state.bus is not bus:
                return
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            detail = str(error)
            if debug:
                detail = f"{detail} ({debug})"
            self._mark_failed(source_id, detail)
        elif message.type == Gst.MessageType.EOS:
            if self._should_loop_source(source_id):
                with self._lock:
                    if source_id not in self._failed_sources:
                        self._loop_counts[source_id] = (
                            self._loop_counts.get(source_id, 0) + 1
                        )
                LOGGER.info(
                    "DeepStream file reached EOS; restarting cleanly: %s",
                    VideoFileIngestor.redact_uri(source_id),
                )
                self._mark_failed(
                    source_id,
                    "End of stream; source restart scheduled",
                    expected=True,
                )
            else:
                LOGGER.info(
                    "DeepStream file reached EOS: %s",
                    VideoFileIngestor.redact_uri(source_id),
                )
                with self._lock:
                    self._failed_sources.add(source_id)
                    self._retry_after[source_id] = float("inf")
        elif message.type == Gst.MessageType.WARNING:
            warning, _ = message.parse_warning()
            safe_warning = self._redact_error(source_id, str(warning))
            with self._lock:
                state = self._states.get(source_id)
                if state is not None:
                    state.warnings += 1
                    state.last_error = safe_warning
            LOGGER.warning(
                "DeepStream source warning: %s: %s",
                VideoFileIngestor.redact_uri(source_id),
                safe_warning,
            )

    def _should_loop_source(self, source_id: str) -> bool:
        """Resolve the live per-source loop policy, falling back to runtime default."""
        record = self.registry.get(source_id)
        return self.loop if record is None else bool(record.loop)

    def _on_new_sample(self, sink: Any, source_id: str) -> Any:
        Gst, _ = self._require_runtime()
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        now = time.monotonic()
        with self._lock:
            state = self._states.get(source_id)
            if state is None or state.decoded_sink is not sink:
                return Gst.FlowReturn.OK
            state.decoded_samples += 1
            if state.delivery_target_fps is not None:
                period = 1.0 / state.delivery_target_fps
                tolerance = min(0.004, period * 0.10)
                if (
                    state.next_frame_due_monotonic > 0.0
                    and now + tolerance < state.next_frame_due_monotonic
                ):
                    state.rate_limited_frames += 1
                    return Gst.FlowReturn.OK
                if state.next_frame_due_monotonic <= 0.0:
                    state.next_frame_due_monotonic = now + period
                else:
                    periods = max(
                        1,
                        math.floor(
                            max(
                                0.0,
                                now - state.next_frame_due_monotonic,
                            )
                            / period
                        )
                        + 1,
                    )
                    state.next_frame_due_monotonic += periods * period

        try:
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            width = int(structure.get_value("width"))
            height = int(structure.get_value("height"))
            pixel_format = str(structure.get_value("format"))
            buffer = sample.get_buffer()
            payload = buffer.extract_dup(0, buffer.get_size())
            frame = self._decode_cpu_sample(
                payload, width=width, height=height, pixel_format=pixel_format
            )
            source_time = None
            if buffer.pts != Gst.CLOCK_TIME_NONE:
                source_time = float(buffer.pts) / float(Gst.SECOND)
        except Exception as exc:
            self._mark_failed(source_id, f"Could not map decoded frame: {type(exc).__name__}: {exc}")
            return Gst.FlowReturn.ERROR

        with self._lock:
            state = self._states.get(source_id)
            if state is None:
                return Gst.FlowReturn.OK
            if state.latest_version > state.submitted_version:
                state.pre_submit_replacements += 1
            state.latest_frame = frame
            state.source_frame_width = int(frame.shape[1])
            state.source_frame_height = int(frame.shape[0])
            state.latest_version += 1
            state.frame_index = self._next_frame_index_locked(source_id)
            state.source_time_seconds = source_time
            state.received_frames += 1
            state.last_frame_monotonic = now
            state.last_error = None
        return Gst.FlowReturn.OK

    @staticmethod
    def _decode_cpu_sample(
        payload: bytes,
        *,
        width: int,
        height: int,
        pixel_format: str,
    ) -> np.ndarray:
        """Normalize the GPU-converted BGRx/BGR appsink buffer for processors.

        Optimized to avoid redundant copies when the row stride matches the
        packed width (no padding).
        """
        channels = {"BGR": 3, "BGRx": 4}.get(pixel_format)
        if width <= 0 or height <= 0 or channels is None:
            raise ValueError(f"Unsupported DeepStream sample format: {pixel_format}")
        row_stride = len(payload) // height
        packed_width = width * channels
        if row_stride < packed_width:
            raise ValueError(
                f"Invalid {pixel_format} sample layout: {width}x{height}, stride={row_stride}"
            )
        flat = np.frombuffer(payload, dtype=np.uint8)
        rows = flat[: row_stride * height].reshape(height, row_stride)[:, :packed_width]
        if channels == 4:
            # BGRx → BGR: slicing creates a non-contiguous view, copy is needed
            return rows.reshape(height, width, 4)[:, :, :3].copy()
        if row_stride == packed_width:
            # No padding — the frombuffer + reshape is already contiguous
            return rows.reshape(height, width, 3)
        # Has row padding — need to strip it
        return rows.reshape(height, width, 3).copy()

    def _next_frame_index_locked(self, source_id: str) -> int:
        """Return a monotonic index that survives EOS pipeline replacement."""
        next_frame_index = self._frame_sequences.get(source_id, -1) + 1
        self._frame_sequences[source_id] = next_frame_index
        return next_frame_index

    def _open_source(self, record: SourceRecord) -> None:
        Gst, _ = self._require_runtime()
        assert record.source_uri is not None

        if not VideoFileIngestor.is_rtsp_uri(record.source_uri):
            raise ValueError(
                f"DeepStreamIngestor only accepts RTSP sources: "
                f"{record.source_uri}"
            )

        gst_uri = record.source_uri
        display_uri = VideoFileIngestor.redact_uri(record.source_uri)
        safe_id = self._safe_element_name(record.source_uri)
        pipeline = Gst.Pipeline.new(f"pipeline_{safe_id}")
        if pipeline is None:
            raise RuntimeError("Could not create a GStreamer pipeline")

        try:
            source = self._make("rtspsrc", f"source_{safe_id}")
            pacer = self._make("identity", f"pacer_{safe_id}")
            gpu_convert = self._make(
                "nvvideoconvert",
                f"gpu_convert_{safe_id}",
            )
            bgrx_caps = self._make(
                "capsfilter",
                f"bgrx_caps_{safe_id}",
            )
            decoded_sink = self._make(
                "appsink",
                f"decoded_sink_{safe_id}",
            )

            source.set_property("location", gst_uri)
            self._set_if_supported(
                source,
                "latency",
                self.rtsp_latency_ms,
            )
            self._set_if_supported(
                source,
                "drop-on-latency",
                True,
            )

            # GstRTSPLowerTrans: UDP=1, UDP_MCAST=2, TCP=4.
            self._set_if_supported(
                source,
                "protocols",
                4 if self.rtsp_transport == "tcp" else 3,
            )

            timeout_us = int(
                self.rtsp_stall_timeout_seconds * 1_000_000
            )
            if timeout_us > 0:
                self._set_if_supported(
                    source,
                    "timeout",
                    timeout_us,
                )
                self._set_if_supported(
                    source,
                    "tcp-timeout",
                    timeout_us,
                )

            pacer.set_property("sync", False)
            bgrx_caps_value = (
                "video/x-raw,format=BGRx,width=640,height=640"
            )
            bgrx_caps.set_property(
                "caps",
                Gst.Caps.from_string(bgrx_caps_value),
            )

            decoded_sink.set_property("emit-signals", True)
            decoded_sink.set_property("sync", False)
            decoded_sink.set_property("max-buffers", 1)
            decoded_sink.set_property("drop", True)
            self._set_if_supported(
                decoded_sink,
                "enable-last-sample",
                False,
            )

            for element in (
                source,
                pacer,
                gpu_convert,
                bgrx_caps,
                decoded_sink,
            ):
                pipeline.add(element)

            pad_context: dict[str, Any] = {
                "pipeline": pipeline,
                "source_id": record.source_uri,
                "safe_id": safe_id,
                "decoded_sink": decoded_sink,
                "gpu_convert": gpu_convert,
                "bgrx_caps": bgrx_caps,
                "pacer": pacer,
                "branch_created": False,
                "codec": None,
                "raw_sink": None,
                "raw_handler_id": None,
            }

            source_pad_handler_id = source.connect(
                "pad-added",
                self._on_rtsp_pad_added,
                pad_context,
            )
            decoded_sink_handler_id = decoded_sink.connect(
                "new-sample",
                self._on_new_sample,
                record.source_uri,
            )

            bus = pipeline.get_bus()
            bus.add_signal_watch()
            bus_handler_id = bus.connect(
                "message",
                self._on_bus_message,
                record.source_uri,
            )

            state = DeepStreamSourceState(
                source_id=record.source_uri,
                source_uri=record.source_uri,
                display_uri=display_uri,
                source_type="rtsp",
                pipeline=pipeline,
                source=source,
                decoded_sink=decoded_sink,
                raw_sink=None,
                bus=bus,
                bus_handler_id=bus_handler_id,
                source_pad_handler_id=source_pad_handler_id,
                decoded_sink_handler_id=decoded_sink_handler_id,
                raw_sink_handler_id=None,
                frame_width=640,
                frame_height=640,
                delivery_target_fps=record.fps,
            )

            with self._lock:
                self._states[record.source_uri] = state

            result = pipeline.set_state(Gst.State.PLAYING)
            if result == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError(
                    "GStreamer pipeline refused the PLAYING state"
                )

            with self._lock:
                self._retry_after.pop(record.source_uri, None)

        except Exception:
            with self._lock:
                state = self._states.pop(record.source_uri, None)

            if state is not None:
                self._schedule_state_disposal(state)
            else:
                pipeline.set_state(Gst.State.NULL)
                try:
                    pipeline.get_state(5 * Gst.SECOND)
                except Exception:
                    pass
            raise

    def _dispose_state(
        self,
        state: DeepStreamSourceState,
    ) -> None:
        Gst, _ = self._require_runtime()

        handlers: tuple[
            tuple[Any | None, int | None],
            ...,
        ] = (
            (
                state.decoded_sink,
                state.decoded_sink_handler_id,
            ),
            (
                state.raw_sink,
                state.raw_sink_handler_id,
            ),
            (
                state.source,
                state.source_pad_handler_id,
            ),
            (
                state.bus,
                state.bus_handler_id,
            ),
        )

        for element, handler_id in handlers:
            if element is None or handler_id is None:
                continue
            try:
                element.disconnect(handler_id)
            except Exception:
                pass

        try:
            state.bus.remove_signal_watch()
        except Exception:
            pass

        state.pipeline.set_state(Gst.State.NULL)
        try:
            state.pipeline.get_state(5 * Gst.SECOND)
        except Exception:
            LOGGER.debug(
                "GStreamer NULL-state wait failed for %s",
                state.source_id,
            )

    def _schedule_state_disposal(self, state: DeepStreamSourceState) -> None:
        source_id = state.source_id
        with self._lock:
            if source_id in self._closing_sources:
                return
            self._closing_sources.add(source_id)

        def dispose() -> None:
            try:
                self._dispose_state(state)
            finally:
                with self._lock:
                    self._closing_sources.discard(source_id)
                    current = self._close_threads.get(source_id)
                    if current is threading.current_thread():
                        self._close_threads.pop(source_id, None)

        thread = threading.Thread(
            target=dispose,
            name="deepstream-source-disposer",
            daemon=True,
        )
        with self._lock:
            self._close_threads[source_id] = thread
        thread.start()

    def _close_source(self, source_id: str) -> None:
        with self._lock:
            state = self._states.pop(source_id, None)
        if state is None:
            return
        self._schedule_state_disposal(state)

    def _join_close_threads(self, timeout: float) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            with self._lock:
                threads = list(self._close_threads.values())
            if not threads:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            for thread in threads:
                thread.join(timeout=min(remaining, 0.25))

    def _active_records(self) -> list[SourceRecord]:
        records = [
            record
            for record in self.registry.list()
            if record.enabled and self.is_supported_source(record)
            and canonical_source_type(
                record.source_uri,
                record.source_type,
            ) == self.source_type_filter
            and (
                self.rtsp_enabled
                or not VideoFileIngestor.is_rtsp_uri(record.source_uri or "")
            )
        ]
        # Enforce max_sources cap
        if len(records) > self.max_sources:
            LOGGER.warning(
                "source_type=%s sources=%d exceeds max_sources=%d; capping",
                self.source_type_filter,
                len(records),
                self.max_sources,
            )
            records = records[: self.max_sources]
        return records

    def _sync_sources(self) -> None:
        records = self._active_records()
        by_id = {record.source_uri: record for record in records}
        with self._lock:
            failed = set(self._failed_sources)
            self._failed_sources.clear()
            current_ids = set(self._states)
        for source_id in failed:
            self._reconnects += 1
            self._close_source(source_id)
        for source_id in current_ids - set(by_id):
            self._close_source(source_id)
            with self._lock:
                self._retry_after.pop(source_id, None)
                self._frame_sequences.pop(source_id, None)
                self._loop_counts.pop(source_id, None)
                self._gst_source_ids.pop(source_id, None)
        with self._lock:
            tracked_source_ids = (
                set(self._frame_sequences)
                | set(self._loop_counts)
                | set(self._gst_source_ids)
            )
            for source_id in tracked_source_ids - set(by_id):
                self._frame_sequences.pop(source_id, None)
                self._loop_counts.pop(source_id, None)
                self._gst_source_ids.pop(source_id, None)

        for record in records:
            with self._lock:
                state = self._states.get(record.source_uri)
                retry_after = self._retry_after.get(record.source_uri, 0.0)
                closing = record.source_uri in self._closing_sources
                if state is not None:
                    if state.delivery_target_fps != record.fps:
                        state.next_frame_due_monotonic = 0.0
                    state.delivery_target_fps = record.fps
            if state is not None and state.source_uri != record.source_uri:
                self._close_source(record.source_uri)
                state = None
                closing = True
            if state is not None or closing or retry_after > time.monotonic():
                continue
            try:
                self._open_source(record)
            except Exception as exc:
                self._open_failures += 1
                safe_uri = VideoFileIngestor.redact_uri(record.source_uri or "")
                safe_error = str(exc).replace(record.source_uri, safe_uri)
                self._last_error = (
                    f"Could not open {safe_uri}: "
                    f"{type(exc).__name__}: {safe_error}"
                )
                self._retry_after[record.source_uri] = (
                    time.monotonic() + self.rtsp_reconnect_seconds
                )
                LOGGER.exception("%s", self._last_error)

    def _submit_latest_round(self) -> None:
        with self._lock:
            selected = [
                state
                for state in self._states.values()
                if state.latest_frame is not None
                and state.latest_version > state.submitted_version
            ]
            # Frames are already at frame_width x frame_height from the
            # GPU capsfilter — no CPU resize needed.
            frames = [state.latest_frame for state in selected]
            source_frames = [
                state.latest_source_frame
                if state.latest_source_frame is not None
                else state.latest_frame
                for state in selected
            ]
            source_ids = [state.source_id for state in selected]
            frame_indexes = [state.frame_index for state in selected]
            source_times = [state.source_time_seconds for state in selected]
            metadata = [
                {
                    "source_uri": state.display_uri,
                    "source_type": state.source_type,
                    "frame_width": 640,
                    "frame_height": 640,
                    "source_frame_width": int(source_frame.shape[1]),
                    "source_frame_height": int(source_frame.shape[0]),
                    "source_frame": source_frame,
                    "ingest_backend": "deepstream",
                }
                for state, source_frame in zip(selected, source_frames)
            ]
            versions = {state.source_id: state.latest_version for state in selected}
        if not frames:
            return

        self._round_sequence += 1
        self.router.submit_round(
            frames=frames,  # type: ignore[arg-type]
            source_ids=source_ids,
            round_sequence=self._round_sequence,
            frame_indexes=frame_indexes,
            source_times_seconds=source_times,
            metadata=metadata,
        )
        with self._lock:
            for source_id, version in versions.items():
                state = self._states.get(source_id)
                if state is not None:
                    state.submitted_version = max(state.submitted_version, version)
                    state.submitted_frames += 1
            self._rounds_submitted += 1
            self._frames_submitted += len(frames)
            self._last_error = None

    def _iterate_glib(self) -> None:
        _, GLib = self._require_runtime()
        context = GLib.MainContext.default()
        while context.pending():
            context.iteration(False)

    def _run(self) -> None:
        self._started.set()
        submit_interval = 1.0 / self.SCHEDULER_MAX_FPS
        next_sync = 0.0
        next_submit = 0.0
        registry_revision = -1
        while not self._stop.is_set():
            now = time.monotonic()
            try:
                self._iterate_glib()
                current_revision = self.registry.revision
                if current_revision != registry_revision or now >= next_sync:
                    self._sync_sources()
                    registry_revision = self.registry.revision
                    next_sync = now + 1.0
                if now >= next_submit:
                    self._submit_latest_round()
                    next_submit = now + submit_interval
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                LOGGER.exception("DeepStream ingestion loop failed")
            self._stop.wait(min(0.05, submit_interval))

        with self._lock:
            source_ids = list(self._states)
        for source_id in source_ids:
            self._close_source(source_id)
        self._join_close_threads(timeout=10.0)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._gst, self._glib = self.gst_loader()
        missing = [
            name
            for name in self.REQUIRED_ELEMENTS
            if self._gst.ElementFactory.find(name) is None
        ]
        if missing:
            raise RuntimeError(
                "DeepStream/GStreamer elements are missing: " + ", ".join(missing)
            )
        self._stop.clear()
        self._started.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="deepstream-ingestor",
            daemon=True,
        )
        self._thread.start()
        self._started.wait(timeout=3.0)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def status(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            return {
                "enabled": True,
                "backend": "deepstream",
                "source_type_filter": self.source_type_filter,
                "max_sources": self.max_sources,
                "running": self._thread is not None and self._thread.is_alive(),
                "fps_control": "sources.fps",
                "gpu_resize_enabled": self.gpu_resize_enabled,
                "gpu_resize_active": self._gpu_resize_available,
                "loop": self.loop,
                "rtsp_enabled": self.rtsp_enabled,
                "rtsp_transport": self.rtsp_transport,
                "rtsp_latency_ms": self.rtsp_latency_ms,
                "rtsp_stall_timeout_seconds": self.rtsp_stall_timeout_seconds,
                "rounds_submitted": self._rounds_submitted,
                "frames_submitted": self._frames_submitted,
                "open_failures": self._open_failures,
                "reconnects": self._reconnects,
                "closing_sources": len(self._closing_sources),
                "last_error": self._last_error,
                "sources": {
                    source_id: {
                        "source_uri": state.display_uri,
                        "source_type": state.source_type,
                        "frame_width": state.frame_width,
                        "frame_height": state.frame_height,
                        "source_frame_width": state.source_frame_width,
                        "source_frame_height": state.source_frame_height,
                        "configured_fps": state.delivery_target_fps,
                        "fps_mode": (
                            "override"
                            if state.delivery_target_fps is not None
                            else "native"
                        ),
                        "delivery_target_fps": state.delivery_target_fps,
                        "decoded_samples": state.decoded_samples,
                        "codec": state.codec,
                        "raw_samples": state.raw_samples,
                        "raw_bytes": state.raw_bytes,
                        "last_raw_packet_age_seconds": (
                            round(
                                now
                                - state.last_raw_packet_monotonic,
                                3,
                            )
                            if state.last_raw_packet_monotonic > 0
                            else None
                        ),
                        "rate_limited_frames": state.rate_limited_frames,
                        "received_frames": state.received_frames,
                        "pre_submit_replacements": state.pre_submit_replacements,
                        "submitted_frames": state.submitted_frames,
                        "frame_index": state.frame_index,
                        "last_frame_age_seconds": (
                            round(now - state.last_frame_monotonic, 3)
                            if state.last_frame_monotonic > 0
                            else None
                        ),
                        "warnings": state.warnings,
                        "last_error": state.last_error,
                    }
                    for source_id, state in self._states.items()
                },
            }

    def restart_source(self, source_id: str) -> bool:
        """Schedule one source for a clean in-thread teardown and immediate reopen."""
        record = self.registry.get(source_id)
        if record is None or not record.enabled or not self.is_supported_source(record):
            return False
        with self._lock:
            self._failed_sources.add(source_id)
            self._retry_after[source_id] = 0.0
        return True