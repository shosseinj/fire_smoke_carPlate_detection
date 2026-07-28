from __future__ import annotations

import logging
import math
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from app.core.raw_stream_router import RawStreamRouter
import cv2
import numpy as np
from app.core.frontend_frame_worker import FrontendFrameWorker
from app.core.router import TaskRouter
from app.core.stream_demand import DemandSnapshot, StreamDemandController
from app.core.source_registry import (
    RTSP,
    STATIC_VIDEO,
    SOURCE_TYPES,
    SourceRecord,
    SourceRegistry,
    canonical_source_type,
)
from app.core.video_ingestor import VideoFileIngestor

LOGGER = logging.getLogger("uvicorn.error")

# GStreamer/NVIDIA plugin construction and teardown are process-global native
# operations.  Both runtime ingestors share this barrier so a dynamic-pad
# callback cannot mutate a pipeline while either ingestor is disposing or
# opening native objects.
_GST_LIFECYCLE_LOCK = threading.RLock()


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
    ai_valve: Any | None
    video_valve: Any | None
    bus: Any

    bus_handler_id: int | None
    source_pad_handler_id: int
    decoded_sink_handler_id: int
    raw_sink_handler_id: int | None

    frame_width: int
    frame_height: int
    delivery_target_fps: float | None
    encoded_sink: Any | None = None
    encoded_sink_handler_id: int | None = None
    generation: int = 0
    frontend_frame_index: int = -1
    frontend_next_frame_due_monotonic: float = 0.0
    frontend_rate_limited_frames: int = 0
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
    pipeline_state: str = "NULL"
    pending_state: str = "VOID_PENDING"
    last_bus_message: str | None = None
    last_bus_element: str | None = None
    last_bus_message_monotonic: float = 0.0
    last_pad_caps: str | None = None
    raw_encoded_packets: int = 0
    position_seconds: float | None = None
    duration_seconds: float | None = None
    seek_failures: int = 0
    paused_for_demand: bool = False


class DeepStreamIngestor:
    """Uses DeepStream/NVDEC for file and RTSP decoding, then feeds the router."""

    SCHEDULER_MAX_FPS = 240.0

    REQUIRED_ELEMENTS = (
        "rtspsrc",
        "filesrc",
        "qtdemux",
        "matroskademux",
        "rtph264depay",
        "rtph265depay",
        "h264parse",
        "h265parse",
        "tee",
        "queue",
        "valve",
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
        demand_controller: StreamDemandController | None = None,
        video_only_mode: bool = False,
        max_sources: int = 256,
        rtsp_enabled: bool = True,
        rtsp_transport: str = "tcp",
        rtsp_latency_ms: int = 500,
        rtsp_reconnect_seconds: float = 3.0,
        rtsp_stall_timeout_seconds: int = 30,
        skip_taskless_sources: bool = True,
        gst_loader: Callable[[], tuple[Any, Any]] = _load_gstreamer,
        max_active_sources: int | None = None,
        source_open_stagger_seconds: float = 0.5,
        source_allowlist: tuple[str, ...] | None = None,
        max_source_opens_per_sync: int = 1,
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
        self.demand_controller = demand_controller
        self.video_only_mode = bool(video_only_mode)
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
        self.max_active_sources = (
            self.max_sources
            if max_active_sources is None
            else max(1, int(max_active_sources))
        )
        self.source_open_stagger_seconds = max(
            0.0, float(source_open_stagger_seconds)
        )
        self.source_allowlist = frozenset(source_allowlist or ())
        self.max_source_opens_per_sync = max(
            1, int(max_source_opens_per_sync)
        )

        self._gst: Any | None = None
        self._glib: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started = threading.Event()
        self._lock = threading.RLock()
        self._states: dict[str, DeepStreamSourceState] = {}
        self._closing_sources: set[str] = set()
        self._close_threads: dict[str, threading.Thread] = {}
        self._commands: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()
        self._owner_thread_id: int | None = None
        self._generation_sequence = 0
        self._next_source_open_monotonic = 0.0
        self._bus_history: dict[str, deque[dict[str, Any]]] = {}
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
        self._last_selection_skip_reasons: dict[str, str] = {}
        self._source_open_errors: dict[str, str] = {}
        self._source_skip_codes: dict[str, str] = {}
        self._demand_unsubscribe: Callable[[], None] | None = None
        if self.demand_controller is not None:
            self._demand_unsubscribe = self.demand_controller.add_listener(
                self._on_demand_changed
            )

    def _video_required(self, source_id: str | None = None) -> bool:
        return (
            self.frontend_frame_worker is not None
            and (
                self.demand_controller is None
                or self.demand_controller.video_required(source_id)
            )
        )

    def _ai_required(self) -> bool:
        return (
            not self.video_only_mode
            and (
                self.demand_controller is None
                or self.demand_controller.ai_required()
            )
        )

    def _on_demand_changed(self, _snapshot: DemandSnapshot) -> None:
        """Wake the owner thread; never mutate GStreamer from subscriber threads."""
        if (
            (self._video_required() or self._ai_required())
            and (self._thread is None or not self._thread.is_alive())
        ):
            self.start()
        self._commands.put(("demand_changed", None))

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

    def _configure_nonblocking_queue(self, element: Any, buffers: int) -> None:
        element.set_property("leaky", 2)
        element.set_property("max-size-buffers", buffers)
        element.set_property("max-size-bytes", 0)
        element.set_property("max-size-time", 0)

    def _build_decoded_output_branches(
        self,
        *,
        pipeline: Any,
        source_id: str,
        safe_id: str,
        decoder: Any,
        generation: int,
    ) -> dict[str, Any]:
        """Attach common original-resolution frontend and optional AI branches."""
        Gst, _ = self._require_runtime()
        decoded_tee = self._make("tee", f"decoded_tee_{safe_id}_{generation}")
        video_valve = self._make("valve", f"video_valve_{safe_id}_{generation}")
        frontend_queue = self._make("queue", f"frontend_queue_{safe_id}_{generation}")
        frontend_convert = self._make(
            "nvvideoconvert", f"frontend_convert_{safe_id}_{generation}"
        )
        frontend_caps = self._make(
            "capsfilter", f"frontend_caps_{safe_id}_{generation}"
        )
        frontend_sink = self._make(
            "appsink", f"frontend_sink_{safe_id}_{generation}"
        )
        video_valve.set_property("drop", not self._video_required(source_id))
        self._configure_nonblocking_queue(frontend_queue, 2)
        frontend_caps.set_property(
            "caps", Gst.Caps.from_string("video/x-raw,format=BGRx")
        )
        record = self.registry.get(source_id)
        static_source = bool(
            record is not None
            and canonical_source_type(
                record.source_uri, record.source_type
            )
            == STATIC_VIDEO
        )
        for name, value in (
            ("emit-signals", True),
            ("sync", static_source),
            ("max-buffers", 1),
            ("drop", True),
        ):
            frontend_sink.set_property(name, value)
        self._set_if_supported(frontend_sink, "enable-last-sample", False)

        elements = [
            decoded_tee,
            video_valve,
            frontend_queue,
            frontend_convert,
            frontend_caps,
            frontend_sink,
        ]
        ai_valve = None
        decoded_sink = None
        decoded_sink_handler_id = 0
        if not self.video_only_mode:
            ai_valve = self._make("valve", f"ai_valve_{safe_id}_{generation}")
            ai_queue = self._make("queue", f"ai_queue_{safe_id}_{generation}")
            pacer = self._make("identity", f"ai_pacer_{safe_id}_{generation}")
            gpu_convert = self._make(
                "nvvideoconvert", f"ai_convert_{safe_id}_{generation}"
            )
            ai_caps = self._make("capsfilter", f"ai_caps_{safe_id}_{generation}")
            decoded_sink = self._make(
                "appsink", f"ai_sink_{safe_id}_{generation}"
            )
            ai_valve.set_property("drop", not self._ai_required())
            self._configure_nonblocking_queue(ai_queue, 2)
            pacer.set_property("sync", False)
            ai_caps.set_property(
                "caps",
                Gst.Caps.from_string(
                    "video/x-raw,format=BGRx,width=640,height=640"
                ),
            )
            for name, value in (
                ("emit-signals", True),
                ("sync", False),
                ("max-buffers", 1),
                ("drop", True),
            ):
                decoded_sink.set_property(name, value)
            self._set_if_supported(decoded_sink, "enable-last-sample", False)
            elements.extend(
                [ai_valve, ai_queue, pacer, gpu_convert, ai_caps, decoded_sink]
            )

        for element in elements:
            pipeline.add(element)
        if not decoder.link(decoded_tee):
            raise RuntimeError("Could not link NVDEC to decoded tee")
        if not decoded_tee.link(video_valve):
            raise RuntimeError("Could not link decoded tee to frontend valve")
        if not video_valve.link(frontend_queue):
            raise RuntimeError("Could not link frontend valve to queue")
        if not frontend_queue.link(frontend_convert):
            raise RuntimeError("Could not link frontend queue to converter")
        if not frontend_convert.link(frontend_caps):
            raise RuntimeError("Could not link frontend converter to BGRx caps")
        if not frontend_caps.link(frontend_sink):
            raise RuntimeError("Could not link frontend caps to appsink")
        if ai_valve is not None and decoded_sink is not None:
            if not decoded_tee.link(ai_valve):
                raise RuntimeError("Could not link decoded tee to AI valve")
            if not ai_valve.link(ai_queue):
                raise RuntimeError("Could not link AI valve to queue")
            if not ai_queue.link(pacer):
                raise RuntimeError("Could not link AI queue to pacer")
            if not pacer.link(gpu_convert):
                raise RuntimeError("Could not link AI pacer to converter")
            if not gpu_convert.link(ai_caps):
                raise RuntimeError("Could not link AI converter to caps")
            if not ai_caps.link(decoded_sink):
                raise RuntimeError("Could not link AI caps to appsink")
            decoded_sink_handler_id = decoded_sink.connect(
                "new-sample", self._on_new_sample, source_id, generation
            )
        frontend_handler_id = frontend_sink.connect(
            "new-sample", self._on_frontend_sample, source_id, generation
        )
        return {
            "frontend_sink": frontend_sink,
            "frontend_handler_id": frontend_handler_id,
            "video_valve": video_valve,
            "ai_valve": ai_valve,
            "decoded_sink": decoded_sink,
            "decoded_sink_handler_id": decoded_sink_handler_id,
            "elements": elements,
        }

    def _build_encoded_branches(
        self,
        *,
        pipeline: Any,
        source_id: str,
        safe_id: str,
        codec: str,
        generation: int,
        rtp: bool,
    ) -> dict[str, Any]:
        """Build parser, optional encoded packet tee, NVDEC and decoded outputs."""
        Gst, _ = self._require_runtime()

        if codec == "h264":
            depay = (
                self._make("rtph264depay", f"h264_depay_{safe_id}_{generation}")
                if rtp
                else None
            )
            parser = self._make("h264parse", f"h264_parser_{safe_id}_{generation}")
            encoded_caps_value = (
                "video/x-h264,stream-format=byte-stream,alignment=au"
            )
        elif codec == "h265":
            depay = (
                self._make("rtph265depay", f"h265_depay_{safe_id}_{generation}")
                if rtp
                else None
            )
            parser = self._make("h265parse", f"h265_parser_{safe_id}_{generation}")
            encoded_caps_value = (
                "video/x-h265,stream-format=byte-stream,alignment=au"
            )
        else:
            raise ValueError(f"Unsupported codec: {codec}")

        self._set_if_supported(parser, "config-interval", -1)

        encoded_caps = self._make(
            "capsfilter", f"encoded_caps_{safe_id}_{generation}"
        )
        encoded_caps.set_property(
            "caps",
            Gst.Caps.from_string(encoded_caps_value),
        )

        decode_queue = self._make("queue", f"decode_queue_{safe_id}_{generation}")
        decoder = self._make("nvv4l2decoder", f"decoder_{safe_id}_{generation}")
        self._configure_nonblocking_queue(decode_queue, 4)
        record = self.registry.get(source_id)
        stream_pacer = None
        if (
            record is not None
            and canonical_source_type(record.source_uri, record.source_type)
            == STATIC_VIDEO
        ):
            stream_pacer = self._make(
                "identity", f"stream_pacer_{safe_id}_{generation}"
            )
            stream_pacer.set_property("sync", True)
        elements = [parser, encoded_caps, decode_queue, decoder]
        if stream_pacer is not None:
            elements.insert(2, stream_pacer)
        if depay is not None:
            elements.insert(0, depay)
        encoded_sink = None
        encoded_handler_id = None
        encoded_tee = None
        if self.raw_stream_router is not None and self.raw_stream_router.enabled:
            encoded_tee = self._make(
                "tee", f"encoded_tee_{safe_id}_{generation}"
            )
            encoded_queue = self._make(
                "queue", f"encoded_queue_{safe_id}_{generation}"
            )
            encoded_sink = self._make(
                "appsink", f"encoded_sink_{safe_id}_{generation}"
            )
            self._configure_nonblocking_queue(encoded_queue, 8)
            for name, value in (
                ("emit-signals", True),
                ("sync", False),
                ("max-buffers", 8),
                ("drop", True),
            ):
                encoded_sink.set_property(name, value)
            elements.extend([encoded_tee, encoded_queue, encoded_sink])
        for element in elements:
            pipeline.add(element)
        if depay is not None and not depay.link(parser):
            raise RuntimeError("Could not link depayloader to parser")
        if not parser.link(encoded_caps):
            raise RuntimeError("Could not link parser to encoded caps")
        encoded_output = stream_pacer or encoded_caps
        if stream_pacer is not None and not encoded_caps.link(stream_pacer):
            raise RuntimeError("Could not link static parser caps to pacer")
        if encoded_tee is None:
            if not encoded_output.link(decode_queue):
                raise RuntimeError("Could not link encoded caps to decode queue")
        else:
            if not encoded_output.link(encoded_tee):
                raise RuntimeError("Could not link parser caps to encoded tee")
            if not encoded_tee.link(decode_queue):
                raise RuntimeError("Could not link encoded tee to decode queue")
            if not encoded_tee.link(encoded_queue):
                raise RuntimeError("Could not link encoded tee to packet queue")
            if not encoded_queue.link(encoded_sink):
                raise RuntimeError("Could not link encoded queue to appsink")
            encoded_handler_id = encoded_sink.connect(
                "new-sample",
                self._on_encoded_sample,
                source_id,
                codec,
                generation,
            )
        if not decode_queue.link(decoder):
            raise RuntimeError("Could not link decode queue to NVDEC")
        outputs = self._build_decoded_output_branches(
            pipeline=pipeline,
            source_id=source_id,
            safe_id=safe_id,
            decoder=decoder,
            generation=generation,
        )
        all_elements = elements + outputs["elements"]
        for element in all_elements:
            if not element.sync_state_with_parent():
                raise RuntimeError(
                    f"Could not sync GStreamer element state: {element.get_name()}"
                )
        return {
            "depay": depay,
            "parser": parser,
            "raw_sink": outputs["frontend_sink"],
            "raw_handler_id": outputs["frontend_handler_id"],
            "video_valve": outputs["video_valve"],
            "ai_valve": outputs["ai_valve"],
            "decoded_sink": outputs["decoded_sink"],
            "decoded_sink_handler_id": outputs["decoded_sink_handler_id"],
            "encoded_sink": encoded_sink,
            "encoded_handler_id": encoded_handler_id,
            "codec": codec,
        }

    def _on_rtsp_pad_added(
        self,
        _: Any,
        pad: Any,
        context: dict[str, Any],
    ) -> None:
        with _GST_LIFECYCLE_LOCK:
            self._on_rtsp_pad_added_locked(pad, context)

    def _on_rtsp_pad_added_locked(
        self, pad: Any, context: dict[str, Any]
    ) -> None:
        Gst, _ = self._require_runtime()

        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            return
        caps_text = caps.to_string()
        LOGGER.info(
            "GST_PAD_CAPS source=%s source_type=rtsp generation=%s caps=%s",
            VideoFileIngestor.redact_uri(context["source_id"]),
            context["generation"],
            caps_text,
        )
        if not self._callback_is_current(
            context["source_id"], context["generation"]
        ):
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

        with self._lock:
            if context["branch_created"]:
                return
            context["branch_created"] = True

        try:
            branch = self._build_encoded_branches(
                pipeline=context["pipeline"],
                source_id=context["source_id"],
                safe_id=context["safe_id"],
                codec=codec,
                generation=context["generation"],
                rtp=True,
            )

            sink_pad = branch["depay"].get_static_pad("sink")
            if sink_pad is None:
                raise RuntimeError("Depayloader sink pad is unavailable")

            result = pad.link(sink_pad)
            if result != Gst.PadLinkReturn.OK:
                raise RuntimeError(
                    f"Could not link RTSP pad to {codec} depayloader: {result}"
                )

            context["codec"] = codec
            context["raw_sink"] = branch["raw_sink"]
            context["raw_handler_id"] = branch["raw_handler_id"]
            context["video_valve"] = branch["video_valve"]

            with self._lock:
                state = self._states.get(context["source_id"])
                if state is not None and state.generation == context["generation"]:
                    state.codec = codec
                    state.raw_sink = branch["raw_sink"]
                    state.raw_sink_handler_id = branch["raw_handler_id"]
                    state.video_valve = branch["video_valve"]
                    state.ai_valve = branch["ai_valve"]
                    state.decoded_sink = branch["decoded_sink"]
                    state.decoded_sink_handler_id = branch[
                        "decoded_sink_handler_id"
                    ]
                    state.encoded_sink = branch["encoded_sink"]
                    state.encoded_sink_handler_id = branch["encoded_handler_id"]
                    state.last_pad_caps = caps_text

            LOGGER.info(
                "DeepStream encoded branch ready source=%s codec=%s",
                VideoFileIngestor.redact_uri(context["source_id"]),
                codec,
            )
        except Exception as exc:
            context["branch_created"] = False
            self._mark_failed(
                context["source_id"],
                f"Could not create encoded RTSP branch: "
                f"{type(exc).__name__}: {exc}",
            )

    def _callback_is_current(
        self,
        source_id: str,
        generation: int,
        *,
        bus: Any | None = None,
        sink: Any | None = None,
    ) -> bool:
        with self._lock:
            state = self._states.get(source_id)
            return bool(
                state is not None
                and state.generation == generation
                and (bus is None or state.bus is bus)
                and (
                    sink is None
                    or state.raw_sink is sink
                    or state.decoded_sink is sink
                )
            )

    def _on_frontend_sample(
        self,
        sink: Any,
        source_id: str,
        generation: int | None = None,
    ) -> Any:
        """Deliver one original-resolution decoded frame to the frontend."""
        Gst, _ = self._require_runtime()
        if generation is None:
            with self._lock:
                current = self._states.get(source_id)
                generation = 0 if current is None else current.generation
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        if (
            not self._video_required(source_id)
            or not self._callback_is_current(
                source_id, generation, sink=sink
            )
        ):
            return Gst.FlowReturn.OK

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

                if state is None or state.generation != generation:
                    return Gst.FlowReturn.OK

                now_monotonic = time.monotonic()
                if state.delivery_target_fps is not None:
                    period = 1.0 / state.delivery_target_fps
                    tolerance = min(0.004, period * 0.10)
                    if (
                        state.frontend_next_frame_due_monotonic > 0.0
                        and now_monotonic + tolerance
                        < state.frontend_next_frame_due_monotonic
                    ):
                        state.frontend_rate_limited_frames += 1
                        return Gst.FlowReturn.OK
                    if state.frontend_next_frame_due_monotonic <= 0.0:
                        state.frontend_next_frame_due_monotonic = (
                            now_monotonic + period
                        )
                    else:
                        periods = max(
                            1,
                            math.floor(
                                max(
                                    0.0,
                                    now_monotonic
                                    - state.frontend_next_frame_due_monotonic,
                                )
                                / period
                            )
                            + 1,
                        )
                        state.frontend_next_frame_due_monotonic += (
                            periods * period
                        )

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
                    source_type=state.source_type,
                    ingest_backend="deepstream",
                )



        except Exception:
            LOGGER.exception(
                "Raw decoded-frame delivery failed source=%s",
                VideoFileIngestor.redact_uri(source_id),
            )
            return Gst.FlowReturn.ERROR

        return Gst.FlowReturn.OK

    # Compatibility alias for older tests/callers.
    _on_raw_sample = _on_frontend_sample

    def _on_encoded_sample(
        self,
        sink: Any,
        source_id: str,
        codec: str,
        generation: int,
    ) -> Any:
        Gst, _ = self._require_runtime()
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        if not self._callback_is_current(source_id, generation):
            return Gst.FlowReturn.OK
        buffer = sample.get_buffer()
        if buffer is None:
            return Gst.FlowReturn.ERROR
        payload = buffer.extract_dup(0, buffer.get_size())
        flags = buffer.get_flags()
        is_keyframe = not bool(flags & Gst.BufferFlags.DELTA_UNIT)
        if self.raw_stream_router is not None:
            self.raw_stream_router.publish_packet(
                source_id=source_id,
                codec=codec,
                pts_ns=None if buffer.pts == Gst.CLOCK_TIME_NONE else int(buffer.pts),
                dts_ns=None if buffer.dts == Gst.CLOCK_TIME_NONE else int(buffer.dts),
                duration_ns=(
                    None
                    if buffer.duration == Gst.CLOCK_TIME_NONE
                    else int(buffer.duration)
                ),
                is_keyframe=is_keyframe,
                data=payload,
            )
        with self._lock:
            state = self._states.get(source_id)
            if state is not None and state.generation == generation:
                state.raw_encoded_packets += 1
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
            self._source_open_errors[source_id] = safe_message
            self._failed_sources.add(source_id)
            self._retry_after[source_id] = (
                time.monotonic()
                if expected
                else time.monotonic() + self.rtsp_reconnect_seconds
            )
        log = LOGGER.info if expected else LOGGER.error
        log("DeepStream source %s: %s", display_uri, safe_message)

    def _on_bus_message(
        self,
        bus: Any,
        message: Any,
        source_id: str,
        generation: int | None = None,
    ) -> None:
        Gst, _ = self._require_runtime()
        if generation is None:
            with self._lock:
                current = self._states.get(source_id)
                generation = 0 if current is None else current.generation
        with self._lock:
            state = self._states.get(source_id)
            if (
                state is None
                or state.generation != generation
                or state.bus is not bus
            ):
                return
            pipeline = state.pipeline
            display_uri = state.display_uri
            source_type = state.source_type
            codec = state.codec
        message_source = getattr(message, "src", None)
        element_name = (
            message_source.get_name()
            if message_source is not None
            else "<none>"
        )
        factory = "<none>"
        if message_source is not None:
            try:
                factory_object = message_source.get_factory()
                if factory_object is not None:
                    factory = factory_object.get_name()
            except Exception:
                factory = "<unknown>"
        type_numeric = int(message.type)
        try:
            type_name = Gst.MessageType.get_name(message.type)
        except Exception:
            type_name = str(message.type)
        current_state = "UNKNOWN"
        pending_state = "UNKNOWN"
        try:
            _, current, pending = pipeline.get_state(0)
            current_state = current.value_nick
            pending_state = pending.value_nick
        except Exception:
            pass
        now_monotonic = time.monotonic()
        event = {
            "timestamp_monotonic": now_monotonic,
            "message_type": type_name,
            "message_type_numeric": type_numeric,
            "element": element_name,
            "factory": factory,
            "pipeline_state": current_state,
            "pending_state": pending_state,
            "generation": generation,
            "codec": codec,
        }
        with self._lock:
            state = self._states.get(source_id)
            if state is None or state.generation != generation:
                return
            state.last_bus_message = type_name
            state.last_bus_element = element_name
            state.last_bus_message_monotonic = now_monotonic
            state.pipeline_state = current_state
            state.pending_state = pending_state
            history = self._bus_history.setdefault(source_id, deque(maxlen=30))
            history.append(event)
        noisy_state_change = (
            message.type == Gst.MessageType.STATE_CHANGED
            and message_source is not pipeline
            and factory
            not in {"nvv4l2decoder", "rtspsrc", "qtdemux", "matroskademux"}
        )
        if not noisy_state_change:
            LOGGER.info(
                "GST_BUS_MESSAGE_BEGIN source=%s source_type=%s pipeline=%s "
                "message_type=%s message_type_name=%s element=%s factory=%s "
                "pipeline_state=%s pending_state=%s monotonic=%.6f thread=%s "
                "codec=%s generation=%s",
                display_uri,
                source_type,
                pipeline.get_name(),
                type_numeric,
                type_name,
                element_name,
                factory,
                current_state,
                pending_state,
                now_monotonic,
                threading.current_thread().name,
                codec,
                generation,
            )
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            detail = str(error)
            if debug:
                detail = f"{detail} ({debug})"
            self._mark_failed(source_id, detail)
        elif message.type == Gst.MessageType.EOS:
            if self._should_loop_source(source_id):
                LOGGER.info(
                    "DeepStream file reached EOS; seek scheduled: %s",
                    VideoFileIngestor.redact_uri(source_id),
                )
                self._commands.put(("seek_loop", (source_id, generation)))
            else:
                LOGGER.info(
                    "DeepStream file reached EOS: %s",
                    VideoFileIngestor.redact_uri(source_id),
                )
                with self._lock:
                    self._failed_sources.add(source_id)
                    self._retry_after[source_id] = float("inf")
        elif message.type == Gst.MessageType.WARNING:
            warning, debug = message.parse_warning()
            safe_warning = self._redact_error(source_id, str(warning))
            if debug:
                safe_warning = (
                    f"{safe_warning} ({self._redact_error(source_id, debug)})"
                )
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

    def _on_new_sample(
        self, sink: Any, source_id: str, generation: int | None = None
    ) -> Any:
        Gst, _ = self._require_runtime()
        if generation is None:
            with self._lock:
                current = self._states.get(source_id)
                generation = 0 if current is None else current.generation
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        if not self._ai_required() or not self.router.task_processing_enabled():
            return Gst.FlowReturn.OK

        now = time.monotonic()
        with self._lock:
            state = self._states.get(source_id)
            if (
                state is None
                or state.generation != generation
                or state.decoded_sink is not sink
            ):
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

    def _legacy_open_source(self, record: SourceRecord) -> None:
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
            ai_valve = self._make("valve", f"ai_valve_{safe_id}")
            ai_valve.set_property("drop", not self._ai_required())
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
                ai_valve,
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
                "ai_valve": ai_valve,
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
                encoded_sink=None,
                ai_valve=ai_valve,
                video_valve=None,
                bus=bus,
                bus_handler_id=bus_handler_id,
                source_pad_handler_id=source_pad_handler_id,
                decoded_sink_handler_id=decoded_sink_handler_id,
                raw_sink_handler_id=None,
                encoded_sink_handler_id=None,
                generation=0,
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

    def _next_generation(self) -> int:
        with self._lock:
            self._generation_sequence += 1
            return self._generation_sequence

    def _new_pipeline_state(
        self,
        *,
        record: SourceRecord,
        pipeline: Any,
        source: Any,
        source_pad_handler_id: int,
        generation: int,
    ) -> DeepStreamSourceState:
        bus = pipeline.get_bus()
        return DeepStreamSourceState(
            source_id=record.source_uri,
            source_uri=record.source_uri,
            display_uri=VideoFileIngestor.redact_uri(record.source_uri),
            source_type=canonical_source_type(
                record.source_uri, record.source_type
            ),
            pipeline=pipeline,
            source=source,
            decoded_sink=None,
            raw_sink=None,
            encoded_sink=None,
            ai_valve=None,
            video_valve=None,
            bus=bus,
            bus_handler_id=None,
            source_pad_handler_id=source_pad_handler_id,
            decoded_sink_handler_id=0,
            raw_sink_handler_id=None,
            encoded_sink_handler_id=None,
            generation=generation,
            frame_width=640,
            frame_height=640,
            delivery_target_fps=record.fps,
        )

    def _open_source(self, record: SourceRecord) -> None:
        source_type = canonical_source_type(
            record.source_uri, record.source_type
        )
        if source_type == RTSP:
            self._open_rtsp_source(record)
            return
        if source_type == STATIC_VIDEO:
            self._open_static_source(record)
            return
        raise ValueError(f"Unsupported DeepStream source type: {source_type}")

    def _open_rtsp_source(self, record: SourceRecord) -> None:
        with _GST_LIFECYCLE_LOCK:
            self._open_rtsp_source_locked(record)

    def _open_rtsp_source_locked(self, record: SourceRecord) -> None:
        Gst, _ = self._require_runtime()
        generation = self._next_generation()
        safe_id = self._safe_element_name(record.source_uri)
        pipeline = Gst.Pipeline.new(f"rtsp_pipeline_{safe_id}_{generation}")
        if pipeline is None:
            raise RuntimeError("Could not create RTSP GStreamer pipeline")
        source = self._make("rtspsrc", f"rtsp_source_{safe_id}_{generation}")
        source.set_property("location", record.source_uri)
        self._set_if_supported(source, "latency", self.rtsp_latency_ms)
        self._set_if_supported(source, "drop-on-latency", True)
        self._set_if_supported(
            source, "protocols", 4 if self.rtsp_transport == "tcp" else 3
        )
        timeout_us = int(self.rtsp_stall_timeout_seconds * 1_000_000)
        if timeout_us > 0:
            self._set_if_supported(source, "timeout", timeout_us)
            self._set_if_supported(source, "tcp-timeout", timeout_us)
        pipeline.add(source)
        context = {
            "pipeline": pipeline,
            "source_id": record.source_uri,
            "safe_id": safe_id,
            "generation": generation,
            "branch_created": False,
            "codec": None,
            "raw_sink": None,
            "raw_handler_id": None,
            "video_valve": None,
        }
        source_pad_handler_id = source.connect(
            "pad-added", self._on_rtsp_pad_added, context
        )
        state = self._new_pipeline_state(
            record=record,
            pipeline=pipeline,
            source=source,
            source_pad_handler_id=source_pad_handler_id,
            generation=generation,
        )
        with self._lock:
            self._states[record.source_uri] = state
        result = pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            with self._lock:
                self._states.pop(record.source_uri, None)
            self._dispose_state(state)
            raise RuntimeError("RTSP pipeline refused PLAYING")
        state.pipeline_state = "PLAYING"
        with self._lock:
            self._retry_after.pop(record.source_uri, None)

    def _on_static_demux_pad_added(
        self, _: Any, pad: Any, context: dict[str, Any]
    ) -> None:
        with _GST_LIFECYCLE_LOCK:
            self._on_static_demux_pad_added_locked(pad, context)

    def _on_static_demux_pad_added_locked(
        self, pad: Any, context: dict[str, Any]
    ) -> None:
        Gst, _ = self._require_runtime()
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            return
        caps_text = caps.to_string()
        LOGGER.info(
            "GST_PAD_CAPS source=%s source_type=static_video generation=%s caps=%s",
            context["display_uri"],
            context["generation"],
            caps_text,
        )
        if not self._callback_is_current(
            context["source_id"], context["generation"]
        ):
            return
        structure_name = caps.get_structure(0).get_name()
        if structure_name == "video/x-h264":
            codec = "h264"
        elif structure_name in {"video/x-h265", "video/x-hevc"}:
            codec = "h265"
        else:
            if structure_name.startswith("video/"):
                with self._lock:
                    self._source_skip_codes[
                        context["source_id"]
                    ] = "unsupported_codec_or_container"
                self._mark_failed(
                    context["source_id"],
                    f"unsupported codec/container caps={caps_text}",
                )
            return
        with self._lock:
            if context["branch_created"]:
                return
            context["branch_created"] = True
        try:
            branch = self._build_encoded_branches(
                pipeline=context["pipeline"],
                source_id=context["source_id"],
                safe_id=context["safe_id"],
                codec=codec,
                generation=context["generation"],
                rtp=False,
            )
            parser_sink = branch["parser"].get_static_pad("sink")
            if parser_sink is None:
                raise RuntimeError("Static parser sink pad unavailable")
            result = pad.link(parser_sink)
            if result != Gst.PadLinkReturn.OK:
                raise RuntimeError(
                    f"Could not link static demux pad caps={caps_text}: {result}"
                )
            with self._lock:
                state = self._states.get(context["source_id"])
                if state is None or state.generation != context["generation"]:
                    return
                state.codec = codec
                state.raw_sink = branch["raw_sink"]
                state.raw_sink_handler_id = branch["raw_handler_id"]
                state.video_valve = branch["video_valve"]
                state.ai_valve = branch["ai_valve"]
                state.decoded_sink = branch["decoded_sink"]
                state.decoded_sink_handler_id = branch[
                    "decoded_sink_handler_id"
                ]
                state.encoded_sink = branch["encoded_sink"]
                state.encoded_sink_handler_id = branch["encoded_handler_id"]
                state.last_pad_caps = caps_text
        except Exception as exc:
            context["branch_created"] = False
            self._mark_failed(
                context["source_id"],
                f"Could not build static encoded branch: "
                f"{type(exc).__name__}: {exc}",
            )

    def _open_static_source(self, record: SourceRecord) -> None:
        with _GST_LIFECYCLE_LOCK:
            self._open_static_source_locked(record)

    def _open_static_source_locked(self, record: SourceRecord) -> None:
        Gst, _ = self._require_runtime()
        generation = self._next_generation()
        safe_id = self._safe_element_name(record.source_uri)
        resolved = Path(record.source_uri).expanduser()
        if not resolved.is_absolute():
            resolved = self.project_root / resolved
        resolved = resolved.resolve()
        suffix = resolved.suffix.lower()
        if suffix in {".mp4", ".mov", ".m4v"}:
            demux_factory = "qtdemux"
        elif suffix in {".mkv", ".webm"}:
            demux_factory = "matroskademux"
        else:
            raise ValueError(
                f"Unsupported controlled static container: {suffix or '<none>'}"
            )
        pipeline = Gst.Pipeline.new(f"static_pipeline_{safe_id}_{generation}")
        if pipeline is None:
            raise RuntimeError("Could not create static GStreamer pipeline")
        source = self._make("filesrc", f"file_source_{safe_id}_{generation}")
        demux = self._make(demux_factory, f"demux_{safe_id}_{generation}")
        source.set_property("location", str(resolved))
        pipeline.add(source)
        pipeline.add(demux)
        if not source.link(demux):
            raise RuntimeError(f"Could not link filesrc to {demux_factory}")
        context = {
            "pipeline": pipeline,
            "source_id": record.source_uri,
            "display_uri": record.source_uri,
            "safe_id": safe_id,
            "generation": generation,
            "branch_created": False,
        }
        source_pad_handler_id = demux.connect(
            "pad-added", self._on_static_demux_pad_added, context
        )
        state = self._new_pipeline_state(
            record=record,
            pipeline=pipeline,
            source=demux,
            source_pad_handler_id=source_pad_handler_id,
            generation=generation,
        )
        with self._lock:
            self._states[record.source_uri] = state
        result = pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            with self._lock:
                self._states.pop(record.source_uri, None)
            self._dispose_state(state)
            raise RuntimeError("Static pipeline refused PLAYING")
        state.pipeline_state = "PLAYING"
        with self._lock:
            self._retry_after.pop(record.source_uri, None)

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
                state.encoded_sink,
                state.encoded_sink_handler_id,
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

        if state.bus_handler_id is not None:
            try:
                state.bus.remove_signal_watch()
            except Exception:
                pass

        state.pipeline.set_state(Gst.State.NULL)
        try:
            state.pipeline.get_state(1 * Gst.SECOND)
        except Exception:
            LOGGER.warning(
                "GStreamer NULL-state wait timed out source=%s generation=%s",
                state.display_uri,
                state.generation,
            )

    def _schedule_state_disposal(self, state: DeepStreamSourceState) -> None:
        if threading.get_ident() == self._owner_thread_id:
            self._dispose_state(state)
            with self._lock:
                self._closing_sources.discard(state.source_id)
            return
        self._commands.put(("dispose_state", state))

    def _close_source(self, source_id: str) -> None:
        if (
            self._owner_thread_id is not None
            and threading.get_ident() != self._owner_thread_id
        ):
            self._commands.put(("close_source", source_id))
            return
        self._close_source_owner(source_id)

    def _close_source_owner(self, source_id: str) -> None:
        with _GST_LIFECYCLE_LOCK:
            with self._lock:
                state = self._states.pop(source_id, None)
                if state is not None:
                    self._closing_sources.add(source_id)
            if state is None:
                return
            try:
                self._dispose_state(state)
            finally:
                with self._lock:
                    self._closing_sources.discard(source_id)

    def _join_close_threads(self, timeout: float) -> None:
        return

    def _apply_demand_state(self) -> None:
        Gst, _ = self._require_runtime()
        with self._lock:
            states = list(self._states.values())
        for state in states:
            video_required = self._video_required(state.source_id)
            if state.video_valve is not None:
                state.video_valve.set_property("drop", not video_required)
            if state.ai_valve is not None:
                state.ai_valve.set_property("drop", not self._ai_required())
            if state.source_type == STATIC_VIDEO:
                target = Gst.State.PLAYING if video_required or self._ai_required() else Gst.State.PAUSED
                if (
                    target == Gst.State.PAUSED
                    and not state.paused_for_demand
                ):
                    state.pipeline.set_state(target)
                    state.paused_for_demand = True
                    state.pipeline_state = "PAUSED"
                elif target == Gst.State.PLAYING and state.paused_for_demand:
                    state.pipeline.set_state(target)
                    state.paused_for_demand = False
                    state.pipeline_state = "PLAYING"

    def _seek_static_loop(self, source_id: str, generation: int) -> None:
        Gst, _ = self._require_runtime()
        with self._lock:
            state = self._states.get(source_id)
        if (
            state is None
            or state.generation != generation
            or state.source_type != STATIC_VIDEO
        ):
            return
        ok = state.pipeline.seek_simple(
            Gst.Format.TIME,
            Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
            0,
        )
        if ok:
            with self._lock:
                self._loop_counts[source_id] = (
                    self._loop_counts.get(source_id, 0) + 1
                )
            state.pipeline.set_state(
                Gst.State.PLAYING
                if self._video_required(source_id) or self._ai_required()
                else Gst.State.PAUSED
            )
        else:
            state.seek_failures += 1
            self._mark_failed(
                source_id,
                "Static EOS seek failed; controlled restart scheduled",
                expected=True,
            )

    def _process_commands(self) -> None:
        while True:
            try:
                command, payload = self._commands.get_nowait()
            except queue.Empty:
                return
            if command == "demand_changed":
                self._apply_demand_state()
            elif command == "close_source":
                self._close_source_owner(str(payload))
            elif command == "dispose_state":
                self._dispose_state(payload)
            elif command == "seek_loop":
                source_id, generation = payload
                self._seek_static_loop(source_id, generation)

    def _update_static_positions(self) -> None:
        Gst, _ = self._require_runtime()
        with self._lock:
            states = [
                state
                for state in self._states.values()
                if state.source_type == STATIC_VIDEO
            ]
        for state in states:
            try:
                ok_position, position = state.pipeline.query_position(
                    Gst.Format.TIME
                )
                ok_duration, duration = state.pipeline.query_duration(
                    Gst.Format.TIME
                )
                if ok_position:
                    state.position_seconds = float(position) / float(Gst.SECOND)
                if ok_duration:
                    state.duration_seconds = float(duration) / float(Gst.SECOND)
            except Exception:
                continue

    def _selection_snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        registered: list[SourceRecord] = []
        eligible: list[SourceRecord] = []
        demanded: list[SourceRecord] = []
        skip_reasons: dict[str, str] = {}
        seen: set[str] = set()
        if self.registry is None:
            return {
                "registered": [],
                "eligible": [],
                "demanded": [],
                "selected": [],
                "skip_reasons": {},
                "active_limit": min(self.max_sources, self.max_active_sources),
            }
        aggregate_video_demand = (
            self.demand_controller.video_required()
            if self.demand_controller is not None
            else self.frontend_frame_worker is not None
        )
        aggregate_demand = aggregate_video_demand or self._ai_required()
        for record in self.registry.list():
            if canonical_source_type(
                record.source_uri, record.source_type
            ) != self.source_type_filter:
                continue
            registered.append(record)
            source_id = record.source_uri
            display_id = VideoFileIngestor.redact_uri(source_id)
            if source_id in seen:
                skip_reasons[display_id] = "duplicate_source"
                continue
            seen.add(source_id)
            if not record.enabled:
                skip_reasons[display_id] = "disabled"
                continue
            with self._lock:
                known_skip = self._source_skip_codes.get(source_id)
            if known_skip is not None:
                skip_reasons[display_id] = known_skip
                continue
            if not self.is_supported_source(record):
                skip_reasons[display_id] = "unsupported_codec_or_container"
                continue
            if (
                not self.rtsp_enabled
                and VideoFileIngestor.is_rtsp_uri(source_id or "")
            ):
                skip_reasons[display_id] = "unsupported_codec_or_container"
                continue
            if self.source_allowlist and (
                source_id not in self.source_allowlist
                and display_id not in self.source_allowlist
            ):
                skip_reasons[display_id] = "not_in_source_allowlist"
                continue
            eligible.append(record)
            source_video_demand = (
                self.demand_controller.video_required(source_id)
                if self.demand_controller is not None
                else self.frontend_frame_worker is not None
            )
            if not aggregate_demand or not (
                self._ai_required() or source_video_demand
            ):
                skip_reasons[display_id] = "no_demand"
                continue
            demanded.append(record)

        active_limit = min(self.max_sources, self.max_active_sources)
        selected = demanded[:active_limit]
        for record in demanded[active_limit:]:
            skip_reasons[
                VideoFileIngestor.redact_uri(record.source_uri)
            ] = "max_sources_reached"
        with self._lock:
            states = set(self._states)
            retry_after = dict(self._retry_after)
            open_errors = dict(self._source_open_errors)
            skip_codes = dict(self._source_skip_codes)
        for record in selected:
            source_id = record.source_uri
            display_id = VideoFileIngestor.redact_uri(source_id)
            if source_id in states:
                continue
            retry_at = retry_after.get(source_id, 0.0)
            if retry_at > now:
                skip_reasons[display_id] = (
                    skip_codes[source_id]
                    if source_id in skip_codes
                    else "pipeline_open_failed"
                    if source_id in open_errors
                    else "quarantined_or_retry_wait"
                )
        return {
            "registered": registered,
            "eligible": eligible,
            "demanded": demanded,
            "selected": selected,
            "skip_reasons": skip_reasons,
            "active_limit": active_limit,
        }

    def _active_records(self) -> list[SourceRecord]:
        # Keep source pipelines fully closed when neither frontend nor AI has
        # a consumer. This is stronger than dropping at the valves: NVDEC,
        # mapping and conversion also consume no resources while idle.
        snapshot = self._selection_snapshot()
        records = snapshot["selected"]
        if len(snapshot["demanded"]) > snapshot["active_limit"]:
            LOGGER.warning(
                "source_type=%s sources=%d exceeds max_sources=%d; capping",
                self.source_type_filter,
                len(snapshot["demanded"]),
                snapshot["active_limit"],
            )
        reasons = snapshot["skip_reasons"]
        if reasons != self._last_selection_skip_reasons:
            for source_id, reason in reasons.items():
                LOGGER.info(
                    "DeepStream source skipped source=%s source_type=%s reason=%s",
                    source_id,
                    self.source_type_filter,
                    reason,
                )
            self._last_selection_skip_reasons = dict(reasons)
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
            with self._lock:
                existing = self._states.get(source_id)
            if (
                existing is not None
                and existing.source_type == STATIC_VIDEO
                and self.registry.get(source_id) is not None
            ):
                continue
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

        opens_remaining = self.max_source_opens_per_sync
        for record in records:
            with self._lock:
                state = self._states.get(record.source_uri)
                retry_after = self._retry_after.get(record.source_uri, 0.0)
                closing = record.source_uri in self._closing_sources
                if state is not None:
                    if state.delivery_target_fps != record.fps:
                        state.next_frame_due_monotonic = 0.0
                        state.frontend_next_frame_due_monotonic = 0.0
                    state.delivery_target_fps = record.fps
            if state is not None and state.source_uri != record.source_uri:
                self._close_source(record.source_uri)
                state = None
                closing = True
            if state is not None or closing or retry_after > time.monotonic():
                continue
            now = time.monotonic()
            if (
                opens_remaining <= 0
                or now < self._next_source_open_monotonic
            ):
                continue
            opens_remaining -= 1
            try:
                self._open_source(record)
                with self._lock:
                    self._source_open_errors.pop(record.source_uri, None)
                self._next_source_open_monotonic = (
                    time.monotonic() + self.source_open_stagger_seconds
                )
            except Exception as exc:
                self._open_failures += 1
                safe_uri = VideoFileIngestor.redact_uri(record.source_uri or "")
                safe_error = str(exc).replace(record.source_uri, safe_uri)
                self._last_error = (
                    f"Could not open {safe_uri}: "
                    f"{type(exc).__name__}: {safe_error}"
                )
                self._source_open_errors[record.source_uri] = self._last_error
                self._retry_after[record.source_uri] = (
                    time.monotonic() + self.rtsp_reconnect_seconds
                )
                LOGGER.exception("%s", self._last_error)

    def _submit_latest_round(self) -> None:
        if not self._ai_required():
            return
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

    def _poll_bus_messages(self) -> None:
        """Poll every source bus on the owner thread; no GLib signal callbacks."""
        with self._lock:
            states = list(self._states.values())
        for state in states:
            while True:
                message = state.bus.pop()
                if message is None:
                    break
                self._on_bus_message(
                    state.bus,
                    message,
                    state.source_id,
                    state.generation,
                )

    def _run(self) -> None:
        self._owner_thread_id = threading.get_ident()
        self._started.set()
        submit_interval = 1.0 / self.SCHEDULER_MAX_FPS
        next_sync = 0.0
        next_submit = 0.0
        registry_revision = -1
        while not self._stop.is_set():
            now = time.monotonic()
            try:
                self._poll_bus_messages()
                self._process_commands()
                current_revision = self.registry.revision
                if current_revision != registry_revision or now >= next_sync:
                    self._sync_sources()
                    self._apply_demand_state()
                    self._update_static_positions()
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
            self._close_source_owner(source_id)
        self._owner_thread_id = None

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
            if self._thread.is_alive():
                LOGGER.warning(
                    "DeepStream owner thread did not stop within timeout "
                    "source_type=%s",
                    self.source_type_filter,
                )
        if self._demand_unsubscribe is not None:
            self._demand_unsubscribe()
            self._demand_unsubscribe = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            selection = self._selection_snapshot()
            skip_sources = selection["skip_reasons"]
            skip_counts: dict[str, int] = {}
            for reason in skip_sources.values():
                skip_counts[reason] = skip_counts.get(reason, 0) + 1
            selected_ids = {
                record.source_uri for record in selection["selected"]
            }
            active_pipeline_count = sum(
                1
                for source_id, state in self._states.items()
                if source_id in selected_ids
                and state.pipeline_state.lower() == "playing"
            )
            paused_pipeline_count = sum(
                1
                for state in self._states.values()
                if state.pipeline_state.lower() == "paused"
                or state.paused_for_demand
            )
            return {
                "enabled": True,
                "backend": "deepstream",
                "source_type_filter": self.source_type_filter,
                "max_sources": self.max_sources,
                "configured_static_video_source_count": (
                    self.max_sources
                    if self.source_type_filter == STATIC_VIDEO
                    else None
                ),
                "max_active_sources": self.max_active_sources,
                "effective_source_limit": selection["active_limit"],
                "registered_source_count": len(selection["registered"]),
                "eligible_source_count": len(selection["eligible"]),
                "demanded_source_count": len(selection["demanded"]),
                "selected_source_count": len(selection["selected"]),
                "max_source_opens_per_sync": self.max_source_opens_per_sync,
                "source_open_stagger_seconds": self.source_open_stagger_seconds,
                "source_allowlist_count": len(self.source_allowlist),
                "active_source_count": len(self._states),
                "active_pipeline_count": active_pipeline_count,
                "paused_pipeline_count": paused_pipeline_count,
                "skip_reasons": {
                    "counts": skip_counts,
                    "sources": skip_sources,
                },
                "pending_source_opens": max(
                    0,
                    len(
                        selected_ids.difference(self._states)
                    ),
                ),
                "running": self._thread is not None and self._thread.is_alive(),
                "fps_control": "sources.fps",
                "gpu_resize_enabled": self.gpu_resize_enabled,
                "gpu_resize_active": self._gpu_resize_available,
                "loop": self.loop,
                "rtsp_enabled": self.rtsp_enabled,
                "video_only_mode": self.video_only_mode,
                "video_required": self._video_required(),
                "ai_required": self._ai_required(),
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
                    state.display_uri: {
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
                        "raw_encoded_packets": state.raw_encoded_packets,
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
                        "frontend_rate_limited_frames": (
                            state.frontend_rate_limited_frames
                        ),
                        "frontend_frame_index": state.frontend_frame_index,
                        "received_frames": state.received_frames,
                        "pre_submit_replacements": state.pre_submit_replacements,
                        "submitted_frames": state.submitted_frames,
                        "frame_index": state.frame_index,
                        "last_frame_age_seconds": (
                            round(now - state.last_frame_monotonic, 3)
                            if state.last_frame_monotonic > 0
                            else None
                        ),
                        "frontend_samples": state.raw_samples,
                        "frontend_frame_age_seconds": (
                            round(now - state.last_raw_packet_monotonic, 3)
                            if state.last_raw_packet_monotonic > 0
                            else None
                        ),
                        "pipeline_state": state.pipeline_state,
                        "pending_state": state.pending_state,
                        "source_generation": state.generation,
                        "last_bus_message": state.last_bus_message,
                        "last_bus_element": state.last_bus_element,
                        "last_bus_message_age_seconds": (
                            round(now - state.last_bus_message_monotonic, 3)
                            if state.last_bus_message_monotonic > 0
                            else None
                        ),
                        "last_pad_caps": state.last_pad_caps,
                        "position_seconds": state.position_seconds,
                        "duration_seconds": state.duration_seconds,
                        "loop_count": self._loop_counts.get(source_id, 0),
                        "seek_failures": state.seek_failures,
                        "paused_for_demand": state.paused_for_demand,
                        "recent_bus_messages": list(
                            self._bus_history.get(source_id, ())
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
