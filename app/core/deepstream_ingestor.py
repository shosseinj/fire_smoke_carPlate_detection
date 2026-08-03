from __future__ import annotations

import logging
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlsplit

import cv2
import numpy as np

from app.core.router import TaskRouter
from app.core.source_registry import (
    RTSP,
    SOURCE_TYPES,
    SourceRecord,
    SourceRegistry,
    canonical_source_type,
)
from app.core.video_ingestor import VideoFileIngestor
from app.core.live_branch import GpuLiveBranchManager, _is_nvmm_caps

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
    sink: Any
    bus: Any
    bus_handler_id: int
    pipeline_handler_id: int
    source_pad_handler_id: int
    sink_handler_id: int
    frame_width: int
    frame_height: int
delivery_target_fps: float | None
    tee: Any = None
    native_caps: Any = None
    next_frame_due_monotonic: float = 0.0
    latest_frame: np.ndarray | None = None
    latest_version: int = 0
    submitted_version: int = 0
    frame_index: int = -1
    source_time_seconds: float | None = None
    decoded_samples: int = 0
    rate_limited_frames: int = 0
    received_frames: int = 0
    pre_submit_replacements: int = 0
    submitted_frames: int = 0
    last_frame_monotonic: float = 0.0
    last_error: str | None = None
    warnings: int = 0
    loop_count: int = 0
    source_frame_width: int = 0
    source_frame_height: int = 0
    live_registration_retry_id: int | None = None
    live_registration_attempts: int = 0


class DeepStreamIngestor:
    """Uses DeepStream/NVDEC for file and RTSP decoding, then feeds the router."""

    SCHEDULER_MAX_FPS = 240.0
    LIVE_REGISTRATION_RETRY_INTERVAL_MS = 50
    LIVE_REGISTRATION_MAX_RETRIES = 20

    REQUIRED_ELEMENTS = (
        "nvurisrcbin",
        "nvvideoconvert",
        "queue",
        "identity",
        "capsfilter",
        "appsink",
        "tee",
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
        max_sources: int = 256,
        rtsp_enabled: bool = True,
        rtsp_transport: str = "tcp",
        rtsp_latency_ms: int = 500,
        rtsp_reconnect_seconds: float = 3.0,
        rtsp_stall_timeout_seconds: int = 30,
        skip_taskless_sources: bool = True,
        gst_loader: Callable[[], tuple[Any, Any]] = _load_gstreamer,
        on_source_started: Callable[[str], None] | None = None,
        on_source_eos: Callable[[str, bool], None] | None = None,
        on_source_failed: Callable[[str, str], None] | None = None,
        live_branch_manager: GpuLiveBranchManager | None = None,
    ) -> None:
        if source_type_filter not in SOURCE_TYPES:
            raise ValueError(f"source_type_filter must be one of {sorted(SOURCE_TYPES)}")
        self.source_type_filter = source_type_filter
        self.max_sources = max(1, int(max_sources))
        self.registry = registry
        self.router = router
        self.project_root = project_root
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
        self.on_source_started = on_source_started
        self.on_source_eos = on_source_eos
        self.on_source_failed = on_source_failed
        self.live_branch_manager = live_branch_manager
        self._started_sources: set[str] = set()

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
        parsed = urlsplit(source_uri)
        path = Path(unquote(parsed.path) if parsed.scheme.lower() == "file" else source_uri).expanduser()
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

    def _cancel_live_registration_retry(self, state: DeepStreamSourceState) -> None:
        retry_id = state.live_registration_retry_id
        state.live_registration_retry_id = None
        if retry_id is None or self._glib is None:
            return
        try:
            self._glib.source_remove(retry_id)
        except Exception:
            LOGGER.debug("Could not cancel live registration retry for %s", state.source_id, exc_info=True)

    def _try_register_live_source(self, source_id: str, pad: Any, tee: Any) -> bool:
        manager = self.live_branch_manager
        if manager is None or not manager.enabled:
            return False
        with self._lock:
            state = self._states.get(source_id)
            if state is None or state.tee is not tee or source_id in self._closing_sources:
                return False
            candidates: list[Any] = []
            for candidate_pad in (pad, tee.get_static_pad("sink")):
                if candidate_pad is None:
                    continue
                try:
                    current = candidate_pad.get_current_caps()
                except Exception:
                    current = None
                if current is not None:
                    candidates.append(current)
                try:
                    queried = candidate_pad.query_caps(None)
                except Exception:
                    queried = None
                if queried is not None:
                    candidates.append(queried)
            caps = next(
                (
                    candidate
                    for candidate in candidates
                    if _is_nvmm_caps(candidate)
                    and re.search(r"width=(?:\(int\))?\d+", candidate.to_string())
                    and re.search(r"height=(?:\(int\))?\d+", candidate.to_string())
                ),
                None,
            )
            if caps is None:
                LOGGER.debug(
                    "Live NVMM registration pending for %s; caps=%s",
                    source_id,
                    [candidate.to_string() for candidate in candidates],
                )
                return False
            self._cancel_live_registration_retry(state)
            state.live_registration_attempts = 0
            try:
                manager.set_runtime(self._gst, self._glib)
                attached = manager.attach_source(
                    source_id, tee, state.pipeline, confirmed_caps=caps
                )
            except Exception:
                LOGGER.exception(
                    "GPU live branch registration failed without stopping AI: %s",
                    source_id,
                )
                return False
            if not attached:
                return False
            LOGGER.info("LIVE_NVMM_CAPS source=%s caps=%s", source_id, caps.to_string())
            return True

    def _schedule_live_registration_retry(self, source_id: str, pad: Any, tee: Any) -> None:
        manager = self.live_branch_manager
        if manager is None or not manager.enabled or self._glib is None:
            return
        with self._lock:
            state = self._states.get(source_id)
            if state is None or state.tee is not tee or state.live_registration_retry_id is not None:
                return
            state.live_registration_attempts = 0

        def retry() -> bool:
            with self._lock:
                state = self._states.get(source_id)
                if state is None or state.tee is not tee or source_id in self._closing_sources:
                    return False
                state.live_registration_attempts += 1
                attempts = state.live_registration_attempts
            if self._try_register_live_source(source_id, pad, tee):
                return False
            if attempts >= self.LIVE_REGISTRATION_MAX_RETRIES:
                with self._lock:
                    current = self._states.get(source_id)
                    if current is not None:
                        current.live_registration_retry_id = None
                LOGGER.warning(
                    "Decoder output never negotiated fixed NVMM caps; live branch disabled for source: %s",
                    source_id,
                )
                return False
            return True

        try:
            retry_id = self._glib.timeout_add(self.LIVE_REGISTRATION_RETRY_INTERVAL_MS, retry)
        except Exception:
            LOGGER.exception("Could not schedule deferred live branch registration: %s", source_id)
            return
        with self._lock:
            state = self._states.get(source_id)
            if state is None or state.tee is not tee or source_id in self._closing_sources:
                try:
                    self._glib.source_remove(retry_id)
                except Exception:
                    pass
                return
            state.live_registration_retry_id = retry_id

def _on_decoded_pad_added(
        self, _: Any, pad: Any, tee: Any, native_caps: Any, ai_pacer: Any, source_id: str
    ) -> None:
        Gst, _ = self._require_runtime()
        caps = pad.get_current_caps() or pad.query_caps(None)
        caps_text = caps.to_string() if caps is not None else ""
        if not caps_text or not caps_text.startswith("video/"):
            return
        sink_pad = native_caps.get_static_pad("sink")
        if sink_pad is None or sink_pad.is_linked():
            return
        result = pad.link(sink_pad)
        if result != Gst.PadLinkReturn.OK:
            LOGGER.error("DeepStream NVMM source pad could not be linked to native caps: %s", result)
            return
        tee_sink = tee.get_static_pad("sink")
        if tee_sink is None or tee_sink.is_linked():
            return
        if not native_caps.get_static_pad("src").link(tee_sink) == Gst.PadLinkReturn.OK:
            LOGGER.error("Could not link native caps to tee: %s", source_id)
            return
        ai_pad = tee.get_request_pad("src_%u")
        ai_sink = ai_pacer.get_static_pad("sink")
        if ai_pad is None or ai_sink is None or ai_pad.link(ai_sink) != Gst.PadLinkReturn.OK:
            if ai_pad is not None:
                tee.release_request_pad(ai_pad)
            LOGGER.error("DeepStream AI tee branch could not be linked: %s", source_id)
            return
        if self.live_branch_manager is not None and self.live_branch_manager.enabled:
            try:
                state = self._states.get(source_id)
                if state is not None:
                    if not self._try_register_live_source(source_id, pad, tee):
                        self._schedule_live_registration_retry(source_id, pad, tee)
            except Exception:
                LOGGER.exception("GPU live branch attachment failed without stopping AI: %s", source_id)

    @staticmethod
    def _on_autoplug_continue(_: Any, __: Any, caps: Any) -> bool:
        """Keep decodebin from searching for decoders for unused audio tracks."""
        return caps is None or not caps.to_string().startswith("audio/")

    def _on_deep_element_added(self, _: Any, __: Any, element: Any) -> None:
        factory = element.get_factory()
        if factory is not None and factory.get_name() == "uridecodebin":
            element.connect("autoplug-continue", self._on_autoplug_continue)

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
            if self.on_source_failed is not None:
                self.on_source_failed(source_id, self._redact_error(source_id, detail))
            self._mark_failed(source_id, detail)
        elif message.type == Gst.MessageType.EOS:
            will_loop = self._should_loop_source(source_id)
            if self.on_source_eos is not None:
                self.on_source_eos(source_id, will_loop)
            if will_loop:
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
            if state is None or state.sink is not sink:
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
            raise ValueError(f"فرمت نمونه DeepStream پشتیبانی نشده: {pixel_format}")
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
        is_rtsp = VideoFileIngestor.is_rtsp_uri(record.source_uri)
        if not is_rtsp:
            parsed = urlsplit(record.source_uri)
            source_path = Path(
                unquote(parsed.path) if parsed.scheme.lower() == "file" else record.source_uri
            ).expanduser()
            if not source_path.is_absolute():
                source_path = self.project_root / source_path
            if not source_path.is_file():
                raise FileNotFoundError(f"Video file was not found: {record.source_uri}")
        gst_uri = self._resolve_uri(record.source_uri)
        display_uri = VideoFileIngestor.redact_uri(record.source_uri)
        safe_id = self._safe_element_name(record.source_uri)
        pipeline = Gst.Pipeline.new(f"pipeline_{safe_id}")
        if pipeline is None:
            raise RuntimeError("Could not create a GStreamer pipeline")

try:
            source = self._make("nvurisrcbin", f"source_{safe_id}")
            queue = self._make("queue", f"queue_{safe_id}")
            pacer = self._make("identity", f"pacer_{safe_id}")
            tee = self._make("tee", f"decode_tee_{safe_id}")
            native_caps = self._make("capsfilter", f"native_caps_{safe_id}")
            gpu_convert = self._make("nvvideoconvert", f"gpu_convert_{safe_id}")
            bgrx_caps = self._make("capsfilter", f"bgrx_caps_{safe_id}")
            sink = self._make("appsink", f"appsink_{safe_id}")

            # DeepStream 7.1 still lets its internal uridecodebin autoplug AAC
            # even with disable-audio=true. Hook it before the source bin is
            # added so encoded audio pads are ignored without requiring an AAC
            # decoder or spending CPU on an unused audio stream.
            pipeline_handler_id = pipeline.connect(
                "deep-element-added", self._on_deep_element_added
            )

            source.set_property("uri", gst_uri)
            self._set_if_supported(
                source, "source-id", self._gst_source_id(record.source_uri)
            )
            self._set_if_supported(source, "disable-audio", True)
            if is_rtsp:
                self._set_if_supported(source, "latency", self.rtsp_latency_ms)
                self._set_if_supported(source, "drop-on-latency", True)
                self._set_if_supported(
                    source,
                    "rtsp-reconnect-interval",
                    self.rtsp_stall_timeout_seconds,
                )
                self._set_if_supported(source, "rtsp-reconnect-attempts", -1)
                self._set_if_supported(
                    source,
                    "select-rtp-protocol",
                    4 if self.rtsp_transport == "tcp" else 0,
                )
            else:
                # nvurisrcbin's internal file-loop can emit buffers before a new
                # segment event on some DeepStream 7.1 builds, stalling NVDEC.
                # Leave it disabled and restart the pipeline cleanly on EOS.
                self._set_if_supported(source, "file-loop", False)

            queue.set_property("leaky", 2)
            queue.set_property("max-size-buffers", 1)
            queue.set_property("max-size-bytes", 0)
            queue.set_property("max-size-time", 0)
            pacer.set_property("sync", not is_rtsp)
            # Always resize on GPU via nvvideoconvert capsfilter to
            # frame_width x frame_height (default 640x640).  This avoids
            # CPU-side resize in _submit_latest_round and eliminates
            # per-frame GPU↔CPU copy for resize.
            bgrx_caps_value = (
                f"video/x-raw,format=BGRx,"
                f"width={record.frame_width},height={record.frame_height}"
            )
            bgrx_caps.set_property(
                "caps",
                Gst.Caps.from_string(bgrx_caps_value),
            )
            sink.set_property("emit-signals", True)
            sink.set_property("sync", False)
            sink.set_property("max-buffers", 1)
            sink.set_property("drop", True)
            self._set_if_supported(sink, "enable-last-sample", False)

            # Allow decoder to output native NVMM resolution by placing a
            # permissive capsfilter before the tee. The AI branch will scale
            # to 640x640 via its own nvvideoconvert + capsfilter, while the
            # live branch receives native resolution.
            native_caps.set_property(
                "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM)")
            )

for element in (
                source,
                tee,
                native_caps,
                queue,
                pacer,
                gpu_convert,
                bgrx_caps,
                sink,
            ):
                pipeline.add(element)
            if not pacer.link(queue):
                raise RuntimeError("Could not link source pacer to DeepStream queue")
            if not queue.link(gpu_convert):
                raise RuntimeError("Could not link DeepStream queue to nvvideoconvert")
            if not gpu_convert.link(bgrx_caps):
                raise RuntimeError("Could not link nvvideoconvert to BGRx caps")
            if not bgrx_caps.link(sink):
                raise RuntimeError("Could not link BGRx caps to appsink")
source_pad_handler_id = source.connect(
                "pad-added", self._on_decoded_pad_added, tee, native_caps, pacer, record.source_uri
            )
            sink_handler_id = sink.connect(
                "new-sample", self._on_new_sample, record.source_uri
            )

            bus = pipeline.get_bus()
            bus.add_signal_watch()
            bus_handler_id = bus.connect("message", self._on_bus_message, record.source_uri)
state = DeepStreamSourceState(
                source_id=record.source_uri,
                source_uri=record.source_uri,
                display_uri=display_uri,
                source_type="rtsp" if is_rtsp else "video_file",
                pipeline=pipeline,
                source=source,
                tee=tee,
                native_caps=native_caps,
                sink=sink,
                bus=bus,
                bus_handler_id=bus_handler_id,
                pipeline_handler_id=pipeline_handler_id,
                source_pad_handler_id=source_pad_handler_id,
                sink_handler_id=sink_handler_id,
                frame_width=record.frame_width,
                frame_height=record.frame_height,
                delivery_target_fps=record.fps,
                loop_count=self._loop_counts.get(record.source_uri, 0),
            )
            with self._lock:
                self._states[record.source_uri] = state
            result = pipeline.set_state(Gst.State.PLAYING)
            if result == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("GStreamer pipeline refused the PLAYING state")
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

    def _dispose_state(self, state: DeepStreamSourceState) -> None:
        Gst, _ = self._require_runtime()
        with self._lock:
            self._cancel_live_registration_retry(state)
        for element, handler_id in (
            (state.sink, state.sink_handler_id),
            (state.source, state.source_pad_handler_id),
            (state.pipeline, state.pipeline_handler_id),
            (state.bus, state.bus_handler_id),
        ):
            try:
                element.disconnect(handler_id)
            except Exception:
                pass
        try:
            state.bus.remove_signal_watch()
        except Exception:
            pass
        if self.live_branch_manager is not None:
            self.live_branch_manager.detach_source(state.source_id)
        state.pipeline.set_state(Gst.State.NULL)
        try:
            # Wait until NVDEC and nvurisrcbin have actually released their
            # resources before another pipeline for this camera is constructed.
            state.pipeline.get_state(5 * Gst.SECOND)
        except Exception:
            LOGGER.debug("GStreamer NULL-state wait failed for %s", state.source_id)

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
            if state is not None:
                self._cancel_live_registration_retry(state)
        if state is None:
            return
        self._schedule_state_disposal(state)

    def release_source(self, source_id: str) -> None:
        self._close_source(source_id)
        with self._lock:
            self._retry_after.pop(source_id, None)
            self._failed_sources.discard(source_id)
            self._frame_sequences.pop(source_id, None)
            self._loop_counts.pop(source_id, None)
            self._gst_source_ids.pop(source_id, None)
            self._started_sources.discard(source_id)

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
            if state is not None and (
                state.source_uri != record.source_uri
                or state.frame_width != record.frame_width
                or state.frame_height != record.frame_height
            ):
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
                if self.on_source_failed is not None:
                    self.on_source_failed(record.source_uri, self._last_error)

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
            source_frames = [state.latest_frame for state in selected]
            frames = source_frames
            source_ids = [state.source_id for state in selected]
            frame_indexes = [state.frame_index for state in selected]
            source_times = [state.source_time_seconds for state in selected]
            metadata = [
                {
                    "source_uri": state.display_uri,
                    "source_type": state.source_type,
                    "frame_width": state.frame_width,
                    "frame_height": state.frame_height,
                    "source_frame_width": state.source_frame_width,
                    "source_frame_height": state.source_frame_height,
                    "source_frame": source_frame,
                    "ingest_backend": "deepstream",
                }
                for state, source_frame in zip(selected, source_frames)
            ]
            versions = {state.source_id: state.latest_version for state in selected}
        if not frames:
            return

        if self.on_source_started is not None:
            for source_id in source_ids:
                if source_id not in self._started_sources:
                    self._started_sources.add(source_id)
                    self.on_source_started(source_id)
        self._round_sequence += 1
        try:
            self.router.submit_round(
                frames=frames,  # type: ignore[arg-type]
                source_ids=source_ids,
                round_sequence=self._round_sequence,
                frame_indexes=frame_indexes,
                source_times_seconds=source_times,
                metadata=metadata,
            )
        except Exception as exc:
            if self.on_source_failed is not None:
                for source_id in source_ids:
                    self.on_source_failed(source_id, str(exc))
            raise
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
                        "rate_limited_frames": state.rate_limited_frames,
                        "received_frames": state.received_frames,
                        "pre_submit_replacements": state.pre_submit_replacements,
                        "submitted_frames": state.submitted_frames,
                        "frame_index": state.frame_index,
                        "loop_count": self._loop_counts.get(
                            source_id, state.loop_count
                        ),
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
