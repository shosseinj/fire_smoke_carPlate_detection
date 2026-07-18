from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.core.router import TaskRouter
from app.core.source_registry import SourceRecord, SourceRegistry
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
    sink: Any
    bus: Any
    bus_handler_id: int
    pipeline_handler_id: int
    source_pad_handler_id: int
    sink_handler_id: int
    frame_width: int
    frame_height: int
    latest_frame: np.ndarray | None = None
    latest_version: int = 0
    submitted_version: int = 0
    frame_index: int = -1
    source_time_seconds: float | None = None
    decoded_samples: int = 0
    received_frames: int = 0
    submitted_frames: int = 0
    last_frame_monotonic: float = 0.0
    last_error: str | None = None
    warnings: int = 0
    loop_count: int = 0


class DeepStreamIngestor:
    """Uses DeepStream/NVDEC for file and RTSP decoding, then feeds the router."""

    REQUIRED_ELEMENTS = (
        "nvurisrcbin",
        "nvvideoconvert",
        "queue",
        "identity",
        "videoconvert",
        "capsfilter",
        "appsink",
    )

    def __init__(
        self,
        *,
        registry: SourceRegistry,
        router: TaskRouter,
        project_root: Path,
        target_fps: float = 5.0,
        loop: bool = True,
        rtsp_enabled: bool = True,
        rtsp_transport: str = "tcp",
        rtsp_latency_ms: int = 500,
        rtsp_reconnect_seconds: float = 3.0,
        rtsp_stall_timeout_seconds: int = 30,
        gst_loader: Callable[[], tuple[Any, Any]] = _load_gstreamer,
    ) -> None:
        self.registry = registry
        self.router = router
        self.project_root = project_root
        self.target_fps = max(0.1, float(target_fps))
        self.loop = bool(loop)
        self.rtsp_enabled = bool(rtsp_enabled)
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
        return "".join(character if character.isalnum() else "_" for character in source_id)

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

    def _on_pad_added(self, _: Any, pad: Any, queue: Any) -> None:
        Gst, _ = self._require_runtime()
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None or not caps.to_string().startswith("video/"):
            return
        sink_pad = queue.get_static_pad("sink")
        if sink_pad is None or sink_pad.is_linked():
            return
        result = pad.link(sink_pad)
        if result != Gst.PadLinkReturn.OK:
            LOGGER.error("DeepStream source pad could not be linked: %s", result)

    @staticmethod
    def _on_autoplug_continue(_: Any, __: Any, caps: Any) -> bool:
        """Keep decodebin from searching for decoders for unused audio tracks."""
        return caps is None or not caps.to_string().startswith("audio/")

    def _on_deep_element_added(self, _: Any, __: Any, element: Any) -> None:
        factory = element.get_factory()
        if factory is not None and factory.get_name() == "uridecodebin":
            element.connect("autoplug-continue", self._on_autoplug_continue)

    def _redact_error(self, source_id: str, message: str) -> str:
        with self._lock:
            state = self._states.get(source_id)
            if state is None:
                return message
            return message.replace(state.source_uri, state.display_uri)

    def _mark_failed(
        self,
        source_id: str,
        message: str,
        *,
        expected: bool = False,
    ) -> None:
        safe_message = self._redact_error(source_id, message)
        with self._lock:
            state = self._states.get(source_id)
            if state is not None:
                state.last_error = safe_message
            self._last_error = f"{source_id}: {safe_message}"
            self._failed_sources.add(source_id)
            self._retry_after[source_id] = (
                time.monotonic()
                if expected
                else time.monotonic() + self.rtsp_reconnect_seconds
            )
        log = LOGGER.info if expected else LOGGER.error
        log("DeepStream source %s: %s", source_id, safe_message)

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
            if self.loop:
                with self._lock:
                    if source_id not in self._failed_sources:
                        self._loop_counts[source_id] = (
                            self._loop_counts.get(source_id, 0) + 1
                        )
                LOGGER.info("DeepStream file reached EOS; restarting cleanly: %s", source_id)
                self._mark_failed(
                    source_id,
                    "End of stream; source restart scheduled",
                    expected=True,
                )
            else:
                LOGGER.info("DeepStream file reached EOS: %s", source_id)
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
            LOGGER.warning("DeepStream source warning: %s: %s", source_id, safe_warning)

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
            minimum_interval = 1.0 / self.target_fps
            if now - state.last_frame_monotonic < minimum_interval:
                return Gst.FlowReturn.OK

        try:
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            width = int(structure.get_value("width"))
            height = int(structure.get_value("height"))
            buffer = sample.get_buffer()
            payload = buffer.extract_dup(0, buffer.get_size())
            row_stride = len(payload) // max(height, 1)
            packed_width = width * 3
            if width <= 0 or height <= 0 or row_stride < packed_width:
                raise ValueError(
                    f"Invalid BGR sample layout: {width}x{height}, stride={row_stride}"
                )
            flat = np.frombuffer(payload, dtype=np.uint8)
            frame = (
                flat[: row_stride * height]
                .reshape(height, row_stride)[:, :packed_width]
                .reshape(height, width, 3)
                .copy()
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
            state.latest_frame = frame
            state.latest_version += 1
            state.frame_index = self._next_frame_index_locked(source_id)
            state.source_time_seconds = source_time
            state.received_frames += 1
            state.last_frame_monotonic = now
            state.last_error = None
        return Gst.FlowReturn.OK

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
            source_path = Path(record.source_uri).expanduser()
            if not source_path.is_absolute():
                source_path = self.project_root / source_path
            if not source_path.is_file():
                raise FileNotFoundError(f"Video file was not found: {record.source_uri}")
        gst_uri = self._resolve_uri(record.source_uri)
        display_uri = VideoFileIngestor.redact_uri(record.source_uri)
        safe_id = self._safe_element_name(record.source_id)
        pipeline = Gst.Pipeline.new(f"pipeline_{safe_id}")
        if pipeline is None:
            raise RuntimeError("Could not create a GStreamer pipeline")

        try:
            source = self._make("nvurisrcbin", f"source_{safe_id}")
            queue = self._make("queue", f"queue_{safe_id}")
            pacer = self._make("identity", f"pacer_{safe_id}")
            gpu_convert = self._make("nvvideoconvert", f"gpu_convert_{safe_id}")
            bgrx_caps = self._make("capsfilter", f"bgrx_caps_{safe_id}")
            cpu_convert = self._make("videoconvert", f"cpu_convert_{safe_id}")
            bgr_caps = self._make("capsfilter", f"bgr_caps_{safe_id}")
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
                source, "source-id", self._gst_source_id(record.source_id)
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
            bgrx_caps.set_property(
                "caps",
                Gst.Caps.from_string(
                    "video/x-raw,format=BGRx,"
                    f"width={record.frame_width},height={record.frame_height}"
                ),
            )
            bgr_caps.set_property("caps", Gst.Caps.from_string("video/x-raw,format=BGR"))
            sink.set_property("emit-signals", True)
            sink.set_property("sync", False)
            sink.set_property("max-buffers", 1)
            sink.set_property("drop", True)
            self._set_if_supported(sink, "enable-last-sample", False)

            for element in (
                source,
                queue,
                pacer,
                gpu_convert,
                bgrx_caps,
                cpu_convert,
                bgr_caps,
                sink,
            ):
                pipeline.add(element)
            if not pacer.link(queue):
                raise RuntimeError("Could not link source pacer to DeepStream queue")
            if not queue.link(gpu_convert):
                raise RuntimeError("Could not link DeepStream queue to nvvideoconvert")
            if not gpu_convert.link(bgrx_caps):
                raise RuntimeError("Could not link nvvideoconvert to BGRx caps")
            if not bgrx_caps.link(cpu_convert):
                raise RuntimeError("Could not link BGRx caps to videoconvert")
            if not cpu_convert.link(bgr_caps):
                raise RuntimeError("Could not link videoconvert to BGR caps")
            if not bgr_caps.link(sink):
                raise RuntimeError("Could not link BGR caps to appsink")
            source_pad_handler_id = source.connect(
                "pad-added", self._on_pad_added, pacer
            )
            sink_handler_id = sink.connect(
                "new-sample", self._on_new_sample, record.source_id
            )

            bus = pipeline.get_bus()
            bus.add_signal_watch()
            bus_handler_id = bus.connect("message", self._on_bus_message, record.source_id)
            state = DeepStreamSourceState(
                source_id=record.source_id,
                source_uri=record.source_uri,
                display_uri=display_uri,
                source_type="rtsp" if is_rtsp else "video_file",
                pipeline=pipeline,
                source=source,
                sink=sink,
                bus=bus,
                bus_handler_id=bus_handler_id,
                pipeline_handler_id=pipeline_handler_id,
                source_pad_handler_id=source_pad_handler_id,
                sink_handler_id=sink_handler_id,
                frame_width=record.frame_width,
                frame_height=record.frame_height,
                loop_count=self._loop_counts.get(record.source_id, 0),
            )
            with self._lock:
                self._states[record.source_id] = state
            result = pipeline.set_state(Gst.State.PLAYING)
            if result == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("GStreamer pipeline refused the PLAYING state")
            with self._lock:
                self._retry_after.pop(record.source_id, None)
        except Exception:
            with self._lock:
                state = self._states.pop(record.source_id, None)
            if state is not None:
                self._dispose_state(state)
            else:
                pipeline.set_state(Gst.State.NULL)
                try:
                    pipeline.get_state(5 * Gst.SECOND)
                except Exception:
                    pass
            raise

    def _dispose_state(self, state: DeepStreamSourceState) -> None:
        Gst, _ = self._require_runtime()
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
        state.pipeline.set_state(Gst.State.NULL)
        try:
            # Wait until NVDEC and nvurisrcbin have actually released their
            # resources before another pipeline for this camera is constructed.
            state.pipeline.get_state(5 * Gst.SECOND)
        except Exception:
            LOGGER.debug("GStreamer NULL-state wait failed for %s", state.source_id)

    def _close_source(self, source_id: str) -> None:
        with self._lock:
            state = self._states.pop(source_id, None)
        if state is None:
            return
        self._dispose_state(state)

    def _active_records(self) -> list[SourceRecord]:
        return [
            record
            for record in self.registry.list()
            if record.enabled and self.is_supported_source(record)
            and (
                self.rtsp_enabled
                or not VideoFileIngestor.is_rtsp_uri(record.source_uri or "")
            )
        ]

    def _sync_sources(self) -> None:
        records = self._active_records()
        by_id = {record.source_id: record for record in records}
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
                state = self._states.get(record.source_id)
                retry_after = self._retry_after.get(record.source_id, 0.0)
            if state is not None and (
                state.source_uri != record.source_uri
                or state.frame_width != record.frame_width
                or state.frame_height != record.frame_height
            ):
                self._close_source(record.source_id)
                state = None
            if state is not None or retry_after > time.monotonic():
                continue
            try:
                self._open_source(record)
            except Exception as exc:
                self._open_failures += 1
                safe_uri = VideoFileIngestor.redact_uri(record.source_uri or "")
                self._last_error = (
                    f"Could not open {record.source_id} ({safe_uri}): "
                    f"{type(exc).__name__}: {exc}"
                )
                self._retry_after[record.source_id] = (
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
            frames = [state.latest_frame for state in selected]
            source_ids = [state.source_id for state in selected]
            frame_indexes = [state.frame_index for state in selected]
            source_times = [state.source_time_seconds for state in selected]
            metadata = [
                {
                    "source_uri": state.display_uri,
                    "source_type": state.source_type,
                    "frame_width": state.frame_width,
                    "frame_height": state.frame_height,
                    "ingest_backend": "deepstream",
                }
                for state in selected
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
        submit_interval = 1.0 / self.target_fps
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
                "running": self._thread is not None and self._thread.is_alive(),
                "target_fps": self.target_fps,
                "loop": self.loop,
                "rtsp_enabled": self.rtsp_enabled,
                "rtsp_transport": self.rtsp_transport,
                "rtsp_latency_ms": self.rtsp_latency_ms,
                "rtsp_stall_timeout_seconds": self.rtsp_stall_timeout_seconds,
                "rounds_submitted": self._rounds_submitted,
                "frames_submitted": self._frames_submitted,
                "open_failures": self._open_failures,
                "reconnects": self._reconnects,
                "last_error": self._last_error,
                "sources": {
                    source_id: {
                        "source_uri": state.display_uri,
                        "source_type": state.source_type,
                        "frame_width": state.frame_width,
                        "frame_height": state.frame_height,
                        "decoded_samples": state.decoded_samples,
                        "received_frames": state.received_frames,
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
