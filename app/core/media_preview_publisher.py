from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.core.media_preview import (
    preview_publish_uri,
    preview_stream_path,
    redact_rtsp_credentials,
)
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
            "Media preview requires Linux and GStreamer Python bindings"
        ) from exc
    Gst.init(None)
    return Gst, GLib


@dataclass(slots=True)
class PreviewPublisherState:
    source_id: str
    source_uri: str
    source_type: str
    path: str
    pipeline: Any
    source: Any
    bus: Any
    bus_handler_id: int
    source_pad_handler_id: int
    active: bool = False
    errors: int = 0
    warnings: int = 0
    eos_count: int = 0
    reconnects: int = 0
    last_error: str | None = None
    started_monotonic: float = 0.0


class MediaPreviewPublisher:
    """Publishes source H264 in pipelines independent from AI ingestion."""

    REQUIRED_ELEMENTS = (
        "filesrc",
        "qtdemux",
        "rtspsrc",
        "rtph264depay",
        "queue",
        "h264parse",
        "capsfilter",
        "identity",
        "rtspclientsink",
    )

    def __init__(
        self,
        *,
        registry: SourceRegistry,
        project_root: Path,
        publish_base: str = "rtsp://mediamtx:8554",
        enabled: bool = True,
        rtsp_enabled: bool = True,
        rtsp_transport: str = "tcp",
        rtsp_latency_ms: int = 500,
        reconnect_seconds: float = 3.0,
        gst_loader: Callable[[], tuple[Any, Any]] = _load_gstreamer,
    ) -> None:
        self.registry = registry
        self.project_root = project_root
        self.publish_base = publish_base.strip()
        self.enabled = bool(enabled)
        self.rtsp_enabled = bool(rtsp_enabled)
        self.rtsp_transport = rtsp_transport.strip().lower()
        self.rtsp_latency_ms = max(0, int(rtsp_latency_ms))
        self.reconnect_seconds = max(0.5, float(reconnect_seconds))
        self.gst_loader = gst_loader
        preview_publish_uri(self.publish_base, "validation")

        self._gst: Any | None = None
        self._glib: Any | None = None
        self._context: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started = threading.Event()
        self._lock = threading.RLock()
        self._states: dict[str, PreviewPublisherState] = {}
        self._failed: set[str] = set()
        self._retry_after: dict[str, float] = {}
        self._reconnect_counts: dict[str, int] = {}
        self._error_counts: dict[str, int] = {}
        self._warning_counts: dict[str, int] = {}
        self._eos_counts: dict[str, int] = {}
        self._last_errors: dict[str, str] = {}
        self._open_failures = 0
        self._last_error: str | None = None

    def _require_runtime(self) -> tuple[Any, Any]:
        if self._gst is None or self._glib is None:
            raise RuntimeError("Preview publisher runtime is not initialized")
        return self._gst, self._glib

    def _make(self, factory: str, name: str) -> Any:
        Gst, _ = self._require_runtime()
        element = Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"Required GStreamer element is unavailable: {factory}")
        return element

    @staticmethod
    def _set_if_supported(element: Any, name: str, value: Any) -> None:
        if element.find_property(name) is not None:
            element.set_property(name, value)

    @staticmethod
    def _safe_name(source_id: str) -> str:
        display_uri = redact_rtsp_credentials(source_id)
        return "".join(char if char.isalnum() else "_" for char in display_uri)

    def _local_path(self, source_uri: str) -> Path:
        path = Path(source_uri).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    def _active_records(self) -> list[SourceRecord]:
        return [
            record
            for record in self.registry.list()
            if record.enabled
            and record.source_uri
            and VideoFileIngestor.is_video_source(record)
            and (
                self.rtsp_enabled
                or not VideoFileIngestor.is_rtsp_uri(record.source_uri)
            )
        ]

    def _on_pad_added(self, _: Any, pad: Any, target: Any, source_id: str) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        caps_text = caps.to_string().lower() if caps is not None else ""
        if "h264" not in caps_text:
            if caps_text.startswith("video/") or "media=(string)video" in caps_text:
                self._fail(source_id, "Preview supports H264 sources only")
            return
        sink_pad = target.get_static_pad("sink")
        if sink_pad is None or sink_pad.is_linked():
            return
        Gst, _ = self._require_runtime()
        if pad.link(sink_pad) != Gst.PadLinkReturn.OK:
            self._fail(source_id, "Could not link H264 preview source pad")

    def _fail(self, source_id: str, message: str, *, immediate: bool = False) -> None:
        safe_message = redact_rtsp_credentials(message)
        display_uri = redact_rtsp_credentials(source_id)
        with self._lock:
            state = self._states.get(source_id)
            if state is not None:
                state.active = False
                state.errors += 1
                state.last_error = safe_message
            self._error_counts[source_id] = self._error_counts.get(source_id, 0) + 1
            self._last_errors[source_id] = safe_message
            self._failed.add(source_id)
            self._retry_after[source_id] = (
                time.monotonic()
                if immediate
                else time.monotonic() + self.reconnect_seconds
            )
            self._last_error = f"{display_uri}: {safe_message}"
        LOGGER.warning("Preview publisher %s: %s", display_uri, safe_message)

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
            self._fail(source_id, detail)
        elif message.type == Gst.MessageType.EOS:
            with self._lock:
                state.eos_count += 1
                self._eos_counts[source_id] = self._eos_counts.get(source_id, 0) + 1
            self._fail(source_id, "End of preview stream; restarting", immediate=True)
        elif message.type == Gst.MessageType.WARNING:
            warning, _ = message.parse_warning()
            safe_warning = redact_rtsp_credentials(str(warning))
            with self._lock:
                state.warnings += 1
                state.last_error = safe_warning
                self._warning_counts[source_id] = (
                    self._warning_counts.get(source_id, 0) + 1
                )
                self._last_errors[source_id] = safe_warning
        elif (
            message.type == Gst.MessageType.STATE_CHANGED
            and getattr(message, "src", None) is state.pipeline
        ):
            _, new_state, _ = message.parse_state_changed()
            if new_state == Gst.State.PLAYING:
                with self._lock:
                    state.active = True
                    state.last_error = None
                    self._last_errors.pop(source_id, None)

    def _open(self, record: SourceRecord) -> None:
        Gst, _ = self._require_runtime()
        assert record.source_uri is not None
        is_rtsp = VideoFileIngestor.is_rtsp_uri(record.source_uri)
        safe_id = self._safe_name(record.source_uri)
        pipeline = Gst.Pipeline.new(f"preview_pipeline_{safe_id}")
        if pipeline is None:
            raise RuntimeError("Could not create preview pipeline")
        source = self._make("rtspsrc" if is_rtsp else "filesrc", f"preview_source_{safe_id}")
        demux_or_depay = self._make(
            "rtph264depay" if is_rtsp else "qtdemux",
            f"preview_input_{safe_id}",
        )
        queue = self._make("queue", f"preview_queue_{safe_id}")
        parser = self._make("h264parse", f"preview_parser_{safe_id}")
        output_caps = self._make("capsfilter", f"preview_caps_{safe_id}")
        pacer = self._make("identity", f"preview_pacer_{safe_id}")
        sink = self._make("rtspclientsink", f"preview_sink_{safe_id}")
        try:
            if is_rtsp:
                source.set_property("location", record.source_uri)
                self._set_if_supported(
                    source,
                    "protocols",
                    4 if self.rtsp_transport == "tcp" else 1,
                )
                self._set_if_supported(source, "latency", self.rtsp_latency_ms)
                self._set_if_supported(source, "drop-on-latency", True)
            else:
                local_path = self._local_path(record.source_uri)
                if not local_path.is_file():
                    raise FileNotFoundError(f"Preview video file was not found: {record.source_uri}")
                source.set_property("location", str(local_path))
            queue.set_property("leaky", 2)
            queue.set_property("max-size-buffers", 2)
            queue.set_property("max-size-bytes", 0)
            queue.set_property("max-size-time", 0)
            self._set_if_supported(queue, "flush-on-eos", True)
            parser.set_property("config-interval", -1)
            self._set_if_supported(parser, "disable-passthrough", True)
            output_caps.set_property(
                "caps",
                Gst.Caps.from_string(
                    "video/x-h264,stream-format=byte-stream,alignment=au"
                ),
            )
            pacer.set_property("sync", not is_rtsp)
            sink.set_property(
                "location", preview_publish_uri(self.publish_base, record.source_uri)
            )
            self._set_if_supported(sink, "protocols", 4)
            self._set_if_supported(sink, "latency", 0)
            for element in (
                source,
                demux_or_depay,
                queue,
                parser,
                output_caps,
                pacer,
                sink,
            ):
                pipeline.add(element)
            if is_rtsp:
                if not demux_or_depay.link(queue):
                    raise RuntimeError("Could not link RTSP H264 depayloader")
            elif not source.link(demux_or_depay):
                raise RuntimeError("Could not link MP4 source to demuxer")
            if (
                not queue.link(parser)
                or not parser.link(output_caps)
                or not output_caps.link(pacer)
                or not pacer.link(sink)
            ):
                raise RuntimeError("Could not link bounded H264 preview publisher")
            source_pad_handler_id = (
                source.connect("pad-added", self._on_pad_added, demux_or_depay, record.source_uri)
                if is_rtsp
                else demux_or_depay.connect("pad-added", self._on_pad_added, queue, record.source_uri)
            )
            bus = pipeline.get_bus()
            bus.add_signal_watch()
            bus_handler_id = bus.connect("message", self._on_bus_message, record.source_uri)
            state = PreviewPublisherState(
                source_id=record.source_uri,
                source_uri=record.source_uri,
                source_type="rtsp_h264" if is_rtsp else "mp4_h264",
                path=preview_stream_path(record.source_uri),
                pipeline=pipeline,
                source=source if is_rtsp else demux_or_depay,
                bus=bus,
                bus_handler_id=bus_handler_id,
                source_pad_handler_id=source_pad_handler_id,
                reconnects=self._reconnect_counts.get(record.source_uri, 0),
                errors=self._error_counts.get(record.source_uri, 0),
                warnings=self._warning_counts.get(record.source_uri, 0),
                eos_count=self._eos_counts.get(record.source_uri, 0),
                last_error=self._last_errors.get(record.source_uri),
                started_monotonic=time.monotonic(),
            )
            with self._lock:
                self._states[record.source_uri] = state
            if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Preview pipeline refused PLAYING state")
            with self._lock:
                self._retry_after.pop(record.source_uri, None)
        except Exception:
            with self._lock:
                state = self._states.pop(record.source_uri, None)
            if state is not None:
                self._dispose(state)
            else:
                pipeline.set_state(Gst.State.NULL)
            raise

    def _dispose(self, state: PreviewPublisherState) -> None:
        Gst, _ = self._require_runtime()
        try:
            state.source.disconnect(state.source_pad_handler_id)
        except Exception:
            pass
        try:
            state.bus.disconnect(state.bus_handler_id)
            state.bus.remove_signal_watch()
        except Exception:
            pass
        state.pipeline.set_state(Gst.State.NULL)
        try:
            state.pipeline.get_state(2 * Gst.SECOND)
        except Exception:
            pass

    def _close_source(self, source_id: str) -> None:
        with self._lock:
            state = self._states.pop(source_id, None)
        if state is not None:
            self._dispose(state)

    def _sync(self) -> None:
        records = {record.source_uri: record for record in self._active_records()}
        with self._lock:
            failed = set(self._failed)
            self._failed.clear()
            current = dict(self._states)
        for source_id in failed:
            self._reconnect_counts[source_id] = self._reconnect_counts.get(source_id, 0) + 1
            self._close_source(source_id)
        for source_id, state in current.items():
            record = records.get(source_id)
            if record is None or record.source_uri != state.source_uri:
                self._close_source(source_id)
        for source_id, record in records.items():
            with self._lock:
                exists = source_id in self._states
                retry_after = self._retry_after.get(source_id, 0.0)
            if exists or retry_after > time.monotonic():
                continue
            try:
                self._open(record)
            except Exception as exc:
                self._open_failures += 1
                safe_error = redact_rtsp_credentials(f"{type(exc).__name__}: {exc}")
                display_uri = redact_rtsp_credentials(source_id)
                self._last_error = f"{display_uri}: {safe_error}"
                self._retry_after[source_id] = time.monotonic() + self.reconnect_seconds
                LOGGER.warning(
                    "Could not open preview publisher %s: %s",
                    display_uri,
                    safe_error,
                )

    def _run(self) -> None:
        _, GLib = self._require_runtime()
        self._context = GLib.MainContext.new()
        self._context.push_thread_default()
        self._started.set()
        next_sync = 0.0
        try:
            while not self._stop.is_set():
                while self._context.pending():
                    self._context.iteration(False)
                now = time.monotonic()
                if now >= next_sync:
                    self._sync()
                    next_sync = now + 0.5
                self._stop.wait(0.05)
        finally:
            with self._lock:
                source_ids = list(self._states)
            for source_id in source_ids:
                self._close_source(source_id)
            self._context.pop_thread_default()

    def start(self) -> None:
        if not self.enabled or (self._thread is not None and self._thread.is_alive()):
            return
        self._gst, self._glib = self.gst_loader()
        missing = [
            name
            for name in self.REQUIRED_ELEMENTS
            if self._gst.ElementFactory.find(name) is None
        ]
        if missing:
            raise RuntimeError("Preview GStreamer elements are missing: " + ", ".join(missing))
        self._stop.clear()
        self._started.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="media-preview-publisher",
            daemon=True,
        )
        self._thread.start()
        self._started.wait(timeout=3.0)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def status(self) -> dict[str, Any]:
        configured = {record.source_uri: record for record in self._active_records()}
        with self._lock:
            now = time.monotonic()
            return {
                "enabled": self.enabled,
                "running": self._thread is not None and self._thread.is_alive(),
                "backend": "independent_h264_remux",
                "codec_contract": "H264 MP4 and H264 RTSP only",
                "queue_capacity_buffers": 2,
                "open_failures": self._open_failures,
                "last_error": self._last_error,
                "sources": {
                    source_id: {
                        "path": preview_stream_path(source_id),
                        "source_type": (
                            state.source_type
                            if state is not None
                            else (
                                "rtsp_h264"
                                if VideoFileIngestor.is_rtsp_uri(
                                    record.source_uri or ""
                                )
                                else "mp4_h264"
                            )
                        ),
                        "active": state.active if state is not None else False,
                        "errors": self._error_counts.get(source_id, 0),
                        "warnings": self._warning_counts.get(source_id, 0),
                        "eos_count": self._eos_counts.get(source_id, 0),
                        "reconnects": self._reconnect_counts.get(source_id, 0),
                        "uptime_seconds": (
                            round(now - state.started_monotonic, 3)
                            if state is not None
                            else None
                        ),
                        "retry_in_seconds": (
                            round(max(0.0, self._retry_after.get(source_id, 0.0) - now), 3)
                            if state is None
                            else None
                        ),
                        "last_error": self._last_errors.get(source_id),
                    }
                    for source_id, record in configured.items()
                    for state in (self._states.get(source_id),)
                },
            }
