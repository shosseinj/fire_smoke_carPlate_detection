from __future__ import annotations

import multiprocessing as mp
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.core.source_registry import RTSP, SourceRecord, canonical_source_type
from app.core.video_ingestor import VideoFileIngestor


class _SingleSourceRegistry:
    def __init__(self, record: SourceRecord) -> None:
        self._record = record
        self.revision = 1

    def list(self) -> list[SourceRecord]:
        return [self._record]

    def get(self, source_id: str) -> SourceRecord | None:
        return self._record if source_id == self._record.source_uri else None


class _IpcFrontend:
    def __init__(self, events: Any) -> None:
        self.events = events

    def submit_frame(self, **payload: Any) -> bool:
        return _put_latest(self.events, ("frontend", payload))


class _IpcRouter:
    def __init__(self, events: Any) -> None:
        self.events = events

    def submit_round(self, **payload: Any) -> dict[str, int]:
        _put_latest(self.events, ("ai_round", payload))
        return {"accepted_sources": len(payload.get("source_ids", ()))}


class _IpcRawRouter:
    enabled = True

    def __init__(self, events: Any) -> None:
        self.events = events

    def publish_packet(self, **payload: Any) -> None:
        _put_latest(self.events, ("raw_packet", payload))


def _put_latest(events: Any, item: tuple[str, Any]) -> bool:
    try:
        events.put_nowait(item)
        return True
    except queue.Full:
        try:
            events.get_nowait()
        except queue.Empty:
            pass
        try:
            events.put_nowait(item)
            return True
        except queue.Full:
            return False


def _rtsp_child_main(
    record_value: dict[str, Any],
    settings: dict[str, Any],
    events: Any,
    stop_event: Any,
) -> None:
    # rtspsrc uses GSocketClient, which otherwise loads libgiolibproxy and
    # libproxy for camera-LAN URIs. The image's libproxy path segfaults inside
    # libstdc++ during proxy discovery. RTSP cameras are direct LAN endpoints,
    # so select GIO's built-in direct resolver before importing GStreamer.
    os.environ["GIO_USE_PROXY_RESOLVER"] = "dummy"
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    # Importing Gst/DeepStream happens only inside this camera process.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    from app.core.deepstream_ingestor import DeepStreamIngestor

    record = SourceRecord.from_dict(record_value)
    router = _IpcRouter(events)
    frontend = _IpcFrontend(events)
    raw_router = _IpcRawRouter(events) if settings.pop("raw_enabled") else None
    ingestor = DeepStreamIngestor(
        registry=_SingleSourceRegistry(record),
        router=router,  # type: ignore[arg-type]
        frontend_frame_worker=frontend,  # type: ignore[arg-type]
        raw_stream_router=raw_router,  # type: ignore[arg-type]
        demand_controller=None,
        source_type_filter=RTSP,
        max_sources=1,
        max_active_sources=1,
        max_source_opens_per_sync=1,
        source_allowlist=(record.source_uri,),
        source_open_stagger_seconds=0.0,
        **settings,
    )
    try:
        ingestor.start()
        while not stop_event.wait(0.5):
            _put_latest(events, ("status", ingestor.status()))
    finally:
        ingestor.close()


@dataclass
class _Child:
    record: SourceRecord
    process: Any | None = None
    stop_event: Any | None = None
    events: Any | None = None
    state: str = "restarting"
    exit_code: int | None = None
    restart_count: int = 0
    consecutive_exit_139: int = 0
    restart_at: float = 0.0
    last_error: str | None = None
    last_frame_time: float | None = None
    last_started: float | None = None


class RtspProcessSupervisor:
    """Supervises one disposable native GStreamer process per RTSP source."""

    def __init__(
        self,
        *,
        registry: Any,
        router: Any,
        project_root: Path,
        frontend_frame_worker: Any,
        raw_stream_router: Any | None = None,
        demand_controller: Any | None = None,
        process_factory: Callable[..., Any] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        backoff_seconds: tuple[float, ...] = (2.0, 5.0, 10.0, 30.0),
        segfault_quarantine_threshold: int = 3,
        quarantine_seconds: float = 300.0,
        max_sources: int = 256,
        max_active_sources: int = 256,
        source_allowlist: tuple[str, ...] | None = None,
        rtsp_enabled: bool = True,
        **child_settings: Any,
    ) -> None:
        self.registry = registry
        self.router = router
        self.project_root = project_root
        self.frontend_frame_worker = frontend_frame_worker
        self.raw_stream_router = raw_stream_router
        self.demand_controller = demand_controller
        self.max_sources = max_sources
        self.max_active_sources = max_active_sources
        self.source_allowlist = frozenset(source_allowlist or ())
        self.rtsp_enabled = rtsp_enabled
        self._clock = monotonic
        self._backoff = backoff_seconds
        self._quarantine_threshold = max(1, segfault_quarantine_threshold)
        self._quarantine_seconds = max(1.0, quarantine_seconds)
        self._ctx = mp.get_context("spawn")
        self._process_factory = process_factory or self._ctx.Process
        self._children: dict[str, _Child] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._source_start_stagger_seconds = max(
            0.0,
            float(child_settings.get("source_open_stagger_seconds", 0.5)),
        )
        self._next_child_start_monotonic = 0.0
        self._child_settings = {
            "project_root": project_root,
            "video_only_mode": bool(child_settings.get("video_only_mode", False)),
            "gpu_resize_enabled": bool(child_settings.get("gpu_resize_enabled", True)),
            "loop": bool(child_settings.get("loop", True)),
            "rtsp_transport": child_settings.get("rtsp_transport", "tcp"),
            "rtsp_latency_ms": int(child_settings.get("rtsp_latency_ms", 500)),
            "rtsp_reconnect_seconds": float(
                child_settings.get("rtsp_reconnect_seconds", 3.0)
            ),
            "rtsp_stall_timeout_seconds": int(
                child_settings.get("rtsp_stall_timeout_seconds", 30)
            ),
            "skip_taskless_sources": bool(
                child_settings.get("skip_taskless_sources", True)
            ),
            "rtsp_enabled": True,
            "raw_enabled": bool(
                raw_stream_router is not None
                and getattr(raw_stream_router, "enabled", False)
            ),
        }

    def _required_records(self) -> list[SourceRecord]:
        if not self.rtsp_enabled:
            return []
        result = []
        for record in self.registry.list():
            if not record.enabled or canonical_source_type(
                record.source_uri, record.source_type
            ) != RTSP:
                continue
            if self.source_allowlist and (
                record.source_uri not in self.source_allowlist
                and VideoFileIngestor.redact_uri(record.source_uri)
                not in self.source_allowlist
            ):
                continue
            if self.demand_controller is not None and not (
                self.demand_controller.video_required(record.source_uri)
                or (
                    not self._child_settings["video_only_mode"]
                    and self.demand_controller.ai_required()
                )
            ):
                continue
            result.append(record)
        return result[: min(self.max_sources, self.max_active_sources)]

    def _start_child(self, child: _Child) -> None:
        child.stop_event = self._ctx.Event()
        child.events = self._ctx.Queue(maxsize=8)
        settings = dict(self._child_settings)
        child.process = self._process_factory(
            target=_rtsp_child_main,
            args=(child.record.to_dict(), settings, child.events, child.stop_event),
            name=f"rtsp-{child.record.id or 'source'}",
            daemon=True,
        )
        child.process.start()
        child.state = "running"
        child.last_started = self._clock()

    def _handle_exit(self, child: _Child, exit_code: int) -> None:
        now = self._clock()
        child.exit_code = exit_code
        child.restart_count += 1
        child.last_error = f"RTSP child exited with code {exit_code}"
        child.consecutive_exit_139 = (
            child.consecutive_exit_139 + 1
            if exit_code in {139, -11}
            else 0
        )
        if child.consecutive_exit_139 >= self._quarantine_threshold:
            child.state = "quarantined"
            child.restart_at = now + self._quarantine_seconds
        else:
            child.state = "restarting"
            delay = self._backoff[min(child.restart_count - 1, len(self._backoff) - 1)]
            child.restart_at = now + delay
        child.process = None
        child.stop_event = None
        if child.events is not None:
            child.events.cancel_join_thread()
            child.events.close()
            child.events = None

    def _forward_event(self, kind: str, payload: Any) -> None:
        if kind == "frontend":
            self.frontend_frame_worker.submit_frame(**payload)
            source_id = payload["source_id"]
            with self._lock:
                child = self._children.get(source_id)
                if child is not None:
                    child.last_frame_time = self._clock()
                    child.consecutive_exit_139 = 0
            return
        if kind == "ai_round":
            self.router.submit_round(**payload)
        elif kind == "raw_packet" and self.raw_stream_router is not None:
            self.raw_stream_router.publish_packet(**payload)

    def poll_once(self) -> None:
        required = {record.source_uri: record for record in self._required_records()}
        now = self._clock()
        start_available = now >= self._next_child_start_monotonic
        with self._lock:
            for source_id, child in list(self._children.items()):
                if source_id not in required:
                    self._stop_child(child)
                    del self._children[source_id]
                    continue
                if child.process is not None:
                    exit_code = child.process.exitcode
                    if exit_code is not None:
                        self._handle_exit(child, int(exit_code))
                if (
                    child.process is None
                    and now >= child.restart_at
                    and start_available
                ):
                    self._start_child(child)
                    start_available = False
                    self._next_child_start_monotonic = (
                        now + self._source_start_stagger_seconds
                    )
            for source_id, record in required.items():
                if source_id not in self._children:
                    child = _Child(record=record)
                    self._children[source_id] = child
                    if start_available:
                        self._start_child(child)
                        start_available = False
                        self._next_child_start_monotonic = (
                            now + self._source_start_stagger_seconds
                        )
        with self._lock:
            event_queues = [
                child.events
                for child in self._children.values()
                if child.events is not None and child.process is not None
            ]
        for events in event_queues:
            while True:
                try:
                    kind, payload = events.get_nowait()
                except queue.Empty:
                    break
                self._forward_event(kind, payload)

    @staticmethod
    def _stop_child(child: _Child) -> None:
        process = child.process
        if process is None:
            return
        if child.stop_event is not None:
            child.stop_event.set()
        process.join(timeout=3.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=1.0)
        child.process = None
        if child.events is not None:
            child.events.cancel_join_thread()
            child.events.close()
            child.events = None
        child.state = "stopped"

    def _run(self) -> None:
        while not self._stop.wait(0.1):
            self.poll_once()
        with self._lock:
            for child in self._children.values():
                self._stop_child(child)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="rtsp-process-supervisor", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._thread = None

    def restart_source(self, source_id: str) -> bool:
        record = self.registry.get(source_id)
        if record is None or canonical_source_type(
            record.source_uri, record.source_type
        ) != RTSP:
            return False
        with self._lock:
            child = self._children.get(source_id)
            if child is not None:
                self._stop_child(child)
                child.state = "restarting"
                child.restart_at = self._clock()
                child.consecutive_exit_139 = 0
        return True

    def status(self) -> dict[str, Any]:
        now = self._clock()
        if not self._lock.acquire(timeout=0.25):
            return {
                "enabled": self.rtsp_enabled,
                "backend": "deepstream-process-isolated",
                "running": True,
                "active_source_count": 0,
                "status_stale": True,
                "sources": {},
            }
        try:
            sources = {}
            for source_id, child in self._children.items():
                process = child.process
                sources[VideoFileIngestor.redact_uri(source_id)] = {
                    "pid": process.pid if process is not None else None,
                    "state": child.state,
                    "running": bool(process is not None and process.is_alive()),
                    "failed": child.exit_code is not None,
                    "exit_code": child.exit_code,
                    "restart_count": child.restart_count,
                    "last_error": child.last_error,
                    "last_frame_time": child.last_frame_time,
                    "last_frame_age_seconds": (
                        round(now - child.last_frame_time, 3)
                        if child.last_frame_time is not None
                        else None
                    ),
                    "restart_in_seconds": (
                        round(max(0.0, child.restart_at - now), 3)
                        if child.state in {"restarting", "quarantined"}
                        else None
                    ),
                    "consecutive_exit_139": child.consecutive_exit_139,
                }
            return {
                "enabled": self.rtsp_enabled,
                "backend": "deepstream-process-isolated",
                "running": self._thread is not None and self._thread.is_alive(),
                "active_source_count": sum(
                    1 for value in sources.values() if value["running"]
                ),
                "source_start_stagger_seconds": (
                    self._source_start_stagger_seconds
                ),
                "sources": sources,
            }
        finally:
            self._lock.release()
