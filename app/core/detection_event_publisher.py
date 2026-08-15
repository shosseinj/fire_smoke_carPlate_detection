from __future__ import annotations

import queue
import threading
import time
from typing import Mapping, Protocol

from app.core.detection_event_schemas import (DetectionEvent, FireSmokeDetectionEvent,
    HumanDetectionEvent, PlateDetectionEvent, RecordingSegmentEvent)


class XAddRedis(Protocol):
    def xadd(self, name: str, fields: Mapping[str, str]) -> object: ...


_ROUTES = {HumanDetectionEvent: "human", FireSmokeDetectionEvent: "fire_smoke",
           PlateDetectionEvent: "plate", RecordingSegmentEvent: "recording_segment"}


class DetectionEventPublisher:
    def __init__(self, redis_client: XAddRedis, streams: Mapping[str, str], *, queue_capacity: int = 256,
                 max_retries: int = 2, retry_backoff_seconds: float = 0.05, close_timeout_seconds: float = 1.0) -> None:
        if queue_capacity <= 0 or max_retries < 0 or retry_backoff_seconds < 0 or close_timeout_seconds < 0:
            raise ValueError("invalid publisher bounds")
        if set(streams) != set(_ROUTES.values()) or any(not isinstance(v, str) or not v.strip() for v in streams.values()):
            raise ValueError("a complete nonblank four-stream mapping is required")
        self._redis, self._streams = redis_client, dict(streams)
        self._queue: queue.Queue[DetectionEvent | object] = queue.Queue(maxsize=queue_capacity)
        self._stop = object()
        self._stop_event = threading.Event()
        self._metrics = {key: 0 for key in ("published", "queued", "dropped", "retried", "failed")}
        self._lock = threading.Lock(); self._closed = False; self._error: str | None = None
        self._max_retries, self._backoff, self._close_timeout = max_retries, retry_backoff_seconds, close_timeout_seconds
        self._worker = threading.Thread(target=self._run, name="detection-event-publisher", daemon=True)
        self._worker.start()

    def publish(self, event: DetectionEvent) -> bool:
        if type(event) not in _ROUTES: raise TypeError("unsupported detection event")
        with self._lock:
            if self._closed: self._metrics["dropped"] += 1; return False
            try: self._queue.put_nowait(event)
            except queue.Full: self._metrics["dropped"] += 1; return False
            self._metrics["queued"] += 1; return True

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._stop: return
                route = _ROUTES[type(item)]
                payload = item.model_dump_json()  # type: ignore[union-attr]
                for attempt in range(self._max_retries + 1):
                    if self._stop_event.is_set():
                        with self._lock: self._metrics["failed"] += 1
                        break
                    try:
                        self._redis.xadd(self._streams[route], {"event": payload})
                        with self._lock: self._metrics["published"] += 1
                        break
                    except Exception:
                        if attempt == self._max_retries:
                            with self._lock: self._metrics["failed"] += 1
                        else:
                            with self._lock: self._metrics["retried"] += 1
                            if self._stop_event.wait(self._backoff):
                                with self._lock: self._metrics["failed"] += 1
                                break
            except Exception:
                with self._lock: self._metrics["failed"] += 1
            finally: self._queue.task_done()

    @property
    def worker_alive(self) -> bool: return self._worker.is_alive()

    def status(self) -> dict[str, int | bool]:
        with self._lock:
            return {"enabled": True, **self._metrics, "running": self._worker.is_alive(),
                    "closed": self._closed and not self._worker.is_alive(), "error": self._error}

    def close(self) -> bool:
        with self._lock:
            if self._closed:
                stopped = not self._worker.is_alive()
                if stopped: self._error = None
                return stopped
            self._closed = True
            self._stop_event.set()
        # Discard pending work so the stop marker can always be inserted.
        while True:
            try:
                item = self._queue.get_nowait(); self._queue.task_done()
                if item is not self._stop:
                    with self._lock: self._metrics["dropped"] += 1
            except queue.Empty: break
        self._queue.put_nowait(self._stop)
        self._worker.join(timeout=self._close_timeout)
        with self._lock:
            self._error = "shutdown_timed_out" if self._worker.is_alive() else None
        return not self._worker.is_alive()
