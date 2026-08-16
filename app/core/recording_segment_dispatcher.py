from __future__ import annotations

import threading
from typing import Any

from app.core.detection_event_schemas import RecordingSegmentEvent
from app.core.recording_segment_store import RecordingSegmentStore


class RecordingSegmentDispatcher:
    def __init__(self, store: RecordingSegmentStore, redis_client: Any, stream: str,
                 *, poll_seconds: float = .5) -> None:
        self.store, self.redis, self.stream, self.poll_seconds = store, redis_client, stream, max(.05, poll_seconds)
        self._stop = threading.Event(); self._thread: threading.Thread | None = None
        self.published = self.failed = 0; self.error: str | None = None

    def dispatch_once(self) -> int:
        sent = 0
        for segment in self.store.unpublished():
            event = RecordingSegmentEvent(
                segment_id=segment.segment_id, camera_id=segment.camera_id, bucket=segment.bucket,
                object_key=segment.object_key, started_at_utc=segment.started_at_utc,
                ended_at_utc=segment.ended_at_utc, frame_width=segment.frame_width,
                frame_height=segment.frame_height, fps=segment.fps, status="ready")
            try:
                self.redis.xadd(self.stream, {"event": event.model_dump_json()})
                self.store.mark_published(segment.segment_id)
                sent += 1; self.published += 1
            except Exception as exc:
                self.failed += 1; self.error = type(exc).__name__; break
        return sent

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="segment-manifest-outbox", daemon=True); self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try: self.dispatch_once(); self.error = None
            except Exception as exc: self.error = type(exc).__name__
            self._stop.wait(self.poll_seconds)

    def close(self, timeout: float = 5.0) -> bool:
        self._stop.set()
        if self._thread is not None: self._thread.join(timeout)
        return self._thread is None or not self._thread.is_alive()

    def status(self) -> dict[str, Any]:
        return {"running": bool(self._thread and self._thread.is_alive()), "published": self.published,
                "failed": self.failed, "error": self.error}
