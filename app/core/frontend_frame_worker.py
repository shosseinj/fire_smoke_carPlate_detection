from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from app.core.latest_buffer import LatestPerSourceBuffer
from app.core.types import FramePacket

LOGGER = logging.getLogger(__name__)


def _redact_source_id(source_id: str) -> str:
    if not source_id.strip().lower().startswith(("rtsp://", "rtsps://")):
        return source_id
    parsed = urlsplit(source_id)
    host = parsed.hostname or "camera"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    credentials = "***:***@" if parsed.username is not None else ""
    return urlunsplit((parsed.scheme, f"{credentials}{host}", parsed.path, "", ""))


class FrontendFrameWorker:
    """Publish original-resolution frames without blocking ingestion."""

    def __init__(
        self,
        *,
        publish_callback: Callable[[FramePacket], None],
        publish_fullscreen_callback: Callable[[FramePacket], None] | None = None,
        queue_capacity: int = 32,
    ) -> None:
        self.publish_callback = publish_callback
        self.publish_fullscreen_callback = (
            publish_fullscreen_callback or publish_callback
        )
        self.queue_capacity = max(1, int(queue_capacity))
        self._buffer = LatestPerSourceBuffer(
            policy="latest_per_source",
            capacity=self.queue_capacity,
        )
        self._fullscreen_buffer = LatestPerSourceBuffer(
            policy="latest_per_source",
            capacity=self.queue_capacity,
        )

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fullscreen_thread: threading.Thread | None = None
        self._stats_lock = threading.Lock()

        self._submitted = 0
        self._published = 0
        self._wall_submitted = 0
        self._wall_published = 0
        self._fullscreen_submitted = 0
        self._fullscreen_published = 0
        self._last_error: str | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop.clear()

        self._thread = threading.Thread(
            target=self._run_profile,
            args=(self._buffer, self.publish_callback, "wall"),
            name="frontend-frame-worker",
            daemon=True,
        )
        self._fullscreen_thread = threading.Thread(
            target=self._run_profile,
            args=(
                self._fullscreen_buffer,
                self.publish_fullscreen_callback,
                "fullscreen",
            ),
            name="frontend-fullscreen-frame-worker",
            daemon=True,
        )
        self._thread.start()
        self._fullscreen_thread.start()

    def close(self) -> None:
        self._stop.set()
        self._buffer.close()
        self._fullscreen_buffer.close()

        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._fullscreen_thread is not None:
            self._fullscreen_thread.join(timeout=5.0)

        self._thread = None
        self._fullscreen_thread = None

    def submit_frame(
        self,
        *,
        source_id: str,
        frame: np.ndarray,
        frame_index: int,
        source_time_seconds: float | None,
        source_type: str,
        ingest_backend: str,
    ) -> bool:
        return self.submit_wall_frame(
            source_id=source_id,
            frame=frame,
            frame_index=frame_index,
            source_time_seconds=source_time_seconds,
            source_type=source_type,
            ingest_backend=ingest_backend,
        )

    def _submit_profile(
        self,
        *,
        profile: str,
        source_id: str,
        frame: np.ndarray,
        frame_index: int,
        source_time_seconds: float | None,
        source_type: str,
        ingest_backend: str,
    ) -> bool:
        if self._stop.is_set():
            return False

        packet = FramePacket(
            source_id=source_id,
            # Both ingestors hand off a newly decoded array and do not mutate it
            # afterwards. Avoid another full-resolution copy on the ingest path.
            frame=np.ascontiguousarray(frame),
            round_sequence=int(frame_index),
            frame_index=int(frame_index),
            captured_monotonic=time.monotonic(),
            captured_at_utc=datetime.now(timezone.utc).isoformat(),
            source_time_seconds=source_time_seconds,
            metadata={
                "source_uri": _redact_source_id(source_id),
                "source_type": source_type,
                "source_frame_width": int(frame.shape[1]),
                "source_frame_height": int(frame.shape[0]),
                "ingest_backend": ingest_backend,
                "stream_mode": "video-stream",
                "stream_profile": profile,
            },
        )

        with self._stats_lock:
            self._submitted += 1
            if profile == "fullscreen":
                self._fullscreen_submitted += 1
            else:
                self._wall_submitted += 1
        return (
            self._fullscreen_buffer.put(packet)
            if profile == "fullscreen"
            else self._buffer.put(packet)
        )

    def submit_wall_frame(self, **payload: object) -> bool:
        return self._submit_profile(profile="wall", **payload)  # type: ignore[arg-type]

    def submit_fullscreen_frame(self, **payload: object) -> bool:
        return self._submit_profile(
            profile="fullscreen", **payload  # type: ignore[arg-type]
        )

    def _run_profile(
        self,
        buffer: LatestPerSourceBuffer,
        publish_callback: Callable[[FramePacket], None],
        profile: str,
    ) -> None:
        while not self._stop.is_set():
            packets = buffer.take_batch(
                maximum=self.queue_capacity,
                max_wait_seconds=0.005,
            )
            if not packets:
                continue

            for packet in packets:
                try:
                    publish_callback(packet)

                    with self._stats_lock:
                        self._published += 1
                        if profile == "fullscreen":
                            self._fullscreen_published += 1
                        else:
                            self._wall_published += 1
                        self._last_error = None

                except Exception as exc:
                    with self._stats_lock:
                        self._last_error = f"{type(exc).__name__}: {exc}"

                    LOGGER.exception(
                        "Frontend frame publication failed source=%s",
                        _redact_source_id(packet.source_id),
                    )

    def status(self) -> dict[str, object]:
        buffer_stats = self._buffer.stats()
        fullscreen_buffer_stats = self._fullscreen_buffer.stats()
        with self._stats_lock:
            submitted = self._submitted
            published = self._published
            last_error = self._last_error
            wall_submitted = self._wall_submitted
            wall_published = self._wall_published
            fullscreen_submitted = self._fullscreen_submitted
            fullscreen_published = self._fullscreen_published

        return {
            "enabled": True,
            "running": (
                self._thread is not None
                and self._thread.is_alive()
            ),
            "queue_policy": buffer_stats.policy,
            "queue_size": buffer_stats.queue_depth,
            "queue_capacity": self.queue_capacity,
            "pending_sources": buffer_stats.pending_sources,
            "submitted": submitted,
            "published": published,
            "dropped": (
                buffer_stats.stale_replaced
                + buffer_stats.full_rejections
                + buffer_stats.rejected_after_close
            ),
            "replaced": buffer_stats.stale_replaced,
            "replaced_by_source": buffer_stats.stale_replaced_by_source,
            "rejected_after_close": buffer_stats.rejected_after_close,
            "last_error": last_error,
            "wall_submitted": wall_submitted,
            "wall_published": wall_published,
            "fullscreen_submitted": fullscreen_submitted,
            "fullscreen_published": fullscreen_published,
            "fullscreen_queue_size": fullscreen_buffer_stats.queue_depth,
            "fullscreen_replaced": fullscreen_buffer_stats.stale_replaced,
        }
