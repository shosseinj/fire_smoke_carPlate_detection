from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FrontendFrameJob:
    source_id: str
    frame: np.ndarray
    frame_index: int
    source_time_seconds: float | None


class FrontendFrameWorker:
    """Publish original-resolution frames without blocking ingestion."""

    def __init__(
        self,
        *,
        publish_callback: Callable[
            [str, np.ndarray, int, float | None],
            None,
        ],
        queue_capacity: int = 32,
    ) -> None:
        self.publish_callback = publish_callback
        self.queue_capacity = max(1, int(queue_capacity))

        self._queue: queue.Queue[FrontendFrameJob] = queue.Queue(
            maxsize=self.queue_capacity
        )

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats_lock = threading.Lock()

        self._submitted = 0
        self._published = 0
        self._dropped = 0
        self._last_error: str | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop.clear()

        self._thread = threading.Thread(
            target=self._run,
            name="frontend-frame-worker",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=5.0)

        self._thread = None

        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

    def submit_frame(
        self,
        *,
        source_id: str,
        frame: np.ndarray,
        frame_index: int,
        source_time_seconds: float | None,
    ) -> None:
        if self._stop.is_set():
            return

        job = FrontendFrameJob(
            source_id=source_id,
            frame=np.ascontiguousarray(frame).copy(),
            frame_index=int(frame_index),
            source_time_seconds=source_time_seconds,
        )

        with self._stats_lock:
            self._submitted += 1

        try:
            self._queue.put_nowait(job)
            return
        except queue.Full:
            pass

        try:
            self._queue.get_nowait()
            self._queue.task_done()

            with self._stats_lock:
                self._dropped += 1

        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(job)
        except queue.Full:
            with self._stats_lock:
                self._dropped += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue

            try:
                self.publish_callback(
                    job.source_id,
                    job.frame,
                    job.frame_index,
                    job.source_time_seconds,
                )

                with self._stats_lock:
                    self._published += 1
                    self._last_error = None

            except Exception as exc:
                with self._stats_lock:
                    self._last_error = (
                        f"{type(exc).__name__}: {exc}"
                    )

                LOGGER.exception(
                    "Frontend frame publication failed source=%s",
                    job.source_id,
                )

            finally:
                self._queue.task_done()

    def status(self) -> dict[str, object]:
        with self._stats_lock:
            submitted = self._submitted
            published = self._published
            dropped = self._dropped
            last_error = self._last_error

        return {
            "enabled": True,
            "running": (
                self._thread is not None
                and self._thread.is_alive()
            ),
            "queue_size": self._queue.qsize(),
            "queue_capacity": self.queue_capacity,
            "submitted": submitted,
            "published": published,
            "dropped": dropped,
            "last_error": last_error,
        }