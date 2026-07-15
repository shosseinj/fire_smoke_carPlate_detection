from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2

from app.core.router import TaskRouter
from app.core.source_registry import SourceRecord, SourceRegistry

LOGGER = logging.getLogger(__name__)
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


@dataclass(slots=True)
class VideoState:
    source_id: str
    source_uri: str
    capture: Any
    fps: float
    stride: int
    frame_index: int = -1
    submitted_frames: int = 0
    loop_count: int = 0
    read_failures: int = 0
    last_error: str | None = None


class VideoFileIngestor:
    """Reads configured video files as looping cameras and feeds synchronized rounds."""

    def __init__(
        self,
        *,
        registry: SourceRegistry,
        router: TaskRouter,
        project_root: Path,
        target_fps: float = 5.0,
        loop: bool = True,
        capture_factory: Callable[[str], Any] = cv2.VideoCapture,
    ) -> None:
        self.registry = registry
        self.router = router
        self.project_root = project_root
        self.target_fps = max(0.1, target_fps)
        self.loop = loop
        self.capture_factory = capture_factory
        self._states: dict[str, VideoState] = {}
        self._thread: threading.Thread | None = None
        self._state_lock = threading.RLock()
        self._stop = threading.Event()
        self._started = threading.Event()
        self._round_sequence = 0
        self._rounds_submitted = 0
        self._frames_submitted = 0
        self._open_failures = 0
        self._last_error: str | None = None

    @staticmethod
    def is_video_source(record: SourceRecord) -> bool:
        if not record.source_uri:
            return False
        if record.metadata.get("kind") == "video_file":
            return True
        uri = record.source_uri.lower()
        return "://" not in uri and Path(uri).suffix.lower() in VIDEO_SUFFIXES

    def _resolve_uri(self, source_uri: str) -> str:
        path = Path(source_uri).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return str(path.resolve())

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._started.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="video-file-ingestor",
            daemon=True,
        )
        self._thread.start()
        self._started.wait(timeout=2.0)

    def _open(self, record: SourceRecord) -> VideoState | None:
        assert record.source_uri is not None
        resolved_uri = self._resolve_uri(record.source_uri)
        capture = self.capture_factory(resolved_uri)
        if not capture.isOpened():
            capture.release()
            self._open_failures += 1
            self._last_error = f"Could not open video source {record.source_id}: {resolved_uri}"
            LOGGER.error(self._last_error)
            return None
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0.0:
            fps = self.target_fps
        return VideoState(
            source_id=record.source_id,
            source_uri=record.source_uri,
            capture=capture,
            fps=fps,
            stride=max(1, round(fps / self.target_fps)),
        )

    def _release(self, source_id: str) -> None:
        state = self._states.pop(source_id, None)
        if state is not None:
            state.capture.release()

    def _read(self, state: VideoState) -> tuple[bool, Any, float | None]:
        ok, frame = state.capture.read()
        if not ok and self.loop:
            state.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            state.loop_count += 1
            ok, frame = state.capture.read()
        if not ok:
            state.read_failures += 1
            state.last_error = "End of stream" if not self.loop else "Frame read failed after rewind"
            return False, None, None

        state.frame_index = max(
            state.frame_index + 1,
            int(state.capture.get(cv2.CAP_PROP_POS_FRAMES) or 1) - 1,
        )
        state.last_error = None
        source_time_ms = float(state.capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        for _ in range(state.stride - 1):
            if not state.capture.grab():
                break
        return True, frame, source_time_ms / 1000.0

    def process_once(self) -> dict[str, int]:
        with self._state_lock:
            return self._process_once_unlocked()

    def _process_once_unlocked(self) -> dict[str, int]:
        records = [
            record
            for record in self.registry.list()
            if record.enabled and self.is_video_source(record)
        ]
        active_ids = {record.source_id for record in records}
        for source_id in set(self._states) - active_ids:
            self._release(source_id)

        frames: list[Any] = []
        source_ids: list[str] = []
        frame_indexes: list[int] = []
        source_times: list[float | None] = []
        metadata: list[dict[str, Any]] = []

        for record in records:
            state = self._states.get(record.source_id)
            if state is not None and state.source_uri != record.source_uri:
                self._release(record.source_id)
                state = None
            if state is None:
                state = self._open(record)
                if state is None:
                    continue
                self._states[record.source_id] = state

            ok, frame, source_time = self._read(state)
            if not ok:
                if not self.loop:
                    self._release(record.source_id)
                continue
            frames.append(frame)
            source_ids.append(record.source_id)
            frame_indexes.append(state.frame_index)
            source_times.append(source_time)
            metadata.append(
                {
                    "source_uri": record.source_uri,
                    "video_loop_count": state.loop_count,
                }
            )
            state.submitted_frames += 1

        if not frames:
            return {"received_frames": 0, "accepted_sources": 0, "task_submissions": 0}

        self._round_sequence += 1
        summary = self.router.submit_round(
            frames=frames,
            source_ids=source_ids,
            round_sequence=self._round_sequence,
            frame_indexes=frame_indexes,
            source_times_seconds=source_times,
            metadata=metadata,
        )
        self._rounds_submitted += 1
        self._frames_submitted += len(frames)
        self._last_error = None
        return summary

    def _run(self) -> None:
        self._started.set()
        interval = 1.0 / self.target_fps
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.process_once()
            except Exception as exc:
                self._last_error = str(exc)
                LOGGER.exception("Video ingestion round failed")
            remaining = interval - (time.monotonic() - started)
            if remaining > 0:
                self._stop.wait(remaining)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        with self._state_lock:
            for source_id in list(self._states):
                self._release(source_id)

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            return self._status_unlocked()

    def _status_unlocked(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "running": self._thread is not None and self._thread.is_alive(),
            "target_fps": self.target_fps,
            "loop": self.loop,
            "rounds_submitted": self._rounds_submitted,
            "frames_submitted": self._frames_submitted,
            "open_failures": self._open_failures,
            "last_error": self._last_error,
            "sources": {
                source_id: {
                    "source_uri": state.source_uri,
                    "source_fps": state.fps,
                    "stride": state.stride,
                    "frame_index": state.frame_index,
                    "submitted_frames": state.submitted_frames,
                    "loop_count": state.loop_count,
                    "read_failures": state.read_failures,
                    "last_error": state.last_error,
                }
                for source_id, state in self._states.items()
            },
        }
