from __future__ import annotations

import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import cv2

from app.core.router import TaskRouter
from app.core.source_registry import (
    RTSP,
    STATIC_VIDEO,
    SOURCE_TYPES,
    SourceRecord,
    SourceRegistry,
    canonical_source_type,
)

LOGGER = logging.getLogger(__name__)
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


@dataclass(slots=True)
class VideoState:
    source_id: str
    source_uri: str
    display_uri: str
    is_live: bool
    capture: Any
    fps: float
    effective_fps: float
    stride: int
    frame_width: int
    frame_height: int
    loop: bool = True
    next_frame_due_monotonic: float = 0.0
    frame_index: int = -1
    submitted_frames: int = 0
    loop_count: int = 0
    read_failures: int = 0
    last_error: str | None = None
    source_frame_width: int = 0
    source_frame_height: int = 0


class VideoFileIngestor:
    """Reads local video files and RTSP cameras into synchronized AI rounds."""

    UNKNOWN_SOURCE_FPS = 30.0
    SCHEDULER_MAX_FPS = 240.0

    def __init__(
        self,
        *,
        registry: SourceRegistry,
        router: TaskRouter,
        project_root: Path,
        loop: bool = True,
        source_type_filter: str = RTSP,
        max_sources: int = 256,
        gpu_resize_enabled: bool = True,
        rtsp_transport: str = "tcp",
        rtsp_open_timeout_ms: int = 20000,
        rtsp_read_timeout_ms: int = 10000,
        rtsp_reconnect_seconds: float = 3.0,
        capture_factory: Callable[..., Any] = cv2.VideoCapture,
        read_workers: int = 1,
        on_source_started: Callable[[str], None] | None = None,
        on_source_eos: Callable[[str, bool], None] | None = None,
        on_source_failed: Callable[[str, str], None] | None = None,
    ) -> None:
        if source_type_filter not in SOURCE_TYPES:
            raise ValueError(f"source_type_filter must be one of {sorted(SOURCE_TYPES)}")
        self.source_type_filter = source_type_filter
        self.max_sources = max(1, int(max_sources))
        self.registry = registry
        self.router = router
        self.project_root = project_root
        self.gpu_resize_enabled = bool(gpu_resize_enabled)
        # Detect GPU resize capability at startup (OpenCV CUDA fallback)
        self._gpu_resize_available = VideoFileIngestor._detect_gpu_resize()
        self.loop = loop
        self.rtsp_transport = (
            rtsp_transport.strip().lower()
            if rtsp_transport.strip().lower() in {"tcp", "udp"}
            else "tcp"
        )
        self.rtsp_open_timeout_ms = max(1000, rtsp_open_timeout_ms)
        self.rtsp_read_timeout_ms = max(1000, rtsp_read_timeout_ms)
        self.rtsp_reconnect_seconds = max(0.5, rtsp_reconnect_seconds)
        self.capture_factory = capture_factory
        self.read_workers = max(1, int(read_workers))
        self.on_source_started = on_source_started
        self.on_source_eos = on_source_eos
        self.on_source_failed = on_source_failed
        self._started_sources: set[str] = set()
        self._read_executor = (
            ThreadPoolExecutor(
                max_workers=self.read_workers,
                thread_name_prefix="video-source-reader",
            )
            if self.read_workers > 1
            else None
        )
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            f"rtsp_transport;{self.rtsp_transport}",
        )
        self._states: dict[str, VideoState] = {}
        self._retry_after: dict[str, float] = {}
        self._thread: threading.Thread | None = None
        self._state_lock = threading.RLock()
        self._stop = threading.Event()
        self._started = threading.Event()
        self._round_sequence = 0
        self._rounds_submitted = 0
        self._frames_submitted = 0
        self._open_failures = 0
        self._reconnects = 0
        self._last_error: str | None = None

    @staticmethod
    def _detect_gpu_resize() -> bool:
        """Check whether OpenCV was built with CUDA for GPU-accelerated resize."""
        try:
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            return False

    @staticmethod
    def is_rtsp_uri(source_uri: str) -> bool:
        return source_uri.strip().lower().startswith(("rtsp://", "rtsps://"))

    @staticmethod
    def redact_uri(source_uri: str) -> str:
        if not VideoFileIngestor.is_rtsp_uri(source_uri):
            return source_uri
        parsed = urlsplit(source_uri)
        host = parsed.hostname or "camera"
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        credentials = "***:***@" if parsed.username is not None else ""
        return urlunsplit(
            (parsed.scheme, f"{credentials}{host}", parsed.path, "", "")
        )

    @staticmethod
    def is_video_source(record: SourceRecord) -> bool:
        if not record.source_uri:
            return False
        # Source_type field takes priority when explicitly set
        if record.source_type in SOURCE_TYPES:
            return True
        if VideoFileIngestor.is_rtsp_uri(record.source_uri):
            return True
        if record.metadata.get("kind") == "video_file":
            return True
        uri = record.source_uri.lower()
        return "://" not in uri and Path(uri).suffix.lower() in VIDEO_SUFFIXES

    def _resolve_uri(self, source_uri: str) -> str:
        if self.is_rtsp_uri(source_uri):
            return source_uri
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
        is_live = self.is_rtsp_uri(resolved_uri)
        display_uri = self.redact_uri(resolved_uri)
        try:
            if is_live:
                parameters = [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    self.rtsp_open_timeout_ms,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    self.rtsp_read_timeout_ms,
                ]
                try:
                    capture = self.capture_factory(
                        resolved_uri,
                        cv2.CAP_FFMPEG,
                        parameters,
                    )
                except TypeError:
                    capture = self.capture_factory(resolved_uri)
            else:
                capture = self.capture_factory(resolved_uri)
        except Exception as exc:
            self._open_failures += 1
            self._retry_after[record.source_uri] = (
                time.monotonic() + self.rtsp_reconnect_seconds
            )
            self._last_error = (
                f"Could not create reader for {record.source_uri} ({display_uri}): "
                f"{type(exc).__name__}"
            )
            LOGGER.error(self._last_error)
            if not is_live and self.on_source_failed is not None:
                self.on_source_failed(record.source_uri, self._last_error)
            return None
        if not capture.isOpened():
            capture.release()
            self._open_failures += 1
            self._retry_after[record.source_uri] = (
                time.monotonic() + self.rtsp_reconnect_seconds
            )
            self._last_error = (
                f"Could not open video source {record.source_uri}: {display_uri}"
            )
            LOGGER.error(self._last_error)
            if not is_live and self.on_source_failed is not None:
                self.on_source_failed(record.source_uri, self._last_error)
            return None
        self._retry_after.pop(record.source_uri, None)
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        effective_fps = self._effective_fps(record, source_fps=source_fps)
        stride = 1
        if not is_live and source_fps > 0.0 and effective_fps > 0.0 and effective_fps < source_fps:
            stride = max(1, round(source_fps / effective_fps))
        return VideoState(
            source_id=record.source_uri,
            source_uri=record.source_uri,
            display_uri=display_uri,
            is_live=is_live,
            capture=capture,
            fps=source_fps,
            effective_fps=effective_fps,
            stride=stride,
            frame_width=record.frame_width,
            frame_height=record.frame_height,
            loop=record.loop,
        )

    def _effective_fps(
        self,
        record: SourceRecord,
        *,
        source_fps: float,
    ) -> float:
        if record.fps is not None and record.fps > 0:
            return float(record.fps)
        if source_fps > 0.0:
            return float(source_fps)
        return self.UNKNOWN_SOURCE_FPS

    def _release(self, source_id: str) -> None:
        state = self._states.pop(source_id, None)
        if state is not None:
            state.capture.release()
        self._started_sources.discard(source_id)

    def _read(self, state: VideoState) -> tuple[bool, Any, float | None]:
        ok, frame = state.capture.read()
        if not ok and state.loop and not state.is_live:
            if self.on_source_eos is not None:
                self.on_source_eos(state.source_id, True)
            state.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            state.loop_count += 1
            ok, frame = state.capture.read()
        if not ok:
            state.read_failures += 1
            if state.is_live:
                state.last_error = "RTSP frame read failed; reconnect scheduled"
            else:
                frame_count = float(state.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                position = float(state.capture.get(cv2.CAP_PROP_POS_FRAMES) or 0.0)
                truncated = not state.loop and frame_count > 0 and position < frame_count - 1
                state.last_error = (
                    "Video decoding stopped before the expected end of file"
                    if truncated
                    else "End of stream" if not state.loop else "Frame read failed after rewind"
                )
                if state.loop or truncated:
                    if self.on_source_failed is not None:
                        self.on_source_failed(state.source_id, state.last_error)
                elif self.on_source_eos is not None:
                    self.on_source_eos(state.source_id, False)
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

    def _read_due_state(
        self,
        item: tuple[SourceRecord, VideoState, float],
    ) -> tuple[SourceRecord, VideoState, Any | None, Any | None, float | None, int]:
        record, state, now_monotonic = item
        ok, source_frame, source_time = self._read(state)
        if not ok:
            return record, state, None, None, None, state.frame_index
        state.source_frame_width = int(source_frame.shape[1])
        state.source_frame_height = int(source_frame.shape[0])
        frame = source_frame
        if (
            source_frame.shape[1] != state.frame_width
            or source_frame.shape[0] != state.frame_height
        ):
            # Use GPU resize when available for better performance
            if self._gpu_resize_available:
                try:
                    gpu_frame = cv2.cuda.GpuMat()
                    gpu_frame.upload(source_frame)
                    frame = cv2.cuda.resize(
                        gpu_frame,
                        (state.frame_width, state.frame_height),
                        interpolation=cv2.INTER_AREA,
                    ).download()
                except Exception:
                    self._gpu_resize_available = False
                    LOGGER.warning("GPU resize failed; using CPU fallback", exc_info=True)
            if frame is source_frame:
                # CPU resize fallback
                frame = cv2.resize(
                    source_frame,
                    (state.frame_width, state.frame_height),
                    interpolation=cv2.INTER_AREA,
                )
        frame_index = state.frame_index
        state.submitted_frames += 1
        period = 1.0 / max(state.effective_fps, 0.1)
        previous_deadline = state.next_frame_due_monotonic
        if previous_deadline <= 0.0:
            state.next_frame_due_monotonic = now_monotonic + period
        else:
            periods = max(
                1,
                math.floor(
                    max(0.0, now_monotonic - previous_deadline) / period
                )
                + 1,
            )
            state.next_frame_due_monotonic = previous_deadline + periods * period
        return record, state, frame, source_frame, source_time, frame_index

    def process_once(self) -> dict[str, int]:
        with self._state_lock:
            return self._process_once_unlocked()

    def _process_once_unlocked(self) -> dict[str, int]:
        records = [
            record
            for record in self.registry.list()
            if record.enabled
            and self.is_video_source(record)
            and canonical_source_type(
                record.source_uri,
                record.source_type,
            ) == self.source_type_filter
        ]
        # Enforce max_sources cap: only open the first max_sources
        if len(records) > self.max_sources:
            LOGGER.warning(
                "source_type=%s sources=%d exceeds max_sources=%d; capping",
                self.source_type_filter,
                len(records),
                self.max_sources,
            )
            records = records[: self.max_sources]
        active_ids = {record.source_uri for record in records}
        for source_id in set(self._states) - active_ids:
            self._release(source_id)
        for source_id in set(self._retry_after) - active_ids:
            self._retry_after.pop(source_id, None)

        frames: list[Any] = []
        source_ids: list[str] = []
        frame_indexes: list[int] = []
        source_times: list[float | None] = []
        metadata: list[dict[str, Any]] = []
        due_states: list[tuple[SourceRecord, VideoState, float]] = []

        for record in records:
            now_monotonic = time.monotonic()
            state = self._states.get(record.source_uri)
            if state is not None and (
                state.source_uri != record.source_uri
                or state.frame_width != record.frame_width
                or state.frame_height != record.frame_height
            ):
                self._release(record.source_uri)
                self._retry_after.pop(record.source_uri, None)
                state = None
            if state is None:
                if self._retry_after.get(record.source_uri, 0.0) > time.monotonic():
                    continue
                state = self._open(record)
                if state is None:
                    continue
                self._states[record.source_uri] = state

            if state.next_frame_due_monotonic > now_monotonic:
                continue
            due_states.append((record, state, now_monotonic))

        if self._read_executor is None:
            read_results = map(self._read_due_state, due_states)
        else:
            read_results = self._read_executor.map(self._read_due_state, due_states)

        for (
            record,
            state,
            frame,
            source_frame,
            source_time,
            frame_index,
        ) in read_results:
            if frame is None or source_frame is None:
                if state.is_live:
                    self._reconnects += 1
                    self._retry_after[record.source_uri] = (
                        time.monotonic() + self.rtsp_reconnect_seconds
                    )
                    self._release(record.source_uri)
                elif not state.loop:
                    self._retry_after[record.source_uri] = float("inf")
                    self._release(record.source_uri)
                continue
            frames.append(frame)
            source_ids.append(record.source_uri)
            frame_indexes.append(frame_index)
            source_times.append(source_time)
            metadata.append(
                {
                    "source_uri": state.display_uri,
                    "source_type": "rtsp" if state.is_live else "video_file",
                    "frame_width": state.frame_width,
                    "frame_height": state.frame_height,
                    "source_frame_width": state.source_frame_width,
                    "source_frame_height": state.source_frame_height,
                    "source_frame": source_frame,
                    "video_loop_count": state.loop_count,
                }
            )

        if not frames:
            return {"received_frames": 0, "accepted_sources": 0, "task_submissions": 0}

        if self.on_source_started is not None:
            for source_id in source_ids:
                if source_id not in self._started_sources:
                    self._started_sources.add(source_id)
                    self.on_source_started(source_id)
        self._round_sequence += 1
        try:
            summary = self.router.submit_round(
                frames=frames,
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
        self._rounds_submitted += 1
        self._frames_submitted += len(frames)
        self._last_error = None
        return summary

    def _run(self) -> None:
        self._started.set()
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.process_once()
            except Exception as exc:
                self._last_error = str(exc)
                LOGGER.exception("Video ingestion round failed")
            remaining = (
                1.0 / self.SCHEDULER_MAX_FPS
            ) - (time.monotonic() - started)
            if remaining > 0:
                self._stop.wait(remaining)

    def restart_source(self, source_id: str) -> bool:
        """Schedule one source for teardown and reopen on the next polling cycle."""
        with self._state_lock:
            if source_id in self._states:
                self._release(source_id)
            self._retry_after.pop(source_id, None)
        return True

    def release_source(self, source_id: str) -> None:
        with self._state_lock:
            self._release(source_id)
            self._retry_after.pop(source_id, None)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._read_executor is not None:
            self._read_executor.shutdown(wait=True, cancel_futures=True)
        with self._state_lock:
            for source_id in list(self._states):
                self._release(source_id)

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            return self._status_unlocked()

    def _configured_fps(self, source_id: str) -> float | None:
        record = self.registry.get(source_id)
        return record.fps if record is not None else None

    def _status_unlocked(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "backend": "opencv",
            "source_type_filter": self.source_type_filter,
            "max_sources": self.max_sources,
            "running": self._thread is not None and self._thread.is_alive(),
            "fps_control": "sources.fps",
            "read_workers": self.read_workers,
            "loop": self.loop,
            "rounds_submitted": self._rounds_submitted,
            "frames_submitted": self._frames_submitted,
            "open_failures": self._open_failures,
            "reconnects": self._reconnects,
            "last_error": self._last_error,
            "sources": {
                source_id: {
                    "source_uri": state.display_uri,
                    "source_type": "rtsp" if state.is_live else "video_file",
                    "source_fps": state.fps,
                    "native_fps": state.fps or None,
                    "configured_fps": self._configured_fps(source_id),
                    "fps_mode": (
                        "override"
                        if self._configured_fps(source_id) is not None
                        else "native"
                    ),
                    "delivery_target_fps": self._configured_fps(source_id),
                    "effective_fps": state.effective_fps,
                    "stride": state.stride,
                    "frame_width": state.frame_width,
                    "frame_height": state.frame_height,
                    "source_frame_width": state.source_frame_width,
                    "source_frame_height": state.source_frame_height,
                    "frame_index": state.frame_index,
                    "decoded_samples": state.submitted_frames,
                    "received_frames": state.submitted_frames,
                    "submitted_frames": state.submitted_frames,
                    "loop_count": state.loop_count,
                    "read_failures": state.read_failures,
                    "last_error": state.last_error,
                }
                for source_id, state in self._states.items()
            },
        }


class StaticVideoFileIngestor(VideoFileIngestor):
    """Ingestor for static (pre-recorded) video files.

    Source pacing comes only from ``sources.fps``; NULL preserves the file's
    native FPS.
    """

    def __init__(
        self,
        *,
        registry: SourceRegistry,
        router: TaskRouter,
        project_root: Path,
        loop: bool = False,
        max_sources: int = 256,
        source_type_filter: str = STATIC_VIDEO,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            registry=registry,
            router=router,
            project_root=project_root,
            loop=loop,
            max_sources=max_sources,
            source_type_filter=source_type_filter,
            read_workers=min(32, max_sources),
            **kwargs,
        )
