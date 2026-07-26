from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from app.core.result_store import ResultStore
from app.core.source_registry import SourceRegistry
from app.core.types import FramePacket, TaskName
from app.core.worker import TaskWorker


class TaskRouter:
    """Routes one extraction round to independent task-specific batch workers."""

    def __init__(
        self,
        *,
        registry: SourceRegistry,
        workers: Mapping[TaskName, TaskWorker],
        result_store: ResultStore,
        play_only_callback: Callable[[FramePacket], None] | None = None,
        source_only_callback: Callable[[FramePacket], None] | None = None,
        bypass_workers_callback: Callable[[], bool] | None = None,
    ) -> None:
        self.registry = registry
        self.workers = dict(workers)
        self.result_store = result_store
        self.play_only_callback = play_only_callback
        self.source_only_callback = source_only_callback
        self.bypass_workers_callback = bypass_workers_callback
        self.rounds_received = 0
        self.frames_received = 0
        self.frames_disabled = 0
        self.frames_unregistered = 0
        self.task_submissions = 0
        self.task_submission_rejections = 0
        self.broadcast_frames = 0
        self.play_only_frames = 0
        self.worker_bypass_frames = 0
        self.last_round_sequence: int | None = None
        self.started = False

    def start(self) -> None:
        if self.started:
            return
        for worker in self.workers.values():
            worker.start()
        self.started = True

    def close(self) -> None:
        for worker in self.workers.values():
            worker.close()
        self.started = False

    def submit_round(
        self,
        *,
        frames: Sequence[np.ndarray],
        source_ids: Sequence[str],
        round_sequence: int,
        frame_indexes: Sequence[int] | None = None,
        source_times_seconds: Sequence[float | None] | None = None,
        metadata: Sequence[Mapping[str, Any] | None] | None = None,
        captured_monotonic: float | None = None,
        captured_at_utc: str | None = None,
    ) -> dict[str, int]:
        if len(frames) != len(source_ids):
            raise ValueError("frames and source_ids must have equal lengths")
        count = len(frames)
        if frame_indexes is not None and len(frame_indexes) != count:
            raise ValueError("frame_indexes length does not match frames")
        if source_times_seconds is not None and len(source_times_seconds) != count:
            raise ValueError("source_times_seconds length does not match frames")
        if metadata is not None and len(metadata) != count:
            raise ValueError("metadata length does not match frames")

        now_monotonic = captured_monotonic if captured_monotonic is not None else time.monotonic()
        now_utc = captured_at_utc or datetime.now(timezone.utc).isoformat()
        accepted_sources = 0
        task_submissions = 0

        for index, (source_id, frame) in enumerate(zip(source_ids, frames)):
            if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
                raise ValueError(f"Invalid frame at index {index}")
            source = self.registry.get(source_id)
            if source is None:
                self.frames_unregistered += 1
                continue
            if not source.enabled:
                self.frames_disabled += 1
                continue
            accepted_sources += 1
            packet_metadata = dict(metadata[index] or {}) if metadata is not None else {}
            packet_metadata["assigned_tasks"] = sorted(task.value for task in source.tasks)
            packet = FramePacket(
                source_id=source_id,
                frame=frame,
                round_sequence=round_sequence,
                frame_index=(frame_indexes[index] if frame_indexes is not None else round_sequence),
                captured_monotonic=now_monotonic,
                captured_at_utc=now_utc,
                source_time_seconds=(source_times_seconds[index] if source_times_seconds is not None else None),
                metadata=packet_metadata,
            )
            if self.source_only_callback is not None:
                self.source_only_callback(packet)
            if self.play_only_callback is not None:
                self.play_only_callback(packet)
                self.broadcast_frames += 1
            if (
                self.bypass_workers_callback is not None
                and self.bypass_workers_callback()
            ):
                self.worker_bypass_frames += 1
                continue
            if not source.tasks:
                self.play_only_frames += 1
            for task in source.tasks:
                worker = self.workers.get(task)
                if worker is None:
                    self.task_submission_rejections += 1
                    continue
                if worker.submit(packet):
                    task_submissions += 1
                    self.task_submissions += 1
                else:
                    self.task_submission_rejections += 1

        self.rounds_received += 1
        self.frames_received += count
        self.last_round_sequence = round_sequence
        return {
            "received_frames": count,
            "accepted_sources": accepted_sources,
            "task_submissions": task_submissions,
        }

    def status(self) -> dict[str, Any]:
        return {
            "started": self.started,
            "rounds_received": self.rounds_received,
            "frames_received": self.frames_received,
            "frames_disabled": self.frames_disabled,
            "frames_unregistered": self.frames_unregistered,
            "task_submissions": self.task_submissions,
            "task_submission_rejections": self.task_submission_rejections,
            "broadcast_frames": self.broadcast_frames,
            "play_only_frames": self.play_only_frames,
            "worker_bypass_frames": self.worker_bypass_frames,
            "last_round_sequence": self.last_round_sequence,
            "enabled_source_ids": self.registry.enabled_source_ids(),
            "workers": {task.value: worker.status() for task, worker in self.workers.items()},
        }
