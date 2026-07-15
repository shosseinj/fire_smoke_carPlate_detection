from __future__ import annotations

import time
from typing import Sequence

from app.core.types import FramePacket, TaskName, TaskResult
from app.processors.base import BatchProcessor


class MockProcessor(BatchProcessor):
    def __init__(self, task: TaskName, sleep_ms: float = 0.0) -> None:
        self.task = task
        self.sleep_ms = sleep_ms
        self.batch_sizes: list[int] = []

    def process_batch(self, packets: Sequence[FramePacket]) -> list[TaskResult]:
        if self.sleep_ms:
            time.sleep(self.sleep_ms / 1000.0)
        self.batch_sizes.append(len(packets))
        return [
            TaskResult.success(
                task=self.task,
                packet=packet,
                processing_ms=self.sleep_ms,
                data={"mock": True, "shape": list(packet.frame.shape)},
            )
            for packet in packets
        ]

    def status(self) -> dict:
        return {
            "task": self.task.value,
            "ready": True,
            "mock": True,
            "batches": len(self.batch_sizes),
            "batch_sizes": self.batch_sizes[-20:],
        }
