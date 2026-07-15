from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from app.core.types import FramePacket, TaskName, TaskResult


class BatchProcessor(ABC):
    task: TaskName

    @abstractmethod
    def process_batch(self, packets: Sequence[FramePacket]) -> list[TaskResult]:
        raise NotImplementedError

    def status(self) -> dict[str, Any]:
        return {"task": self.task.value, "ready": True}

    def close(self) -> None:
        return None
