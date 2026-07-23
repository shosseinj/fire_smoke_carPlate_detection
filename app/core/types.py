from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

import numpy as np


class TaskName(str, Enum):
    """Python 3.10-compatible string enum used by FastAPI and Pydantic."""

    PLATE_RECOGNITION = "plate_recognition"
    FIRE_SMOKE = "fire_smoke"
    FACE_RECOGNITION = "face_recognition"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class FramePacket:
    source_id: str
    frame: np.ndarray
    round_sequence: int
    frame_index: int
    captured_monotonic: float
    captured_at_utc: str
    source_time_seconds: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def source_frame(self) -> np.ndarray:
        """Native decoded frame retained for high-resolution evidence crops."""
        value = self.metadata.get("source_frame")
        if isinstance(value, np.ndarray) and value.ndim in (2, 3):
            return value
        return self.frame


@dataclass(slots=True)
class TaskResult:
    task: TaskName
    source_id: str
    round_sequence: int
    frame_index: int
    captured_at_utc: str
    processed_at_utc: str
    processing_ms: float
    frame_width: int | None = None
    frame_height: int | None = None
    source_time_seconds: float | None = None
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def success(
        cls,
        *,
        task: TaskName,
        packet: FramePacket,
        processing_ms: float,
        data: dict[str, Any],
    ) -> "TaskResult":
        return cls(
            task=task,
            source_id=packet.source_id,
            round_sequence=packet.round_sequence,
            frame_index=packet.frame_index,
            captured_at_utc=packet.captured_at_utc,
            processed_at_utc=datetime.now(timezone.utc).isoformat(),
            processing_ms=round(processing_ms, 3),
            frame_width=int(packet.frame.shape[1]),
            frame_height=int(packet.frame.shape[0]),
            source_time_seconds=packet.source_time_seconds,
            data=data,
        )

    @classmethod
    def failed(
        cls,
        *,
        task: TaskName,
        packet: FramePacket,
        processing_ms: float,
        error: Exception | str,
    ) -> "TaskResult":
        return cls(
            task=task,
            source_id=packet.source_id,
            round_sequence=packet.round_sequence,
            frame_index=packet.frame_index,
            captured_at_utc=packet.captured_at_utc,
            processed_at_utc=datetime.now(timezone.utc).isoformat(),
            processing_ms=round(processing_ms, 3),
            frame_width=int(packet.frame.shape[1]),
            frame_height=int(packet.frame.shape[0]),
            source_time_seconds=packet.source_time_seconds,
            error=str(error),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "source_id": self.source_id,
            "round_sequence": self.round_sequence,
            "frame_index": self.frame_index,
            "captured_at_utc": self.captured_at_utc,
            "processed_at_utc": self.processed_at_utc,
            "processing_ms": self.processing_ms,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "source_time_seconds": self.source_time_seconds,
            "data": self.data,
            "error": self.error,
        }
