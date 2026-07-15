from __future__ import annotations

import time

import numpy as np

from app.core.latest_buffer import LatestPerSourceBuffer
from app.core.types import FramePacket


def packet(source_id: str, frame_index: int) -> FramePacket:
    return FramePacket(
        source_id=source_id,
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
        round_sequence=frame_index,
        frame_index=frame_index,
        captured_monotonic=time.monotonic(),
        captured_at_utc="2026-01-01T00:00:00+00:00",
    )


def test_latest_frame_replaces_stale_frame_without_duplicate_queue_entry() -> None:
    buffer = LatestPerSourceBuffer()
    assert buffer.put(packet("camera-01", 1))
    assert buffer.put(packet("camera-01", 2))
    assert buffer.put(packet("camera-02", 1))

    batch = buffer.take_batch(8, 0)
    assert [(item.source_id, item.frame_index) for item in batch] == [
        ("camera-01", 2),
        ("camera-02", 1),
    ]
    assert buffer.stats().stale_replaced == 1
    buffer.close()
