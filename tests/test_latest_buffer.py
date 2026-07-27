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
    stats = buffer.stats()
    assert stats.stale_replaced == 1
    assert stats.accepted_by_source == {"camera-01": 2, "camera-02": 1}
    assert stats.stale_replaced_by_source == {"camera-01": 1}
    buffer.close()


def test_lossless_fifo_preserves_every_frame_in_order() -> None:
    buffer = LatestPerSourceBuffer(policy="lossless_fifo", capacity=3)
    packets = [packet("camera-01", frame_index) for frame_index in range(3)]

    assert all(buffer.put(item) for item in packets)
    result = buffer.take_batch(3, 0.0)

    assert [item.frame_index for item in result] == [0, 1, 2]
    stats = buffer.stats()
    assert stats.policy == "lossless_fifo"
    assert stats.queue_depth == 0
    assert stats.stale_replaced == 0
    buffer.close()


def test_clear_discards_pending_latest_frames_for_all_sources() -> None:
    buffer = LatestPerSourceBuffer()
    assert buffer.put(packet("camera-01", 1))
    assert buffer.put(packet("camera-02", 1))

    assert buffer.clear() == 2
    assert buffer.stats().pending_sources == 0
    buffer.close()
