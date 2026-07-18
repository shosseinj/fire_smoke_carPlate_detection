from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.human_log_store import HumanLogStore
from app.core.types import FramePacket, TaskName, TaskResult


def packet(frame_index: int) -> FramePacket:
    return FramePacket(
        source_id="camera-01",
        frame=np.zeros((120, 160, 3), dtype=np.uint8),
        round_sequence=frame_index,
        frame_index=frame_index,
        captured_monotonic=float(frame_index),
        captured_at_utc=f"2026-07-18T00:00:0{frame_index}+00:00",
    )


def result(source: FramePacket, name: str, score: float) -> TaskResult:
    return TaskResult.success(
        task=TaskName.FACE_RECOGNITION,
        packet=source,
        processing_ms=1.0,
        data={
            "tracking_session_id": "session-a",
            "humans": [
                {
                    "track_id": 13,
                    "bbox": [10, 10, 100, 110],
                    "person": name,
                    "recognition_score": score,
                    "ref_img_id": "reference-1" if name != "Unknown" else None,
                }
            ],
        },
    )


def test_one_log_per_human_track_is_upgraded_after_recognition(tmp_path: Path) -> None:
    store = HumanLogStore(tmp_path / "logs.sqlite3", tmp_path / "media")
    try:
        first = packet(1)
        recognized = packet(2)
        store.observe_result(first, result(first, "Unknown", 0.0))
        store.observe_result(recognized, result(recognized, "Alice", 0.93))
        store.flush()

        rows = store.list(camera="camera-01", track_id=13)
        assert len(rows) == 1
        assert rows[0]["name"] == "Alice"
        assert rows[0]["first_seen"] == "2026-07-18T00:00:01+00:00"
        assert rows[0]["last_seen"] == "2026-07-18T00:00:02+00:00"
        assert rows[0]["recognition_score"] == 0.93
        assert rows[0]["snapshot_url"].startswith("/media/human_snapshots/")
        snapshot = tmp_path / "media" / "human_snapshots" / Path(
            rows[0]["snapshot_url"]
        ).name
        assert snapshot.is_file()
        assert store.status()["saved_snapshots"] == 2
    finally:
        store.close()
