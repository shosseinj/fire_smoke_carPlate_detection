from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np

from app.core.plate_log_store import PlateLogStore
from app.core.types import FramePacket, TaskName, TaskResult


def plate_result() -> TaskResult:
    return TaskResult(
        task=TaskName.PLATE_RECOGNITION,
        source_id="camera-01",
        round_sequence=1,
        frame_index=6908,
        captured_at_utc="2026-07-15T08:00:00+00:00",
        processed_at_utc="2026-07-15T08:00:01+00:00",
        processing_ms=12.0,
        data={
            "plate_count": 1,
            "plates": [{"plate": "23ن92917"}],
        },
    )


def test_plate_log_table_saves_detection_and_snapshot_url(tmp_path: Path) -> None:
    database_path = tmp_path / "plate_logs.sqlite3"
    store = PlateLogStore(database_path, draw_info=False, save_plate_snapshot=False)

    source_packet = FramePacket(
        source_id="camera-01",
        frame=np.zeros((24, 32, 3), dtype=np.uint8),
        round_sequence=1,
        frame_index=6908,
        captured_monotonic=time.monotonic(),
        captured_at_utc="2026-07-15T08:00:00+00:00",
    )
    assert store.insert_result(source_packet, plate_result()) == 1
    rows = store.list()
    assert rows[0]["camera"] == "camera-01"
    assert rows[0]["time"] == "2026-07-15T08:00:01+00:00"
    assert rows[0]["plate"] == "23ن92917"
    assert rows[0]["snapshot_url"].startswith("/media/plate_snapshots/")

    with sqlite3.connect(database_path) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(plate_logs)")]
    assert columns == ["camera", "time", "plate", "snapshot_url"]


def test_manual_plate_log_insert_and_filters(tmp_path: Path) -> None:
    store = PlateLogStore(
        tmp_path / "plate_logs.sqlite3",
        draw_info=False,
        save_plate_snapshot=False,
    )
    store.insert(
        camera="camera-02",
        time="2026-07-15T08:00:00+00:00",
        plate="A1",
        snapshot_url="/media/plate_snapshots/a1.jpg",
    )
    store.insert(
        camera="camera-03",
        time="2026-07-15T08:00:01+00:00",
        plate="B2",
        snapshot_url="/media/plate_snapshots/b2.jpg",
    )

    assert store.count() == 2
    assert store.list(camera="camera-03")[0]["plate"] == "B2"
    assert store.list(plate="A1")[0]["camera"] == "camera-02"
