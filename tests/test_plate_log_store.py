from __future__ import annotations

import sqlite3
from pathlib import Path

from app.core.plate_log_store import PlateLogStore
from app.core.types import TaskName, TaskResult


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


def test_plate_log_table_has_three_fields_and_saves_detection(tmp_path: Path) -> None:
    database_path = tmp_path / "plate_logs.sqlite3"
    store = PlateLogStore(database_path)

    assert store.insert_result(plate_result()) == 1
    rows = store.list()
    assert rows == [
        {
            "camera": "camera-01",
            "time": "2026-07-15T08:00:01+00:00",
            "plate": "23ن92917",
        }
    ]

    with sqlite3.connect(database_path) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(plate_logs)")]
    assert columns == ["camera", "time", "plate"]


def test_manual_plate_log_insert_and_filters(tmp_path: Path) -> None:
    store = PlateLogStore(tmp_path / "plate_logs.sqlite3")
    store.insert(camera="camera-02", time="2026-07-15T08:00:00+00:00", plate="A1")
    store.insert(camera="camera-03", time="2026-07-15T08:00:01+00:00", plate="B2")

    assert store.count() == 2
    assert store.list(camera="camera-03")[0]["plate"] == "B2"
    assert store.list(plate="A1")[0]["camera"] == "camera-02"
