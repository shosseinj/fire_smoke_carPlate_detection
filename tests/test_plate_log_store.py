from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from app.core.plate_log_store import PlateLogStore
from app.core.types import FramePacket, TaskName, TaskResult
from app.database import Database, metadata


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


def test_plate_log_table_saves_detection_and_snapshot_url(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store = PlateLogStore(
        postgres_database,
        draw_info=False,
        save_plate_snapshot=False,
        media_root=tmp_path / "media",
    )

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
    assert rows[0]["time"] == "2026-07-15T08:00:01Z"
    assert rows[0]["plate"] == "23ن92917"
    assert rows[0]["snapshot_url"].startswith("/media/plate_snapshots/")
    assert rows[0]["video_url"].startswith("/media/plate_videos/")
    video = tmp_path / "media" / rows[0]["video_url"].removeprefix("/media/")
    assert video.is_file()

    assert set(metadata.tables["plate_logs"].c.keys()) == {
        "id", "camera", "time", "plate", "snapshot_url", "video_url", "details_json"
    }


def test_manual_plate_log_insert_and_filters(postgres_database: Database) -> None:
    store = PlateLogStore(
        postgres_database,
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


def test_serialize_plate_log_maps_legacy_detection_columns() -> None:
    row = PlateLogStore._serialize_plate_log(
        {
            "plate_full_number": None,
            "camera_id": None,
            "detection_time": None,
            "snapshot_path": None,
            "plate": "23ن92917",
            "camera": "camera-01",
            "time": "2026-07-15T08:00:01+00:00",
            "snapshot_url": "/media/plate_snapshots/plate.jpg",
            "is_verified": 0,
        }
    )

    assert row["plate_full_number"] == "23ن92917"
    assert row["camera_id"] == "camera-01"
    assert row["detection_time"] == "2026-07-15T08:00:01+00:00"
    assert row["snapshot_path"] == "/media/plate_snapshots/plate.jpg"
    assert row["is_verified"] is False
