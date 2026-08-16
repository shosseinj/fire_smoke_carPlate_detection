from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from app.api.plate_logs import PlateLogCreate
from app.core.car_plate_store import CarPlateStore
from app.core.plate_log_store import PlateLogStore
from app.core.types import FramePacket, TaskName, TaskResult
from app.database import Database, metadata


def plate_result(source_id: str = "camera-01") -> TaskResult:
    return TaskResult(
        task=TaskName.PLATE_RECOGNITION,
        source_id=source_id,
        round_sequence=1,
        frame_index=6908,
        captured_at_utc="2026-07-15T08:00:00+00:00",
        processed_at_utc="2026-07-15T08:00:01+00:00",
        processing_ms=12.0,
        data={
            "plate_count": 1,
            "plate_ocr_results": [
                {
                    "raw_plate_text": "23 ن 92917",
                    "plate_number": "23ن92917",
                    "is_valid_plate": True,
                    "recognizer_confidence": 0.91,
                }
            ],
            "plates": [
                {
                    "plate": "23ن92917",
                    "recognizer_confidence": 0.91,
                    "detector_confidence": 0.88,
                }
            ],
        },
    )


def packet(source_id: str = "camera-01") -> FramePacket:
    return FramePacket(
        source_id=source_id,
        frame=np.zeros((24, 32, 3), dtype=np.uint8),
        round_sequence=1,
        frame_index=6908,
        captured_monotonic=time.monotonic(),
        captured_at_utc="2026-07-15T08:00:00+00:00",
    )


def test_camera_detection_uses_normalized_fields_and_capture_time(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    with postgres_database.connection() as connection:
        registered = connection.execute(
            """
            INSERT INTO car_plates (
                left_digits, plate_alphabet, right_digits, iran_code,
                usage_type, vehicle_type, owner_name, owner_phone
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING id
            """,
            ("23", "ن", "929", "17", "personal", "sedan", "Test", "+989121234567"),
        ).fetchone()
        connection.commit()
    store = PlateLogStore(
        postgres_database,
        draw_info=False,
        save_plate_snapshot=True,
        media_root=tmp_path / "media",
    )
    try:
        assert store.insert_result(packet(), plate_result()) == 1
        row = store.list_logs()[0]
        assert row["source_type"] == "camera"
        assert row["source_uri"] == "camera-01"
        assert row["static_video_id"] is None
        assert row["plate_id"] == int(registered["id"])
        assert row["is_registered"] is True
        assert row["plate_number"] == "23ن92917"
        assert row["raw_plate_text"] == "23 ن 92917"
        assert row["confidence"] == 0.91
        assert str(row["detection_time"]).startswith("2026-07-15 08:00:00")
        assert row["snapshot_key"].startswith("plate/snapshots/")
        assert row["snapshot_url"].startswith("/media/plate/snapshots/")
        assert row["video_key"].startswith("plate/videos/")
        assert (tmp_path / "media" / row["snapshot_key"]).is_file()
        assert (tmp_path / "media" / row["video_key"]).is_file()
    finally:
        store.close()

    assert set(metadata.tables["plate_logs"].c.keys()) == {
        "id", "source_type", "source_uri", "static_video_id",
        "created_by_user_id", "updated_by_user_id", "plate_id",
        "plate_number", "raw_plate_text", "confidence", "detection_time",
        "snapshot_key", "video_key", "notes", "created_at", "updated_at",
    }


def test_registering_plate_links_all_unassigned_historical_logs(
    postgres_database: Database,
) -> None:
    with postgres_database.connection() as connection:
        other_plate = connection.execute(
            """
            INSERT INTO car_plates (
                left_digits, plate_alphabet, right_digits, iran_code,
                usage_type, vehicle_type, owner_name, owner_phone
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING id
            """,
            ("99", "Ù†", "999", "99", "personal", "sedan", "Other", "+989121234568"),
        ).fetchone()
        connection.executemany(
            """
            INSERT INTO plate_logs (
                source_type, source_uri, plate_id, plate_number, detection_time
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                ("camera", "camera-01", None, "23Ù†92917", "2026-07-15T08:00:00+00:00"),
                ("camera", "camera-02", None, "23Ù†92917", "2026-07-15T09:00:00+00:00"),
                ("camera", "camera-03", None, "11Ø¨11111", "2026-07-15T10:00:00+00:00"),
                ("camera", "camera-04", int(other_plate["id"]), "23Ù†92917", "2026-07-15T11:00:00+00:00"),
            ],
        )
        connection.commit()

    registered = CarPlateStore(postgres_database).create(
        {
            "left_digits": "23",
            "plate_alphabet": "Ù†",
            "right_digits": "929",
            "iran_code": "17",
            "usage_type": "personal",
            "vehicle_type": "sedan",
            "owner_name": "Test",
            "owner_phone": "+989121234567",
        }
    )

    with postgres_database.connection() as connection:
        rows = connection.execute(
            "SELECT source_uri, plate_id FROM plate_logs ORDER BY source_uri"
        ).fetchall()

    assert [row["plate_id"] for row in rows[:2]] == [registered["id"], registered["id"]]
    assert rows[2]["plate_id"] is None
    assert rows[3]["plate_id"] == int(other_plate["id"])


def test_static_video_detection_uses_static_video_id(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    source_uri = "/media/static/plate.mp4"
    with postgres_database.connection() as connection:
        row = connection.execute(
            "INSERT INTO static_videos (name, source_uri) VALUES (?, ?) RETURNING id",
            ("plate video", source_uri),
        ).fetchone()
    store = PlateLogStore(
        postgres_database,
        draw_info=False,
        save_plate_snapshot=False,
        media_root=tmp_path / "media",
    )
    try:
        assert store.insert_result(packet(source_uri), plate_result(source_uri)) == 1
        saved = store.list_logs()[0]
        assert saved["source_type"] == "static_video"
        assert saved["source_uri"] is None
        assert saved["static_video_id"] == int(row["id"])
        assert saved["snapshot_key"] is None
    finally:
        store.close()


def test_manual_log_and_update_record_user_ids(postgres_database: Database) -> None:
    store = PlateLogStore(postgres_database, draw_info=False, save_plate_snapshot=False)
    try:
        created = store.create_log(
            {
                "source_type": "manual",
                "source_uri": None,
                "static_video_id": None,
                "created_by_user_id": 7,
                "plate_number": "23ن92917",
                "detection_time": "2026-07-15T08:00:00+00:00",
            }
        )
        assert created["created_by_user_id"] == 7
        assert created["source_uri"] is None
        updated = store.update_log(
            created["id"], {"notes": "reviewed", "updated_by_user_id": 9}
        )
        assert updated is not None
        assert updated["updated_by_user_id"] == 9
        assert updated["notes"] == "reviewed"
    finally:
        store.close()


def test_manual_api_schema_rejects_automated_source_type() -> None:
    try:
        PlateLogCreate(
            source_type="camera",
            plate_number="23ن92917",
            detection_time="2026-07-15T08:00:00+00:00",
        )
    except ValueError as exc:
        assert "manual" in str(exc).lower() or "دستی" in str(exc)
    else:
        raise AssertionError("camera source type must not be accepted by manual API")


def test_invalid_ocr_text_is_saved_without_plate_number(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    result = plate_result()
    result.data["plates"] = []
    result.data["plate_count"] = 0
    result.data["plate_ocr_results"] = [
        {
            "raw_plate_text": " 12 ب 3456 ",
            "plate_number": None,
            "is_valid_plate": False,
            "recognizer_confidence": 0.84,
        }
    ]
    store = PlateLogStore(
        postgres_database,
        draw_info=False,
        save_plate_snapshot=False,
        media_root=tmp_path / "media",
    )
    try:
        assert store.insert_result(packet(), result) == 1
        saved = store.list_logs()[0]
        assert saved["raw_plate_text"] == " 12 ب 3456 "
        assert saved["plate_number"] is None
        assert saved["plate_id"] is None
        assert saved["is_registered"] is False
    finally:
        store.close()

import pytest

pytestmark = pytest.mark.postgresql
