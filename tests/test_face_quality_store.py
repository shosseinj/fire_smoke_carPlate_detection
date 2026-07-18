from __future__ import annotations

import sqlite3
from pathlib import Path

from app.core.face_quality_store import FaceQualityPolicy, FaceQualitySettingsStore


def test_face_quality_settings_are_validated_and_persistent(tmp_path: Path) -> None:
    path = tmp_path / "logs.sqlite3"
    store = FaceQualitySettingsStore(path, FaceQualityPolicy())

    updated = store.update(
        {
            "quality_threshold": 0.72,
            "min_face_width": 48,
            "min_face_height": 56,
            "max_abs_pitch": 68.0,
            "require_landmarks": True,
        }
    )

    assert updated.quality_threshold == 0.72
    assert updated.min_face_width == 48
    assert updated.min_face_height == 56
    assert updated.max_abs_pitch == 68.0
    reopened = FaceQualitySettingsStore(
        path,
        FaceQualityPolicy(quality_threshold=0.1, max_abs_pitch=10.0),
    )
    assert reopened.get() == updated


def test_legacy_single_face_size_is_migrated_to_width_and_height(
    tmp_path: Path,
) -> None:
    path = tmp_path / "logs.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE face_quality_settings (
                singleton INTEGER PRIMARY KEY,
                quality_threshold REAL NOT NULL,
                blur_threshold REAL NOT NULL,
                min_face_size INTEGER NOT NULL,
                min_eye_distance REAL NOT NULL,
                max_abs_yaw REAL NOT NULL,
                max_abs_pitch REAL NOT NULL,
                max_abs_roll REAL NOT NULL,
                require_landmarks INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO face_quality_settings VALUES (1, .55, 20, 33, 8, 45, 55, 35, 1)"
        )

    store = FaceQualitySettingsStore(path, FaceQualityPolicy())

    assert store.get().min_face_width == 33
    assert store.get().min_face_height == 33
    with sqlite3.connect(path) as connection:
        columns = [
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(face_quality_settings)"
            )
        ]
    assert "min_face_size" not in columns
