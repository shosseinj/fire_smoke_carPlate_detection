from __future__ import annotations

from pathlib import Path

from app.core.face_quality_store import FaceQualityPolicy, FaceQualitySettingsStore


def test_face_quality_settings_are_validated_and_persistent(tmp_path: Path) -> None:
    path = tmp_path / "logs.sqlite3"
    store = FaceQualitySettingsStore(path, FaceQualityPolicy())

    updated = store.update(
        {
            "quality_threshold": 0.72,
            "max_abs_pitch": 68.0,
            "require_landmarks": True,
        }
    )

    assert updated.quality_threshold == 0.72
    assert updated.max_abs_pitch == 68.0
    reopened = FaceQualitySettingsStore(
        path,
        FaceQualityPolicy(quality_threshold=0.1, max_abs_pitch=10.0),
    )
    assert reopened.get() == updated
