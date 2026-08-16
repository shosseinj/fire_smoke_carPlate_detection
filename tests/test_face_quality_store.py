from __future__ import annotations

from app.core.face_quality_store import FaceQualityPolicy, FaceQualitySettingsStore
from app.database import Database, metadata
from app.processors.face_recognition import FaceMatch, SourceFaceTracker, TrackState


def test_face_quality_settings_are_validated_and_persistent(
    postgres_database: Database,
) -> None:
    store = FaceQualitySettingsStore(postgres_database, FaceQualityPolicy())

    updated = store.update(
        {
            "quality_threshold": 0.72,
            "min_face_width": 48,
            "min_face_height": 56,
        "max_abs_pitch": 68.0,
        "require_landmarks": True,
        "human_pose_min_keypoints": 6,
        "human_pose_keypoint_confidence": 0.35,
        "recognition_quality_weight": 0.8,
        }
    )

    assert updated.quality_threshold == 0.72
    assert updated.min_face_width == 48
    assert updated.min_face_height == 56
    assert updated.max_abs_pitch == 68.0
    assert updated.human_pose_min_keypoints == 6
    assert updated.human_pose_keypoint_confidence == 0.35
    assert updated.recognition_quality_weight == 0.8
    reopened = FaceQualitySettingsStore(
        postgres_database,
        FaceQualityPolicy(quality_threshold=0.1, max_abs_pitch=10.0),
    )
    assert reopened.get() == updated


def test_tracked_identity_keeps_best_recognition_score_and_face_quality() -> None:
    tracker = SourceFaceTracker.__new__(SourceFaceTracker)
    tracker.tracks = {
        1: TrackState(
            track_id=1,
            bbox=[0.0, 0.0, 10.0, 10.0],
            stable_person="Alice",
            stable_score=0.60,
            stable_ref_img_id="old",
        )
    }

    improved = tracker.observe(1, FaceMatch("Alice", 0.91, "new"))
    tracker.record_face_quality(1, 0.88)

    assert improved.score == 0.91
    assert improved.ref_img_id == "new"
    assert tracker.best_quality(1) == 0.88


def test_postgresql_schema_has_separate_face_dimensions() -> None:
    columns = metadata.tables["face_quality_settings"].c
    assert "min_face_width" in columns
    assert "min_face_height" in columns
    assert "min_face_size" not in columns

import pytest

pytestmark = pytest.mark.postgresql
