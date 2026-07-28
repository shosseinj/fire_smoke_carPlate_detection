from __future__ import annotations

from app.database import Connection, Database, IntegrityError, OperationalError, Row, ensure_database

import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class FaceQualityPolicy:
    quality_threshold: float = 0.55
    blur_threshold: float = 20.0
    min_face_width: int = 24
    min_face_height: int = 24
    min_eye_distance: float = 8.0
    max_abs_yaw: float = 45.0
    max_abs_pitch: float = 55.0
    max_abs_roll: float = 35.0
    require_landmarks: bool = True
    human_pose_enabled: bool = True
    human_pose_min_keypoints: int = 4
    human_pose_keypoint_confidence: float = 0.25
    recognition_quality_weight: float = 0.5

    def validated(self) -> "FaceQualityPolicy":
        if not 0.0 <= self.quality_threshold <= 1.0:
            raise ValueError("آستانه کیفیت باید بین 0 و 1 باشد")
        if self.blur_threshold < 0:
            raise ValueError("آستانه تاری باید غیرمنفی باشد")
        for name in ("min_face_width", "min_face_height"):
            if not 1 <= int(getattr(self, name)) <= 4096:
                raise ValueError(f"{name} must be between 1 and 4096")
        if self.min_eye_distance < 0:
            raise ValueError("حداقل فاصله چشم باید غیرمنفی باشد")
        for name in ("max_abs_yaw", "max_abs_pitch", "max_abs_roll"):
            if not 0 < float(getattr(self, name)) <= 90:
                raise ValueError(f"{name} must be greater than 0 and at most 90")
        if not 1 <= int(self.human_pose_min_keypoints) <= 17:
            raise ValueError("حداقل نقاط کلیدی وضعیت بدن باید بین 1 و 17 باشد")
        if not 0.0 <= float(self.human_pose_keypoint_confidence) <= 1.0:
            raise ValueError("اطمینان نقاط کلیدی وضعیت بدن باید بین 0 و 1 باشد")
        if not 0.0 <= float(self.recognition_quality_weight) <= 1.0:
            raise ValueError("وزن کیفیت تشخیص باید بین 0 و 1 باشد")
        return self


class FaceQualitySettingsStore:
    """Persistent singleton policy used to select recognition-quality faces."""

    def __init__(self, database: Database | str, default_policy: FaceQualityPolicy) -> None:
        self.database = ensure_database(database)
        self._lock = threading.RLock()
        self._create(default_policy.validated())

    def _connect(self) -> Connection:
        return self.database.connection()

    def _create(self, default_policy: FaceQualityPolicy) -> None:
        values = asdict(default_policy)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO face_quality_settings (
                    singleton, quality_threshold, blur_threshold,
                    min_face_width, min_face_height,
                    min_eye_distance, max_abs_yaw, max_abs_pitch, max_abs_roll,
                    require_landmarks, human_pose_enabled, human_pose_min_keypoints,
                    human_pose_keypoint_confidence, recognition_quality_weight
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(singleton) DO NOTHING
                """,
                (
                    values["quality_threshold"],
                    values["blur_threshold"],
                    values["min_face_width"],
                    values["min_face_height"],
                    values["min_eye_distance"],
                    values["max_abs_yaw"],
                    values["max_abs_pitch"],
                    values["max_abs_roll"],
                    int(values["require_landmarks"]),
                    int(values["human_pose_enabled"]),
                    values["human_pose_min_keypoints"],
                    values["human_pose_keypoint_confidence"],
                    values["recognition_quality_weight"],
                ),
            )

    def get(self) -> FaceQualityPolicy:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM face_quality_settings WHERE singleton = 1"
            ).fetchone()
        assert row is not None
        return FaceQualityPolicy(
            quality_threshold=float(row["quality_threshold"]),
            blur_threshold=float(row["blur_threshold"]),
            min_face_width=int(row["min_face_width"]),
            min_face_height=int(row["min_face_height"]),
            min_eye_distance=float(row["min_eye_distance"]),
            max_abs_yaw=float(row["max_abs_yaw"]),
            max_abs_pitch=float(row["max_abs_pitch"]),
            max_abs_roll=float(row["max_abs_roll"]),
            require_landmarks=bool(row["require_landmarks"]),
            human_pose_enabled=bool(row["human_pose_enabled"]),
            human_pose_min_keypoints=int(row["human_pose_min_keypoints"]),
            human_pose_keypoint_confidence=float(row["human_pose_keypoint_confidence"]),
            recognition_quality_weight=float(row["recognition_quality_weight"]),
        ).validated()

    def update(self, changes: dict[str, Any]) -> FaceQualityPolicy:
        current = self.get()
        unexpected = set(changes) - set(asdict(current))
        if unexpected:
            raise ValueError(f"تنظیمات کیفیت چهره پشتیبانی نشده: {sorted(unexpected)}")
        updated = replace(current, **changes).validated()
        values = asdict(updated)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE face_quality_settings SET
                    quality_threshold = ?, blur_threshold = ?,
                    min_face_width = ?, min_face_height = ?,
                    min_eye_distance = ?, max_abs_yaw = ?, max_abs_pitch = ?,
                    max_abs_roll = ?, require_landmarks = ?, human_pose_enabled = ?,
                    human_pose_min_keypoints = ?, human_pose_keypoint_confidence = ?,
                    recognition_quality_weight = ?
                WHERE singleton = 1
                """,
                (
                    values["quality_threshold"],
                    values["blur_threshold"],
                    values["min_face_width"],
                    values["min_face_height"],
                    values["min_eye_distance"],
                    values["max_abs_yaw"],
                    values["max_abs_pitch"],
                    values["max_abs_roll"],
                    int(values["require_landmarks"]),
                    int(values["human_pose_enabled"]),
                    values["human_pose_min_keypoints"],
                    values["human_pose_keypoint_confidence"],
                    values["recognition_quality_weight"],
                ),
            )
        return updated

    def as_dict(self) -> dict[str, Any]:
        return asdict(self.get())
