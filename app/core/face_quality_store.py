from __future__ import annotations

import sqlite3
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class FaceQualityPolicy:
    quality_threshold: float = 0.55
    blur_threshold: float = 20.0
    min_face_size: int = 24
    min_eye_distance: float = 8.0
    max_abs_yaw: float = 45.0
    max_abs_pitch: float = 55.0
    max_abs_roll: float = 35.0
    require_landmarks: bool = True

    def validated(self) -> "FaceQualityPolicy":
        if not 0.0 <= self.quality_threshold <= 1.0:
            raise ValueError("quality_threshold must be between 0 and 1")
        if self.blur_threshold < 0:
            raise ValueError("blur_threshold must be non-negative")
        if not 1 <= self.min_face_size <= 4096:
            raise ValueError("min_face_size must be between 1 and 4096")
        if self.min_eye_distance < 0:
            raise ValueError("min_eye_distance must be non-negative")
        for name in ("max_abs_yaw", "max_abs_pitch", "max_abs_roll"):
            if not 0 < float(getattr(self, name)) <= 90:
                raise ValueError(f"{name} must be greater than 0 and at most 90")
        return self


class FaceQualitySettingsStore:
    """Persistent singleton policy used to select recognition-quality faces."""

    def __init__(self, database_path: Path, default_policy: FaceQualityPolicy) -> None:
        self.database_path = database_path.resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._create(default_policy.validated())

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _create(self, default_policy: FaceQualityPolicy) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS face_quality_settings (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
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
            values = asdict(default_policy)
            connection.execute(
                """
                INSERT OR IGNORE INTO face_quality_settings (
                    singleton, quality_threshold, blur_threshold, min_face_size,
                    min_eye_distance, max_abs_yaw, max_abs_pitch, max_abs_roll,
                    require_landmarks
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    values["quality_threshold"],
                    values["blur_threshold"],
                    values["min_face_size"],
                    values["min_eye_distance"],
                    values["max_abs_yaw"],
                    values["max_abs_pitch"],
                    values["max_abs_roll"],
                    int(values["require_landmarks"]),
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
            min_face_size=int(row["min_face_size"]),
            min_eye_distance=float(row["min_eye_distance"]),
            max_abs_yaw=float(row["max_abs_yaw"]),
            max_abs_pitch=float(row["max_abs_pitch"]),
            max_abs_roll=float(row["max_abs_roll"]),
            require_landmarks=bool(row["require_landmarks"]),
        ).validated()

    def update(self, changes: dict[str, Any]) -> FaceQualityPolicy:
        current = self.get()
        unexpected = set(changes) - set(asdict(current))
        if unexpected:
            raise ValueError(f"Unsupported face quality settings: {sorted(unexpected)}")
        updated = replace(current, **changes).validated()
        values = asdict(updated)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE face_quality_settings SET
                    quality_threshold = ?, blur_threshold = ?, min_face_size = ?,
                    min_eye_distance = ?, max_abs_yaw = ?, max_abs_pitch = ?,
                    max_abs_roll = ?, require_landmarks = ?
                WHERE singleton = 1
                """,
                (
                    values["quality_threshold"],
                    values["blur_threshold"],
                    values["min_face_size"],
                    values["min_eye_distance"],
                    values["max_abs_yaw"],
                    values["max_abs_pitch"],
                    values["max_abs_roll"],
                    int(values["require_landmarks"]),
                ),
            )
        return updated

    def as_dict(self) -> dict[str, Any]:
        return asdict(self.get())
