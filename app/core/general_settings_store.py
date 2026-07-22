from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.database import Database, Row, ensure_database
from app.time_utils import utc_now_text


_GENERAL_COLUMNS = (
    "id, enable_processing, process_fire, process_plate, "
    "counts_for_attendance, margin_level, draw_box, draw_face, "
    "draw_skeleton, draw_zones, face_rec_score, face_det_score, "
    "human_det_score, confirmation_threshold, created_by, updated_by, "
    "created_at_utc, updated_at_utc"
)

_DEFAULTS: dict[str, Any] = {
    "enable_processing": True,
    "process_fire": False,
    "process_plate": False,
    "counts_for_attendance": True,
    "margin_level": 1.0,
    "draw_box": True,
    "draw_face": True,
    "draw_skeleton": False,
    "draw_zones": True,
    "face_rec_score": 0.4,
    "face_det_score": 0.4,
    "human_det_score": 0.4,
    "confirmation_threshold": 0.6,
}


@dataclass(frozen=True, slots=True)
class GeneralSettingsRecord:
    id: int
    enable_processing: bool
    process_fire: bool
    process_plate: bool
    counts_for_attendance: bool
    margin_level: float
    draw_box: bool
    draw_face: bool
    draw_skeleton: bool
    draw_zones: bool
    face_rec_score: float
    face_det_score: float
    human_det_score: float
    confirmation_threshold: float
    created_by: int | None
    updated_by: int | None
    created_at_utc: str
    updated_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "enable_processing": self.enable_processing,
            "process_fire": self.process_fire,
            "process_plate": self.process_plate,
            "counts_for_attendance": self.counts_for_attendance,
            "margin_level": self.margin_level,
            "draw_box": self.draw_box,
            "draw_face": self.draw_face,
            "draw_skeleton": self.draw_skeleton,
            "draw_zones": self.draw_zones,
            "face_rec_score": self.face_rec_score,
            "face_det_score": self.face_det_score,
            "human_det_score": self.human_det_score,
            "confirmation_threshold": self.confirmation_threshold,
            "created_at": self.created_at_utc,
            "updated_at": self.updated_at_utc,
            "created_by": self.created_by,
            "updated_by": self.updated_by,
        }


class GeneralSettingsStore:
    def __init__(self, database: Database | str) -> None:
        self.database = ensure_database(database)
        self._lock = threading.Lock()
        self._ensure_singleton()

    def _connection(self):
        return self.database.connection()

    @staticmethod
    def _row_to_record(row: Row) -> GeneralSettingsRecord:
        return GeneralSettingsRecord(
            id=int(row["id"]),
            enable_processing=bool(row["enable_processing"]),
            process_fire=bool(row["process_fire"]),
            process_plate=bool(row["process_plate"]),
            counts_for_attendance=bool(row["counts_for_attendance"]),
            margin_level=float(row["margin_level"]),
            draw_box=bool(row["draw_box"]),
            draw_face=bool(row["draw_face"]),
            draw_skeleton=bool(row["draw_skeleton"]),
            draw_zones=bool(row["draw_zones"]),
            face_rec_score=float(row["face_rec_score"]),
            face_det_score=float(row["face_det_score"]),
            human_det_score=float(row["human_det_score"]),
            confirmation_threshold=float(row["confirmation_threshold"]),
            created_by=row["created_by"],
            updated_by=row["updated_by"],
            created_at_utc=str(row["created_at_utc"]),
            updated_at_utc=str(row["updated_at_utc"]),
        )

    def _ensure_singleton(self) -> None:
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                "SELECT COUNT(*) FROM general_settings"
            ).fetchone()[0]
            if existing == 0:
                now = utc_now_text()
                conn.execute(
                    "INSERT INTO general_settings (id, enable_processing, process_fire, "
                    "process_plate, counts_for_attendance, margin_level, draw_box, "
                    "draw_face, draw_skeleton, draw_zones, face_rec_score, face_det_score, "
                    "human_det_score, confirmation_threshold, created_at_utc, updated_at_utc) "
                    "VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        1 if _DEFAULTS["enable_processing"] else 0,
                        1 if _DEFAULTS["process_fire"] else 0,
                        1 if _DEFAULTS["process_plate"] else 0,
                        1 if _DEFAULTS["counts_for_attendance"] else 0,
                        _DEFAULTS["margin_level"],
                        1 if _DEFAULTS["draw_box"] else 0,
                        1 if _DEFAULTS["draw_face"] else 0,
                        1 if _DEFAULTS["draw_skeleton"] else 0,
                        1 if _DEFAULTS["draw_zones"] else 0,
                        _DEFAULTS["face_rec_score"],
                        _DEFAULTS["face_det_score"],
                        _DEFAULTS["human_det_score"],
                        _DEFAULTS["confirmation_threshold"],
                        now,
                        now,
                    ),
                )
                conn.commit()

    def get(self) -> GeneralSettingsRecord:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                f"SELECT {_GENERAL_COLUMNS} FROM general_settings WHERE id = 1"
            ).fetchone()
            if row is None:
                self._ensure_singleton()
                row = conn.execute(
                    f"SELECT {_GENERAL_COLUMNS} FROM general_settings WHERE id = 1"
                ).fetchone()
                if row is None:
                    raise RuntimeError("Failed to create general settings singleton")
            return self._row_to_record(row)

    def update(self, changes: dict[str, Any], updated_by: int | None = None) -> GeneralSettingsRecord:
        with self._lock, self._connection() as conn:
            current = conn.execute(
                f"SELECT {_GENERAL_COLUMNS} FROM general_settings WHERE id = 1"
            ).fetchone()
            if current is None:
                raise RuntimeError("General settings singleton not found")

            now = utc_now_text()
            sets: list[str] = []
            params: list[Any] = []

            bool_fields = {
                "enable_processing", "process_fire", "process_plate",
                "counts_for_attendance", "draw_box", "draw_face",
                "draw_skeleton", "draw_zones",
            }
            float_fields = {
                "margin_level", "face_rec_score", "face_det_score",
                "human_det_score", "confirmation_threshold",
            }

            for key, value in changes.items():
                if key in bool_fields:
                    sets.append(f"{key} = ?")
                    params.append(1 if value else 0)
                elif key in float_fields:
                    sets.append(f"{key} = ?")
                    params.append(float(value))
                elif key in ("created_by", "updated_by"):
                    continue

            sets.append("updated_at_utc = ?")
            params.append(now)

            if updated_by is not None:
                sets.append("updated_by = ?")
                params.append(updated_by)

            params.append(1)
            conn.execute(
                f"UPDATE general_settings SET {', '.join(sets)} WHERE id = ?",
                params,
            )
            conn.commit()

            updated = conn.execute(
                f"SELECT {_GENERAL_COLUMNS} FROM general_settings WHERE id = 1"
            ).fetchone()
            if updated is None:
                raise RuntimeError("Failed to reload updated general settings")
            return self._row_to_record(updated)

    def reset(self, updated_by: int | None = None) -> GeneralSettingsRecord:
        with self._lock, self._connection() as conn:
            current = conn.execute(
                "SELECT created_at_utc, created_by FROM general_settings WHERE id = 1"
            ).fetchone()
            created_at = str(current["created_at_utc"]) if current else utc_now_text()
            original_created_by = current["created_by"] if current else None

            now = utc_now_text()
            conn.execute(
                "UPDATE general_settings SET "
                "enable_processing = ?, process_fire = ?, process_plate = ?, "
                "counts_for_attendance = ?, margin_level = ?, draw_box = ?, "
                "draw_face = ?, draw_skeleton = ?, draw_zones = ?, "
                "face_rec_score = ?, face_det_score = ?, human_det_score = ?, "
                "confirmation_threshold = ?, "
                "updated_at_utc = ?, updated_by = ? "
                "WHERE id = 1",
                (
                    1 if _DEFAULTS["enable_processing"] else 0,
                    1 if _DEFAULTS["process_fire"] else 0,
                    1 if _DEFAULTS["process_plate"] else 0,
                    1 if _DEFAULTS["counts_for_attendance"] else 0,
                    _DEFAULTS["margin_level"],
                    1 if _DEFAULTS["draw_box"] else 0,
                    1 if _DEFAULTS["draw_face"] else 0,
                    1 if _DEFAULTS["draw_skeleton"] else 0,
                    1 if _DEFAULTS["draw_zones"] else 0,
                    _DEFAULTS["face_rec_score"],
                    _DEFAULTS["face_det_score"],
                    _DEFAULTS["human_det_score"],
                    _DEFAULTS["confirmation_threshold"],
                    now,
                    updated_by,
                ),
            )
            conn.commit()

            updated = conn.execute(
                f"SELECT {_GENERAL_COLUMNS} FROM general_settings WHERE id = 1"
            ).fetchone()
            if updated is None:
                raise RuntimeError("Failed to reload reset general settings")
            return self._row_to_record(updated)
