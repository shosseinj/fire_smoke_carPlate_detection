from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any

from app.core.operational_settings import OperationalSettings
from app.database import Database, Row, ensure_database
from app.time_utils import utc_now_text

_GENERAL_COLUMNS = (
    "id, enable_processing, process_fire, process_plate, counts_for_attendance, "
    "margin_level, draw_box, draw_face, draw_skeleton, draw_zones, face_rec_score, "
    "face_det_score, human_det_score, confirmation_threshold, operational_json, "
    "created_by, updated_by, created_at_utc, updated_at_utc"
)

_DEFAULTS: dict[str, Any] = {
    "enable_processing": True, "process_fire": False, "process_plate": False,
    "counts_for_attendance": True, "margin_level": 1.0, "draw_box": True,
    "draw_face": True, "draw_skeleton": False, "draw_zones": True,
    "face_rec_score": 0.4, "face_det_score": 0.4, "human_det_score": 0.4,
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
    operational: OperationalSettings
    created_by: int | None
    updated_by: int | None
    created_at_utc: str
    updated_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "enable_processing": self.enable_processing,
            "process_fire": self.process_fire, "process_plate": self.process_plate,
            "counts_for_attendance": self.counts_for_attendance,
            "margin_level": self.margin_level, "draw_box": self.draw_box,
            "draw_face": self.draw_face, "draw_skeleton": self.draw_skeleton,
            "draw_zones": self.draw_zones, "face_rec_score": self.face_rec_score,
            "face_det_score": self.face_det_score, "human_det_score": self.human_det_score,
            "confirmation_threshold": self.confirmation_threshold,
            "operational": self.operational.to_dict(),
            "created_at": self.created_at_utc, "updated_at": self.updated_at_utc,
            "created_by": self.created_by, "updated_by": self.updated_by,
        }


class GeneralSettingsStore:
    def __init__(self, database: Database | str, default_operational: OperationalSettings | None = None) -> None:
        self.database = ensure_database(database)
        self._lock = threading.RLock()
        self._default_operational = default_operational or OperationalSettings()
        self._ensure_singleton()

    def _connection(self):
        return self.database.connection()

    def _row_to_record(self, row: Row) -> GeneralSettingsRecord:
        raw = row["operational_json"] or "{}"
        try:
            values = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            values = {}
        operational = self._default_operational.updated(values)
        return GeneralSettingsRecord(
            id=int(row["id"]), enable_processing=bool(row["enable_processing"]),
            process_fire=bool(row["process_fire"]), process_plate=bool(row["process_plate"]),
            counts_for_attendance=bool(row["counts_for_attendance"]), margin_level=float(row["margin_level"]),
            draw_box=bool(row["draw_box"]), draw_face=bool(row["draw_face"]),
            draw_skeleton=bool(row["draw_skeleton"]), draw_zones=bool(row["draw_zones"]),
            face_rec_score=float(row["face_rec_score"]), face_det_score=float(row["face_det_score"]),
            human_det_score=float(row["human_det_score"]), confirmation_threshold=float(row["confirmation_threshold"]),
            operational=operational, created_by=row["created_by"], updated_by=row["updated_by"],
            created_at_utc=str(row["created_at_utc"]), updated_at_utc=str(row["updated_at_utc"]),
        )

    def _ensure_singleton(self) -> None:
        with self._lock, self._connection() as conn:
            row = conn.execute("SELECT id, operational_json FROM general_settings WHERE id = 1").fetchone()
            if row is None:
                now = utc_now_text()
                conn.execute(
                    "INSERT INTO general_settings (id, enable_processing, process_fire, process_plate, counts_for_attendance, margin_level, draw_box, draw_face, draw_skeleton, draw_zones, face_rec_score, face_det_score, human_det_score, confirmation_threshold, operational_json, created_at_utc, updated_at_utc) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (1, 0, 0, 1, 1.0, 1, 1, 0, 1, 0.4, 0.4, 0.4, 0.6, json.dumps(self._default_operational.to_dict(), sort_keys=True), now, now),
                )
                conn.commit()
            elif not row["operational_json"] or row["operational_json"] == "{}":
                conn.execute("UPDATE general_settings SET operational_json = ?, updated_at_utc = ? WHERE id = 1", (json.dumps(self._default_operational.to_dict(), sort_keys=True), utc_now_text()))
                conn.commit()

    def get(self) -> GeneralSettingsRecord:
        with self._lock, self._connection() as conn:
            row = conn.execute(f"SELECT {_GENERAL_COLUMNS} FROM general_settings WHERE id = 1").fetchone()
            if row is None:
                raise RuntimeError("General settings singleton not found")
            return self._row_to_record(row)

    def update(self, changes: dict[str, Any], updated_by: int | None = None) -> GeneralSettingsRecord:
        with self._lock, self._connection() as conn:
            current = self.get()
            sets: list[str] = []
            params: list[Any] = []
            bool_fields = {"enable_processing", "process_fire", "process_plate", "counts_for_attendance", "draw_box", "draw_face", "draw_skeleton", "draw_zones"}
            float_fields = {"margin_level", "face_rec_score", "face_det_score", "human_det_score", "confirmation_threshold"}
            operational_changes = dict(changes.pop("operational", {}) or {})
            for key, value in changes.items():
                if key in bool_fields:
                    sets.append(f"{key} = ?"); params.append(1 if value else 0)
                elif key in float_fields:
                    sets.append(f"{key} = ?"); params.append(float(value))
            if operational_changes:
                updated = current.operational.updated(operational_changes)
                sets.append("operational_json = ?"); params.append(json.dumps(updated.to_dict(), sort_keys=True))
            sets.append("updated_at_utc = ?"); params.append(utc_now_text())
            if updated_by is not None:
                sets.append("updated_by = ?"); params.append(updated_by)
            params.append(1)
            conn.execute(f"UPDATE general_settings SET {', '.join(sets)} WHERE id = ?", params)
            conn.commit()
            row = conn.execute(f"SELECT {_GENERAL_COLUMNS} FROM general_settings WHERE id = 1").fetchone()
            if row is None:
                raise RuntimeError("Failed to reload general settings")
            return self._row_to_record(row)

    def reset(self, updated_by: int | None = None) -> GeneralSettingsRecord:
        values = dict(_DEFAULTS)
        values["operational"] = self._default_operational.to_dict()
        return self.update(values, updated_by=updated_by)
