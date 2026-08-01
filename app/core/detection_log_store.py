from __future__ import annotations

from app.database import Connection, Database, IntegrityError, OperationalError, Row, ensure_database
from app.time_utils import utc_now_text
from app.core.detection_media import VALID_MEDIA_STATUSES

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DetectionLogRecord:
    id: int
    source_system: str
    source_event_key: str | None
    source_human_log_id: int | None
    personnel_id: int | None
    person: str
    confidence: float
    detection_time: str
    ref_img_id: str | None
    room_id: int | None
    camera_id: str | None
    access_granted: bool
    counts_for_attendance: bool
    log_type: str
    import_source_parts: str | None
    face_image: str | None
    face_thumbnail: str | None
    body_image: str | None
    snapshot_image: str | None
    video: str | None
    face_video_or_unknown_faces: str | None
    video_status: str
    face_video_status: str
    media_finalized_at: str | None
    created_by: int | None
    updated_by: int | None
    created_at_utc: str
    updated_at_utc: str


def _now() -> str:
    return utc_now_text()


class DetectionLogStore:
    """PostgreSQL-backed store for detection_logs."""

    def __init__(self, database: Database | str) -> None:
        self.database = ensure_database(database)
        self._lock = threading.RLock()
        self._listeners: set[Callable[[str, DetectionLogRecord], None]] = set()
        self._init_db()

    def add_listener(
        self, listener: Callable[[str, DetectionLogRecord], None]
    ) -> None:
        with self._lock:
            self._listeners.add(listener)

    def remove_listener(
        self, listener: Callable[[str, DetectionLogRecord], None]
    ) -> None:
        with self._lock:
            self._listeners.discard(listener)

    def _notify(self, action: str, record: DetectionLogRecord) -> None:
        with self._lock:
            listeners = tuple(self._listeners)
        for listener in listeners:
            try:
                listener(action, record)
            except Exception:
                LOGGER.exception(
                    "Detection-log listener failed: action=%s log_id=%s",
                    action,
                    record.id,
                )

    def _connection(self) -> Connection:
        return self.database.connection()

    def _init_db(self) -> None:
        return None

    @staticmethod
    def _row_to_log(row: Row) -> DetectionLogRecord:
        return DetectionLogRecord(
            id=row["id"],
            source_system=row["source_system"],
            source_event_key=row.get("source_event_key"),
            source_human_log_id=row.get("source_human_log_id"),
            personnel_id=row.get("personnel_id"),
            person=row["person"],
            confidence=float(row["confidence"]),
            detection_time=row["detection_time"],
            ref_img_id=row.get("ref_img_id"),
            room_id=row.get("room_id"),
            camera_id=row.get("camera_id"),
            access_granted=bool(row["access_granted"]),
            counts_for_attendance=bool(row["counts_for_attendance"]),
            log_type=row["log_type"],
            import_source_parts=row.get("import_source_parts"),
            face_image=row.get("face_image"),
            face_thumbnail=row.get("face_thumbnail"),
            body_image=row.get("body_image"),
            snapshot_image=row.get("snapshot_image"),
            video=row.get("video"),
            face_video_or_unknown_faces=row.get("face_video_or_unknown_faces"),
            video_status=row.get("video_status") or "missing",
            face_video_status=row.get("face_video_status") or "missing",
            media_finalized_at=row.get("media_finalized_at"),
            created_by=row.get("created_by"),
            updated_by=row.get("updated_by"),
            created_at_utc=row["created_at_utc"],
            updated_at_utc=row["updated_at_utc"],
        )

    def create(
        self,
        source_system: str = "face_recognition",
        source_event_key: str | None = None,
        source_human_log_id: int | None = None,
        personnel_id: int | None = None,
        person: str = "Unknown",
        confidence: float = 0.0,
        detection_time: str = "",
        ref_img_id: str | None = None,
        room_id: int | None = None,
        camera_id: str | None = None,
        access_granted: bool = False,
        counts_for_attendance: bool = True,
        log_type: str = "real_time",
        import_source_parts: str | None = None,
        face_image: str | None = None,
        face_thumbnail: str | None = None,
        body_image: str | None = None,
        snapshot_image: str | None = None,
        video: str | None = None,
        face_video_or_unknown_faces: str | None = None,
        video_status: str = "missing",
        face_video_status: str = "missing",
        media_finalized_at: str | None = None,
        created_by: int | None = None,
    ) -> DetectionLogRecord:
        if video_status not in VALID_MEDIA_STATUSES:
            raise ValueError(f"Invalid video_status: {video_status}")
        if face_video_status not in VALID_MEDIA_STATUSES:
            raise ValueError(f"Invalid face_video_status: {face_video_status}")
        if not detection_time:
            detection_time = _now()
        now = _now()
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "INSERT INTO detection_logs "
                "(source_system, source_event_key, source_human_log_id, "
                "personnel_id, person, confidence, detection_time, ref_img_id, "
                "room_id, camera_id, access_granted, counts_for_attendance, "
                "log_type, import_source_parts, face_image, face_thumbnail, body_image, "
                "snapshot_image, video, face_video_or_unknown_faces, video_status, "
                "face_video_status, media_finalized_at, created_by, updated_by, "
                "created_at_utc, updated_at_utc) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    source_system, source_event_key, source_human_log_id,
                    personnel_id, person, confidence, detection_time, ref_img_id,
                    room_id, camera_id, int(access_granted), int(counts_for_attendance),
                    log_type, import_source_parts, face_image, face_thumbnail, body_image,
                    snapshot_image, video, face_video_or_unknown_faces, video_status,
                    face_video_status, media_finalized_at, created_by, created_by, now, now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM detection_logs WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            if row is None:
                raise RuntimeError("Failed to retrieve created detection log")
            record = self._row_to_log(row)
        self._notify("created", record)
        return record

    def get(self, log_id: int) -> DetectionLogRecord | None:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM detection_logs WHERE id = ?", (log_id,)
            ).fetchone()
            return self._row_to_log(row) if row is not None else None

    def get_by_source_event_key(self, source_event_key: str) -> DetectionLogRecord | None:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM detection_logs WHERE source_event_key = ?",
                (source_event_key,),
            ).fetchone()
            return self._row_to_log(row) if row is not None else None

    def update(self, log_id: int, **kwargs: Any) -> DetectionLogRecord | None:
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                "SELECT * FROM detection_logs WHERE id = ?", (log_id,)
            ).fetchone()
            if existing is None:
                return None
            if not kwargs:
                return self._row_to_log(existing)
            bool_fields = {"access_granted", "counts_for_attendance"}
            allowed_fields = {
                "source_system", "source_event_key", "source_human_log_id",
                "personnel_id", "person", "confidence", "detection_time",
                "ref_img_id", "room_id", "camera_id", "access_granted",
                "counts_for_attendance", "log_type", "import_source_parts",
                "face_image", "face_thumbnail", "body_image", "snapshot_image",
                "video", "face_video_or_unknown_faces", "video_status",
                "face_video_status", "media_finalized_at", "created_by", "updated_by",
            }
            unknown_fields = set(kwargs) - allowed_fields
            if unknown_fields:
                raise ValueError(f"Unsupported detection-log fields: {sorted(unknown_fields)}")
            for status_field in ("video_status", "face_video_status"):
                if (
                    status_field in kwargs
                    and kwargs[status_field] not in VALID_MEDIA_STATUSES
                ):
                    raise ValueError(
                        f"Invalid {status_field}: {kwargs[status_field]}"
                    )
            set_parts: list[str] = []
            params: list[Any] = []
            for key, value in kwargs.items():
                if key in bool_fields:
                    value = int(bool(value))
                set_parts.append(f"{key} = ?")
                params.append(value)
            set_parts.append("updated_at_utc = ?")
            params.append(_now())
            params.append(log_id)
            conn.execute(
                f"UPDATE detection_logs SET {', '.join(set_parts)} WHERE id = ?",
                params,
            )
            row = conn.execute(
                "SELECT * FROM detection_logs WHERE id = ?", (log_id,)
            ).fetchone()
            record = self._row_to_log(row) if row is not None else None
        if record is not None:
            self._notify("updated", record)
        return record

    def delete(self, log_id: int) -> bool:
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM detection_logs WHERE id = ?", (log_id,)
            )
            return cursor.rowcount > 0

    def delete_in_time_range(
        self,
        from_date_utc: str,
        to_date_utc: str,
    ) -> list[DetectionLogRecord]:
        """Delete and return logs in the half-open UTC interval [start, end)."""
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "DELETE FROM detection_logs "
                "WHERE detection_time >= ?::timestamptz "
                "AND detection_time < ?::timestamptz "
                "RETURNING *",
                (from_date_utc, to_date_utc),
            ).fetchall()
            return [self._row_to_log(row) for row in rows]

    def find_dedup(
        self,
        personnel_id: int,
        room_id: int,
        detection_time_utc_str: str,
        time_window_seconds: int = 60,
    ) -> DetectionLogRecord | None:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM detection_logs "
                "WHERE personnel_id = ? AND room_id = ? "
                "AND detection_time >= (?::timestamptz - make_interval(secs => ?)) "
                "AND detection_time <= ?::timestamptz "
                "ORDER BY detection_time DESC LIMIT 1",
                (
                    personnel_id,
                    room_id,
                    detection_time_utc_str,
                    float(time_window_seconds),
                    detection_time_utc_str,
                ),
            ).fetchone()
            return self._row_to_log(row) if row is not None else None

    def list_filter(
        self,
        offset: int = 0,
        limit: int = 200,
        personnel_id: int | None = None,
        national_code: str | None = None,
        room_id: int | None = None,
        camera_id: str | None = None,
        section_id: int | None = None,
        building_id: int | None = None,
        access_granted: bool | None = None,
        counts_for_attendance: bool | None = None,
        log_type: str | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
        from_date_utc: str | None = None,
        to_date_utc: str | None = None,
        include_thumbnails: bool = False,
        include_total: bool = True,
    ) -> tuple[list[DetectionLogRecord], int]:
        where_clauses: list[str] = []
        params: list[Any] = []
        joins: list[str] = []

        if personnel_id is not None:
            where_clauses.append("d.personnel_id = ?")
            params.append(personnel_id)
        if national_code is not None:
            joins.append("LEFT JOIN personnel p ON d.personnel_id = p.id")
            where_clauses.append("p.national_code = ?")
            params.append(national_code)
        if room_id is not None:
            where_clauses.append("d.room_id = ?")
            params.append(room_id)
        if camera_id is not None:
            where_clauses.append("d.camera_id = ?")
            params.append(camera_id)
        if section_id is not None or building_id is not None:
            joins.append("LEFT JOIN rooms r ON d.room_id = r.id")
            joins.append("LEFT JOIN sections s ON r.section_id = s.id")
            if section_id is not None:
                where_clauses.append("r.section_id = ?")
                params.append(section_id)
            if building_id is not None:
                joins.append("LEFT JOIN buildings b ON s.building_id = b.id")
                where_clauses.append("s.building_id = ?")
                params.append(building_id)
        if access_granted is not None:
            where_clauses.append("d.access_granted = ?")
            params.append(int(access_granted))
        if counts_for_attendance is not None:
            where_clauses.append("d.counts_for_attendance = ?")
            params.append(int(counts_for_attendance))
        if log_type is not None:
            where_clauses.append("d.log_type = ?")
            params.append(log_type)
        if min_confidence is not None:
            where_clauses.append("d.confidence >= ?")
            params.append(min_confidence)
        if max_confidence is not None:
            where_clauses.append("d.confidence <= ?")
            params.append(max_confidence)
        if from_date_utc is not None:
            where_clauses.append("d.detection_time >= ?::timestamptz")
            params.append(from_date_utc)
        if to_date_utc is not None:
            where_clauses.append("d.detection_time < ?::timestamptz")
            params.append(to_date_utc)

        where = ""
        if where_clauses:
            where = " WHERE " + " AND ".join(where_clauses)
        join_clause = " ".join(joins)

        select_cols = (
            "d.id, d.source_system, d.source_event_key, d.source_human_log_id, "
            "d.personnel_id, d.person, d.confidence, d.detection_time, d.ref_img_id, "
            "d.room_id, d.camera_id, d.access_granted, d.counts_for_attendance, "
            "d.log_type, d.import_source_parts, d.face_image, d.body_image, "
            "d.snapshot_image, d.video, d.face_video_or_unknown_faces, "
            "d.video_status, d.face_video_status, d.media_finalized_at, "
            "d.created_by, d.updated_by, d.created_at_utc, d.updated_at_utc"
        )
        if include_thumbnails:
            select_cols += ", d.face_thumbnail"
        else:
            select_cols += ", NULL AS face_thumbnail"

        with self._lock, self._connection() as conn:
            total = 0
            if include_total:
                total = conn.execute(
                    f"SELECT COUNT(*) FROM detection_logs d {join_clause}{where}",
                    params,
                ).fetchone()[0]
            rows = conn.execute(
                f"SELECT {select_cols} FROM detection_logs d {join_clause}{where} "
                "ORDER BY d.detection_time DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            return [self._row_to_log(r) for r in rows], int(total)

    def count_by_status(self) -> dict[str, int]:
        with self._lock, self._connection() as conn:
            by_type = conn.execute(
                "SELECT log_type, COUNT(*) as cnt FROM detection_logs GROUP BY log_type"
            ).fetchall()
            by_access = conn.execute(
                "SELECT access_granted, COUNT(*) as cnt FROM detection_logs GROUP BY access_granted"
            ).fetchall()
            result: dict[str, int] = {}
            for r in by_type:
                result[f"log_type:{r['log_type']}"] = int(r["cnt"])
            for r in by_access:
                result[f"access_granted:{bool(r['access_granted'])}"] = int(r["cnt"])
            return result

    def bulk_delete(self, ids: list[int]) -> int:
        if not ids:
            return 0
        placeholders = ", ".join("?" for _ in ids)
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                f"DELETE FROM detection_logs WHERE id IN ({placeholders})",
                ids,
            )
            return cursor.rowcount

    def is_media_key_referenced(self, media_key: str) -> bool:
        if not media_key:
            return False
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM detection_logs WHERE "
                "face_image = ? OR face_thumbnail = ? OR body_image = ? OR "
                "snapshot_image = ? OR video = ? OR "
                "face_video_or_unknown_faces = ? LIMIT 1",
                (media_key,) * 6,
            ).fetchone()
            return row is not None

    def list_all(self) -> list[DetectionLogRecord]:
        with self._lock, self._connection() as conn:
            rows = conn.execute("SELECT * FROM detection_logs ORDER BY id").fetchall()
            return [self._row_to_log(row) for row in rows]

    def delete_all(self) -> int:
        with self._lock, self._connection() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM detection_logs").fetchone()
            count = int(row["count"] if row is not None else 0)
            conn.execute("TRUNCATE TABLE detection_logs")
            return count
