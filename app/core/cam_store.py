from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from app.database import Database, IntegrityError, Row, ensure_database
from app.time_utils import utc_now_text

CAM_SOURCE_TYPES = frozenset({"usb", "rtsp", "other"})


@dataclass(frozen=True, slots=True)
class CamRecord:
    id: int
    camera_name: str
    camera_number: int
    width: int
    high: int
    source_type: str
    section_id: int
    url: str
    created_at_utc: str
    updated_at_utc: str
    created_by: int | None = None
    updated_by: int | None = None


class CamStore:
    """PostgreSQL-backed CRUD store for the dedicated ``cam`` table."""

    def __init__(self, database: Database | str) -> None:
        self.database = ensure_database(database)
        self._lock = threading.RLock()

    @staticmethod
    def _row_to_record(row: Row) -> CamRecord:
        return CamRecord(
            id=int(row["id"]),
            camera_name=str(row["camera_name"]),
            camera_number=int(row["camera_number"]),
            width=int(row["width"]),
            high=int(row["high"]),
            source_type=str(row["source_type"]),
            section_id=int(row["section_id"]),
            url=str(row["url"]),
            created_at_utc=str(row["created_at_utc"]),
            updated_at_utc=str(row["updated_at_utc"]),
            created_by=row.get("created_by"),
            updated_by=row.get("updated_by"),
        )

    @staticmethod
    def _clean_values(
        *,
        camera_name: str,
        camera_number: int,
        width: int,
        high: int,
        source_type: str,
        section_id: int,
        url: str,
    ) -> dict[str, Any]:
        name = camera_name.strip()
        source_url = url.strip()
        normalized_source_type = source_type.strip().lower()

        if not name:
            raise ValueError("camera_name cannot be blank")
        if not source_url:
            raise ValueError("url cannot be blank")
        if camera_number < 1:
            raise ValueError("camera_number must be greater than zero")
        if width < 1:
            raise ValueError("width must be greater than zero")
        if high < 1:
            raise ValueError("high must be greater than zero")
        if normalized_source_type not in CAM_SOURCE_TYPES:
            raise ValueError(
                f"source_type must be one of {sorted(CAM_SOURCE_TYPES)}"
            )
        if section_id < 1:
            raise ValueError("section_id must be greater than zero")

        return {
            "camera_name": name,
            "camera_number": int(camera_number),
            "width": int(width),
            "high": int(high),
            "source_type": normalized_source_type,
            "section_id": int(section_id),
            "url": source_url,
        }

    @staticmethod
    def _validate_section(conn, *, section_id: int) -> None:
        section = conn.execute(
            "SELECT id FROM sections WHERE id = ?", (section_id,)
        ).fetchone()
        if section is None:
            raise ValueError(f"Section not found: {section_id}")

    def create(
        self,
        *,
        camera_name: str,
        camera_number: int,
        width: int,
        high: int,
        source_type: str,
        section_id: int,
        url: str,
        created_by: int | None = None,
    ) -> CamRecord:
        values = self._clean_values(
            camera_name=camera_name,
            camera_number=camera_number,
            width=width,
            high=high,
            source_type=source_type,
            section_id=section_id,
            url=url,
        )
        now = utc_now_text()
        try:
            with self._lock, self.database.connection() as conn:
                self._validate_section(conn, section_id=values["section_id"])
                cursor = conn.execute(
                    "INSERT INTO cam (camera_name, camera_number, width, high, source_type, "
                    "section_id, url, created_at_utc, updated_at_utc, created_by) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        values["camera_name"],
                        values["camera_number"],
                        values["width"],
                        values["high"],
                        values["source_type"],
                        values["section_id"],
                        values["url"],
                        now,
                        now,
                        created_by,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM cam WHERE id = ?", (cursor.lastrowid,)
                ).fetchone()
                if row is None:
                    raise RuntimeError("Failed to retrieve created cam")
                return self._row_to_record(row)
        except IntegrityError as exc:
            raise ValueError(
                "camera_number must be unique within the selected section"
            ) from exc

    def get(self, cam_id: int) -> CamRecord | None:
        with self._lock, self.database.connection() as conn:
            row = conn.execute(
                "SELECT * FROM cam WHERE id = ?", (cam_id,)
            ).fetchone()
            return self._row_to_record(row) if row is not None else None

    def list(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
        section_id: int | None = None,
        source_type: str | None = None,
    ) -> tuple[list[CamRecord], int]:
        clauses: list[str] = []
        params: list[Any] = []
        if section_id is not None:
            clauses.append("section_id = ?")
            params.append(section_id)
        if source_type is not None:
            normalized = source_type.strip().lower()
            if normalized not in CAM_SOURCE_TYPES:
                raise ValueError(
                    f"source_type must be one of {sorted(CAM_SOURCE_TYPES)}"
                )
            clauses.append("source_type = ?")
            params.append(normalized)

        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._lock, self.database.connection() as conn:
            total_row = conn.execute(
                f"SELECT COUNT(*) FROM cam{where}", params
            ).fetchone()
            rows = conn.execute(
                f"SELECT * FROM cam{where} ORDER BY id DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            return [self._row_to_record(row) for row in rows], int(total_row[0])

    def count(self, *, section_id: int | None = None) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if section_id is not None:
            clauses.append("section_id = ?")
            params.append(section_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._lock, self.database.connection() as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM cam{where}", params).fetchone()
            return int(row[0])

    def update(self, cam_id: int, **changes: Any) -> CamRecord | None:
        allowed = {
            "camera_name",
            "camera_number",
            "width",
            "high",
            "source_type",
            "section_id",
            "url",
            "updated_by",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unsupported cam fields: {sorted(unknown)}")
        if not changes:
            raise ValueError("No fields to update")

        try:
            with self._lock, self.database.connection() as conn:
                existing = conn.execute(
                    "SELECT * FROM cam WHERE id = ?", (cam_id,)
                ).fetchone()
                if existing is None:
                    return None

                merged = self._clean_values(
                    camera_name=changes.get("camera_name", existing["camera_name"]),
                    camera_number=changes.get("camera_number", existing["camera_number"]),
                    width=changes.get("width", existing["width"]),
                    high=changes.get("high", existing["high"]),
                    source_type=changes.get("source_type", existing["source_type"]),
                    section_id=changes.get("section_id", existing["section_id"]),
                    url=changes.get("url", existing["url"]),
                )
                updated_by = changes.get("updated_by", existing.get("updated_by"))
                self._validate_section(conn, section_id=merged["section_id"])
                conn.execute(
                    "UPDATE cam SET camera_name=?, camera_number=?, width=?, high=?, "
                    "source_type=?, section_id=?, url=?, updated_at_utc=?, updated_by=? WHERE id=?",
                    (
                        merged["camera_name"],
                        merged["camera_number"],
                        merged["width"],
                        merged["high"],
                        merged["source_type"],
                        merged["section_id"],
                        merged["url"],
                        utc_now_text(),
                        updated_by,
                        cam_id,
                    ),
                )
                # ``rooms.section_id`` is retained as a compatibility/cache column.
                # Keep it synchronized with the room's parent cam so the effective
                # hierarchy remains building -> section -> cam -> room.
                if int(existing["section_id"]) != merged["section_id"]:
                    conn.execute(
                        "UPDATE rooms SET section_id=?, updated_at_utc=? WHERE cam_id=?",
                        (merged["section_id"], utc_now_text(), cam_id),
                    )
                row = conn.execute(
                    "SELECT * FROM cam WHERE id = ?", (cam_id,)
                ).fetchone()
                return self._row_to_record(row) if row is not None else None
        except IntegrityError as exc:
            raise ValueError(
                "camera_number must be unique within the selected section"
            ) from exc

    def delete(self, cam_id: int) -> bool:
        with self._lock, self.database.connection() as conn:
            existing = conn.execute(
                "SELECT id FROM cam WHERE id = ?", (cam_id,)
            ).fetchone()
            if existing is None:
                return False
            room_count_row = conn.execute(
                "SELECT COUNT(*) FROM rooms WHERE cam_id = ?", (cam_id,)
            ).fetchone()
            room_count = int(room_count_row[0]) if room_count_row else 0
            if room_count:
                raise ValueError(
                    f"Cam cannot be deleted because it is assigned to {room_count} room(s)"
                )
            cursor = conn.execute("DELETE FROM cam WHERE id = ?", (cam_id,))
            return cursor.rowcount > 0
