from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BuildingRecord:
    id: int
    name: str
    address: str | None
    description: str | None
    created_at_utc: str
    updated_at_utc: str


@dataclass(frozen=True, slots=True)
class SectionRecord:
    id: int
    building_id: int | None
    name: str
    description: str | None
    created_at_utc: str
    updated_at_utc: str


@dataclass(frozen=True, slots=True)
class RoomRecord:
    id: int
    section_id: int | None
    name: str
    description: str | None
    polygon_json: str | None
    created_at_utc: str
    updated_at_utc: str


@dataclass(frozen=True, slots=True)
class PersonnelRoomAccessRecord:
    id: int
    personnel_id: int
    room_id: int
    granted_at_utc: str
    granted_by: str | None


@dataclass(frozen=True, slots=True)
class DetectionRoomMatchRecord:
    id: int
    detection_type: str
    detection_event_id: int
    room_id: int
    personnel_id: int | None
    camera_id: str | None
    matched_at_utc: str


def point_in_polygon(px: float, py: float, polygon: list[list[float]]) -> bool:
    """PNPoly algorithm: check if point (px, py) is inside a 2D polygon.

    The polygon should be a list of [x, y] vertices, either open or closed.
    """
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside


def parse_polygon(polygon_json: str | None) -> list[list[float]]:
    """Parse a polygon JSON string into a list of [x, y] points."""
    if not polygon_json:
        return []
    try:
        points: list[list[float]] = json.loads(polygon_json)
        if not isinstance(points, list) or len(points) < 3:
            return []
        # Validate each point
        for p in points:
            if not isinstance(p, list) or len(p) < 2:
                return []
        return points
    except (json.JSONDecodeError, TypeError):
        return []


class LocationStore:
    """SQLite-backed store for Buildings, Sections, Rooms, access, and detection matching."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path.resolve()
        self._lock = threading.RLock()
        self._init_db()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS buildings (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    name            TEXT    NOT NULL,
                    address         TEXT,
                    description     TEXT,
                    created_at_utc  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    updated_at_utc  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
                );
                CREATE INDEX IF NOT EXISTS idx_buildings_name ON buildings(name);

                CREATE TABLE IF NOT EXISTS sections (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    building_id     INTEGER REFERENCES buildings(id) ON DELETE SET NULL,
                    name            TEXT    NOT NULL,
                    description     TEXT,
                    created_at_utc  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    updated_at_utc  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
                );
                CREATE INDEX IF NOT EXISTS idx_sections_building ON sections(building_id);
                CREATE INDEX IF NOT EXISTS idx_sections_name ON sections(name);

                CREATE TABLE IF NOT EXISTS rooms (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    section_id      INTEGER REFERENCES sections(id) ON DELETE SET NULL,
                    name            TEXT    NOT NULL,
                    description     TEXT,
                    polygon_json    TEXT,
                    created_at_utc  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    updated_at_utc  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
                );
                CREATE INDEX IF NOT EXISTS idx_rooms_section ON rooms(section_id);
                CREATE INDEX IF NOT EXISTS idx_rooms_name ON rooms(name);

                CREATE TABLE IF NOT EXISTS personnel_room_access (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    personnel_id    INTEGER NOT NULL,
                    room_id         INTEGER NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
                    granted_at_utc  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    granted_by      TEXT,
                    UNIQUE(personnel_id, room_id)
                );
                CREATE INDEX IF NOT EXISTS idx_access_personnel ON personnel_room_access(personnel_id);
                CREATE INDEX IF NOT EXISTS idx_access_room ON personnel_room_access(room_id);

                CREATE TABLE IF NOT EXISTS detection_room_matches (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    detection_type  TEXT    NOT NULL,
                    detection_event_id INTEGER NOT NULL,
                    room_id         INTEGER NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
                    personnel_id    INTEGER,
                    camera_id       TEXT,
                    matched_at_utc  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
                );
                CREATE INDEX IF NOT EXISTS idx_matches_room ON detection_room_matches(room_id);
                CREATE INDEX IF NOT EXISTS idx_matches_personnel ON detection_room_matches(personnel_id);
                CREATE INDEX IF NOT EXISTS idx_matches_detection ON detection_room_matches(detection_type, detection_event_id);
            """)

            # Check for columns that might be missing (migrations)
            self._migrate_columns(conn)

    def _migrate_columns(self, conn: sqlite3.Connection) -> None:
        """Add any missing columns from schema evolution."""
        tables = {
            "buildings": {"address", "description"},
            "sections": {"description"},
            "rooms": {"description"},
        }
        for table, expected_cols in tables.items():
            existing = {
                str(row[1])
                for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for col in expected_cols:
                if col not in existing:
                    try:
                        conn.execute(
                            f"ALTER TABLE {table} ADD COLUMN {col} TEXT"
                        )
                    except sqlite3.OperationalError:
                        pass

    def _now(self) -> str:
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Buildings ─────────────────────────────────────────────────────

    @staticmethod
    def _row_to_building(row: sqlite3.Row) -> BuildingRecord:
        return BuildingRecord(
            id=row["id"],
            name=row["name"],
            address=row["address"],
            description=row["description"],
            created_at_utc=row["created_at_utc"],
            updated_at_utc=row["updated_at_utc"],
        )

    def create_building(
        self, name: str, address: str | None = None, description: str | None = None
    ) -> BuildingRecord:
        name = name.strip()
        if not name:
            raise ValueError("Building name is required")
        now = self._now()
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "INSERT INTO buildings (name, address, description, created_at_utc, updated_at_utc) "
                "VALUES (?, ?, ?, ?, ?)",
                (name, address, description, now, now),
            )
            row = conn.execute(
                "SELECT * FROM buildings WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            if row is None:
                raise RuntimeError("Failed to retrieve created building")
            return self._row_to_building(row)

    def get_building(self, building_id: int) -> BuildingRecord | None:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM buildings WHERE id = ?", (building_id,)
            ).fetchone()
            return self._row_to_building(row) if row is not None else None

    def update_building(
        self,
        building_id: int,
        name: str | None = None,
        address: str | None = None,
        description: str | None = None,
    ) -> BuildingRecord | None:
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                "SELECT * FROM buildings WHERE id = ?", (building_id,)
            ).fetchone()
            if existing is None:
                return None
            new_name = name.strip() if name else existing["name"]
            if name is not None and not new_name:
                raise ValueError("Building name cannot be blank")
            new_address = address if address is not None else existing["address"]
            new_description = description if description is not None else existing["description"]
            now = self._now()
            conn.execute(
                "UPDATE buildings SET name=?, address=?, description=?, updated_at_utc=? WHERE id=?",
                (new_name, new_address, new_description, now, building_id),
            )
            row = conn.execute(
                "SELECT * FROM buildings WHERE id = ?", (building_id,)
            ).fetchone()
            return self._row_to_building(row)

    def delete_building(self, building_id: int) -> bool:
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM buildings WHERE id = ?", (building_id,)
            )
            return cursor.rowcount > 0

    def list_buildings(
        self, offset: int = 0, limit: int = 50, search: str | None = None
    ) -> tuple[list[BuildingRecord], int]:
        where = ""
        params: list[Any] = []
        if search is not None:
            where = " WHERE name LIKE ? OR description LIKE ?"
            pattern = f"%{search}%"
            params = [pattern, pattern]
        with self._lock, self._connection() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM buildings{where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM buildings{where} ORDER BY id DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            return [self._row_to_building(r) for r in rows], int(total)

    # ── Sections ──────────────────────────────────────────────────────

    @staticmethod
    def _row_to_section(row: sqlite3.Row) -> SectionRecord:
        return SectionRecord(
            id=row["id"],
            building_id=row["building_id"],
            name=row["name"],
            description=row["description"],
            created_at_utc=row["created_at_utc"],
            updated_at_utc=row["updated_at_utc"],
        )

    def create_section(
        self,
        name: str,
        building_id: int | None = None,
        description: str | None = None,
    ) -> SectionRecord:
        name = name.strip()
        if not name:
            raise ValueError("Section name is required")
        now = self._now()
        with self._lock, self._connection() as conn:
            if building_id is not None:
                bld = conn.execute(
                    "SELECT id FROM buildings WHERE id = ?", (building_id,)
                ).fetchone()
                if bld is None:
                    raise ValueError(f"Building not found: {building_id}")
            cursor = conn.execute(
                "INSERT INTO sections (building_id, name, description, created_at_utc, updated_at_utc) "
                "VALUES (?, ?, ?, ?, ?)",
                (building_id, name, description, now, now),
            )
            row = conn.execute(
                "SELECT * FROM sections WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            if row is None:
                raise RuntimeError("Failed to retrieve created section")
            return self._row_to_section(row)

    def get_section(self, section_id: int) -> SectionRecord | None:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM sections WHERE id = ?", (section_id,)
            ).fetchone()
            return self._row_to_section(row) if row is not None else None

    def update_section(
        self,
        section_id: int,
        name: str | None = None,
        building_id: int | None = None,
        description: str | None = None,
    ) -> SectionRecord | None:
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                "SELECT * FROM sections WHERE id = ?", (section_id,)
            ).fetchone()
            if existing is None:
                return None
            new_name = name.strip() if name else existing["name"]
            if name is not None and not new_name:
                raise ValueError("Section name cannot be blank")
            new_building_id = (
                building_id if building_id is not None else existing["building_id"]
            )
            if building_id is not None:
                bld = conn.execute(
                    "SELECT id FROM buildings WHERE id = ?", (building_id,)
                ).fetchone()
                if bld is None:
                    raise ValueError(f"Building not found: {building_id}")
            new_description = description if description is not None else existing["description"]
            now = self._now()
            conn.execute(
                "UPDATE sections SET name=?, building_id=?, description=?, updated_at_utc=? WHERE id=?",
                (new_name, new_building_id, new_description, now, section_id),
            )
            row = conn.execute(
                "SELECT * FROM sections WHERE id = ?", (section_id,)
            ).fetchone()
            return self._row_to_section(row)

    def delete_section(self, section_id: int) -> bool:
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM sections WHERE id = ?", (section_id,)
            )
            return cursor.rowcount > 0

    def list_sections(
        self,
        offset: int = 0,
        limit: int = 50,
        building_id: int | None = None,
        search: str | None = None,
    ) -> tuple[list[SectionRecord], int]:
        where_clauses: list[str] = []
        params: list[Any] = []
        if building_id is not None:
            where_clauses.append("building_id = ?")
            params.append(building_id)
        if search is not None:
            where_clauses.append("(name LIKE ? OR description LIKE ?)")
            pattern = f"%{search}%"
            params.extend([pattern, pattern])
        where = ""
        if where_clauses:
            where = " WHERE " + " AND ".join(where_clauses)
        with self._lock, self._connection() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM sections{where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM sections{where} ORDER BY id DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            return [self._row_to_section(r) for r in rows], int(total)

    def filter_cameras_by_section(
        self,
        cameras: list[dict[str, Any]],
        section_id: int,
    ) -> list[dict[str, Any]]:
        """Filter a list of camera dicts by section_id.

        Cameras should have a 'metadata' dict or 'section_id' key.
        """
        return [
            cam
            for cam in cameras
            if cam.get("section_id") == section_id
            or cam.get("metadata", {}).get("section_id") == section_id
        ]

    # ── Rooms ─────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_room(row: sqlite3.Row) -> RoomRecord:
        return RoomRecord(
            id=row["id"],
            section_id=row["section_id"],
            name=row["name"],
            description=row["description"],
            polygon_json=row["polygon_json"],
            created_at_utc=row["created_at_utc"],
            updated_at_utc=row["updated_at_utc"],
        )

    def create_room(
        self,
        name: str,
        section_id: int | None = None,
        description: str | None = None,
        polygon_json: str | None = None,
    ) -> RoomRecord:
        name = name.strip()
        if not name:
            raise ValueError("Room name is required")
        # Validate polygon if provided
        if polygon_json:
            points = parse_polygon(polygon_json)
            if len(points) < 3:
                raise ValueError("Polygon must have at least 3 vertices")
        now = self._now()
        with self._lock, self._connection() as conn:
            if section_id is not None:
                sec = conn.execute(
                    "SELECT id FROM sections WHERE id = ?", (section_id,)
                ).fetchone()
                if sec is None:
                    raise ValueError(f"Section not found: {section_id}")
            cursor = conn.execute(
                "INSERT INTO rooms (section_id, name, description, polygon_json, created_at_utc, updated_at_utc) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (section_id, name, description, polygon_json, now, now),
            )
            row = conn.execute(
                "SELECT * FROM rooms WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            if row is None:
                raise RuntimeError("Failed to retrieve created room")
            return self._row_to_room(row)

    def get_room(self, room_id: int) -> RoomRecord | None:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM rooms WHERE id = ?", (room_id,)
            ).fetchone()
            return self._row_to_room(row) if row is not None else None

    def update_room(
        self,
        room_id: int,
        name: str | None = None,
        section_id: int | None = None,
        description: str | None = None,
        polygon_json: str | None = None,
    ) -> RoomRecord | None:
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                "SELECT * FROM rooms WHERE id = ?", (room_id,)
            ).fetchone()
            if existing is None:
                return None
            new_name = name.strip() if name else existing["name"]
            if name is not None and not new_name:
                raise ValueError("Room name cannot be blank")
            new_section_id = (
                section_id if section_id is not None else existing["section_id"]
            )
            if section_id is not None:
                sec = conn.execute(
                    "SELECT id FROM sections WHERE id = ?", (section_id,)
                ).fetchone()
                if sec is None:
                    raise ValueError(f"Section not found: {section_id}")
            new_description = description if description is not None else existing["description"]
            new_polygon = polygon_json if polygon_json is not None else existing["polygon_json"]
            if polygon_json is not None:
                points = parse_polygon(polygon_json)
                if len(points) < 3:
                    raise ValueError("Polygon must have at least 3 vertices")
            now = self._now()
            conn.execute(
                "UPDATE rooms SET name=?, section_id=?, description=?, polygon_json=?, "
                "updated_at_utc=? WHERE id=?",
                (new_name, new_section_id, new_description, new_polygon, now, room_id),
            )
            row = conn.execute(
                "SELECT * FROM rooms WHERE id = ?", (room_id,)
            ).fetchone()
            return self._row_to_room(row)

    def delete_room(self, room_id: int) -> bool:
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM rooms WHERE id = ?", (room_id,)
            )
            return cursor.rowcount > 0

    def list_rooms(
        self,
        offset: int = 0,
        limit: int = 50,
        section_id: int | None = None,
        search: str | None = None,
    ) -> tuple[list[RoomRecord], int]:
        where_clauses: list[str] = []
        params: list[Any] = []
        if section_id is not None:
            where_clauses.append("section_id = ?")
            params.append(section_id)
        if search is not None:
            where_clauses.append("(name LIKE ? OR description LIKE ?)")
            pattern = f"%{search}%"
            params.extend([pattern, pattern])
        where = ""
        if where_clauses:
            where = " WHERE " + " AND ".join(where_clauses)
        with self._lock, self._connection() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM rooms{where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM rooms{where} ORDER BY id DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            return [self._row_to_room(r) for r in rows], int(total)

    # ── Personnel Room Access ─────────────────────────────────────────

    def grant_room_access(
        self, personnel_id: int, room_id: int, granted_by: str | None = None
    ) -> PersonnelRoomAccessRecord:
        now = self._now()
        with self._lock, self._connection() as conn:
            # Verify room exists
            room = conn.execute(
                "SELECT id FROM rooms WHERE id = ?", (room_id,)
            ).fetchone()
            if room is None:
                raise ValueError(f"Room not found: {room_id}")
            try:
                cursor = conn.execute(
                    "INSERT INTO personnel_room_access (personnel_id, room_id, granted_at_utc, granted_by) "
                    "VALUES (?, ?, ?, ?)",
                    (personnel_id, room_id, now, granted_by),
                )
                row = conn.execute(
                    "SELECT * FROM personnel_room_access WHERE id = ?",
                    (cursor.lastrowid,),
                ).fetchone()
                if row is None:
                    raise RuntimeError("Failed to retrieve created access record")
                return PersonnelRoomAccessRecord(
                    id=row["id"],
                    personnel_id=row["personnel_id"],
                    room_id=row["room_id"],
                    granted_at_utc=row["granted_at_utc"],
                    granted_by=row["granted_by"],
                )
            except sqlite3.IntegrityError:
                raise ValueError(
                    f"Personnel {personnel_id} already has access to room {room_id}"
                )

    def revoke_room_access(self, personnel_id: int, room_id: int) -> bool:
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM personnel_room_access WHERE personnel_id = ? AND room_id = ?",
                (personnel_id, room_id),
            )
            return cursor.rowcount > 0

    def check_room_access(self, personnel_id: int, room_id: int) -> bool:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT id FROM personnel_room_access WHERE personnel_id = ? AND room_id = ?",
                (personnel_id, room_id),
            ).fetchone()
            return row is not None

    def list_personnel_rooms(self, personnel_id: int) -> list[RoomRecord]:
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT r.* FROM rooms r "
                "INNER JOIN personnel_room_access a ON a.room_id = r.id "
                "WHERE a.personnel_id = ? "
                "ORDER BY r.name",
                (personnel_id,),
            ).fetchall()
            return [self._row_to_room(r) for r in rows]

    def list_room_personnel(
        self, room_id: int
    ) -> list[dict[str, Any]]:
        """Return personnel IDs who have access to this room."""
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT a.personnel_id, a.granted_at_utc, a.granted_by "
                "FROM personnel_room_access a "
                "WHERE a.room_id = ? "
                "ORDER BY a.granted_at_utc",
                (room_id,),
            ).fetchall()
            return [
                {
                    "personnel_id": row["personnel_id"],
                    "granted_at_utc": row["granted_at_utc"],
                    "granted_by": row["granted_by"],
                }
                for row in rows
            ]

    # ── Detection Room Matching (polygon matching) ─────────────────────

    def match_detection_to_rooms(
        self,
        section_id: int,
        detection_type: str,
        detection_event_id: int,
        bbox_center_x: float,
        bbox_center_y: float,
        personnel_id: int | None = None,
        camera_id: str | None = None,
    ) -> list[DetectionRoomMatchRecord]:
        """Match a detection point to rooms via polygon containment.

        Checks every room in the given section for polygon containment
        of the specified point. The caller is responsible for resolving
        the section_id from the camera/source.

        Returns a list of DetectionRoomMatchRecord for each matched room.
        """
        if section_id is None:
            return []
        with self._lock, self._connection() as conn:
            rooms = conn.execute(
                "SELECT id, polygon_json FROM rooms WHERE section_id = ? AND polygon_json IS NOT NULL",
                (section_id,),
            ).fetchall()

            matched: list[DetectionRoomMatchRecord] = []
            now = self._now()

            for room in rooms:
                polygon = parse_polygon(room["polygon_json"])
                if len(polygon) < 3:
                    continue
                if point_in_polygon(bbox_center_x, bbox_center_y, polygon):
                    cursor = conn.execute(
                        "INSERT INTO detection_room_matches "
                        "(detection_type, detection_event_id, room_id, personnel_id, camera_id, matched_at_utc) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (detection_type, detection_event_id, room["id"], personnel_id, camera_id, now),
                    )
                    row = conn.execute(
                        "SELECT * FROM detection_room_matches WHERE id = ?",
                        (cursor.lastrowid,),
                    ).fetchone()
                    if row is not None:
                        matched.append(DetectionRoomMatchRecord(
                            id=row["id"],
                            detection_type=row["detection_type"],
                            detection_event_id=row["detection_event_id"],
                            room_id=row["room_id"],
                            personnel_id=row["personnel_id"],
                            camera_id=row["camera_id"],
                            matched_at_utc=row["matched_at_utc"],
                        ))
            return matched

    def get_matches_for_detection(
        self, detection_type: str, detection_event_id: int
    ) -> list[DetectionRoomMatchRecord]:
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM detection_room_matches "
                "WHERE detection_type = ? AND detection_event_id = ?",
                (detection_type, detection_event_id),
            ).fetchall()
            return [
                DetectionRoomMatchRecord(
                    id=row["id"],
                    detection_type=row["detection_type"],
                    detection_event_id=row["detection_event_id"],
                    room_id=row["room_id"],
                    personnel_id=row["personnel_id"],
                    camera_id=row["camera_id"],
                    matched_at_utc=row["matched_at_utc"],
                )
                for row in rows
            ]

    def list_matches_for_room(
        self, room_id: int, limit: int = 50, offset: int = 0
    ) -> tuple[list[DetectionRoomMatchRecord], int]:
        with self._lock, self._connection() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM detection_room_matches WHERE room_id = ?",
                (room_id,),
            ).fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM detection_room_matches WHERE room_id = ? "
                "ORDER BY matched_at_utc DESC LIMIT ? OFFSET ?",
                (room_id, limit, offset),
            ).fetchall()
            records = [
                DetectionRoomMatchRecord(
                    id=row["id"],
                    detection_type=row["detection_type"],
                    detection_event_id=row["detection_event_id"],
                    room_id=row["room_id"],
                    personnel_id=row["personnel_id"],
                    camera_id=row["camera_id"],
                    matched_at_utc=row["matched_at_utc"],
                )
                for row in rows
            ]
            return records, int(total)

    # ── Counts ────────────────────────────────────────────────────────

    def count_buildings(self) -> int:
        with self._lock, self._connection() as conn:
            return int(
                conn.execute("SELECT COUNT(*) FROM buildings").fetchone()[0]
            )

    def count_sections(self) -> int:
        with self._lock, self._connection() as conn:
            return int(
                conn.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
            )

    def count_rooms(self) -> int:
        with self._lock, self._connection() as conn:
            return int(
                conn.execute("SELECT COUNT(*) FROM rooms").fetchone()[0]
            )
