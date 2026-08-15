from __future__ import annotations

from app.database import Connection, Database, IntegrityError, OperationalError, Row, ensure_database
from app.time_utils import utc_now_text

import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# Default full-frame polygon used when no rooms in a section have custom polygons.
# Covers the entire 640x640 frame: [[0,0], [0,640], [640,640], [640,0]].
DEFAULT_POLYGON: list[list[float]] = [[0.0, 0.0], [0.0, 640.0], [640.0, 640.0], [640.0, 0.0]]
# Sentinel room_id used for default-polygon transition tracking (not stored in DB).
_DEFAULT_ROOM_ID: int = -1
_UNSET = object()
_POLYGON_CACHE_MAX_ENTRIES = 1024


@dataclass(frozen=True, slots=True)
class BuildingRecord:
    id: int
    name: str
    address: str | None
    description: str | None
    created_at_utc: str
    updated_at_utc: str
    created_by: int | None = None
    updated_by: int | None = None


@dataclass(frozen=True, slots=True)
class SectionRecord:
    id: int
    building_id: int | None
    name: str
    description: str | None
    created_at_utc: str
    updated_at_utc: str
    created_by: int | None = None
    updated_by: int | None = None
    floor: str | None = None
    is_active: bool = True


@dataclass(frozen=True, slots=True)
class RoomRecord:
    id: int
    section_id: int | None
    cam_id: int | None
    name: str
    room_number: str | None
    room_type: str | None
    description: str | None
    is_active: bool
    polygon_json: str | None
    created_at_utc: str
    updated_at_utc: str
    created_by: int | None = None
    updated_by: int | None = None


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
    track_id: int | None = None
    transition_type: str | None = None


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
    """PostgreSQL-backed store for Buildings, Sections, Rooms, access, and detection matching."""

    def __init__(self, database: Database | str) -> None:
        self.database = ensure_database(database)
        self._lock = threading.RLock()
        self._init_db()
        # Tracks entry/exit state: key=(camera_id, track_id, room_id) -> is_inside
        self._entry_state: dict[tuple[str, int, int], bool] = {}
        self._room_polygon_cache: OrderedDict[
            int, tuple[tuple[float, float], ...]
        ] = OrderedDict()
        self._camera_polygon_cache: OrderedDict[
            int, tuple[tuple[int, tuple[tuple[float, float], ...]], ...]
        ] = OrderedDict()

    def _connection(self) -> Connection:
        return self.database.connection()

    def _init_db(self) -> None:
        # Alembic owns the PostgreSQL schema; runtime startup validates it.
        return None


    def _now(self) -> str:
        return utc_now_text()

    @staticmethod
    def _freeze_polygon(
        polygon: list[list[float]],
    ) -> tuple[tuple[float, float], ...]:
        return tuple((float(point[0]), float(point[1])) for point in polygon)

    @staticmethod
    def _thaw_polygon(
        polygon: tuple[tuple[float, float], ...],
    ) -> list[list[float]]:
        return [[x, y] for x, y in polygon]

    @staticmethod
    def _cache_put(cache: OrderedDict, key: int, value: Any) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > _POLYGON_CACHE_MAX_ENTRIES:
            cache.popitem(last=False)

    def _invalidate_polygon_cache(self) -> None:
        self._room_polygon_cache.clear()
        self._camera_polygon_cache.clear()

    # ── Buildings ─────────────────────────────────────────────────────

    @staticmethod
    def _row_to_building(row: Row) -> BuildingRecord:
        return BuildingRecord(
            id=row["id"],
            name=row["name"],
            address=row["address"],
            description=row["description"],
            created_at_utc=row["created_at_utc"],
            updated_at_utc=row["updated_at_utc"],
            created_by=row.get("created_by"),
            updated_by=row.get("updated_by"),
        )

    def create_building(
        self, name: str, address: str | None = None, description: str | None = None,
        created_by: int | None = None,
    ) -> BuildingRecord:
        name = name.strip()
        if not name:
            raise ValueError("نام ساختمان الزامی است")
        now = self._now()
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "INSERT INTO buildings (name, address, description, created_at_utc, updated_at_utc, created_by) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (name, address, description, now, now, created_by),
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
        updated_by: int | None = None,
    ) -> BuildingRecord | None:
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                "SELECT * FROM buildings WHERE id = ?", (building_id,)
            ).fetchone()
            if existing is None:
                return None
            new_name = name.strip() if name else existing["name"]
            if name is not None and not new_name:
                raise ValueError("نام ساختمان نمی‌تواند خالی باشد")
            new_address = address if address is not None else existing["address"]
            new_description = description if description is not None else existing["description"]
            now = self._now()
            conn.execute(
                "UPDATE buildings SET name=?, address=?, description=?, updated_at_utc=?, updated_by=? WHERE id=?",
                (new_name, new_address, new_description, now, updated_by, building_id),
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
        self, offset: int = 0, limit: int = 50, search: str | None = None,
        allowed_ids: set[int] | None = None,
    ) -> tuple[list[BuildingRecord], int]:
        where = ""
        params: list[Any] = []
        clauses: list[str] = []
        if allowed_ids is not None:
            if not allowed_ids:
                return [], 0
            clauses.append("id IN (" + ", ".join("?" for _ in allowed_ids) + ")")
            params.extend(sorted(allowed_ids))
        if search is not None:
            clauses.append("(name LIKE ? OR description LIKE ?)")
            pattern = f"%{search}%"
            params.extend([pattern, pattern])
        if clauses:
            where = " WHERE " + " AND ".join(clauses)
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
    def _row_to_section(row: Row) -> SectionRecord:
        return SectionRecord(
            id=row["id"],
            building_id=row["building_id"],
            name=row["name"],
            description=row["description"],
            created_at_utc=row["created_at_utc"],
            updated_at_utc=row["updated_at_utc"],
            created_by=row.get("created_by"),
            updated_by=row.get("updated_by"),
            floor=row.get("floor"),
            is_active=bool(row.get("is_active", 1)),
        )

    def create_section(
        self,
        name: str,
        building_id: int | None = None,
        description: str | None = None,
        floor: str | None = None,
        is_active: bool = True,
        created_by: int | None = None,
    ) -> SectionRecord:
        name = name.strip()
        if not name:
            raise ValueError("نام بخش الزامی است")
        now = self._now()
        with self._lock, self._connection() as conn:
            if building_id is not None:
                bld = conn.execute(
                    "SELECT id FROM buildings WHERE id = ?", (building_id,)
                ).fetchone()
                if bld is None:
                    raise ValueError(f"Building not found: {building_id}")
            cursor = conn.execute(
                "INSERT INTO sections (building_id, name, description, floor, is_active, created_at_utc, updated_at_utc, created_by) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (building_id, name, description, floor, int(is_active), now, now, created_by),
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
        floor: str | None = None,
        is_active: bool | None = None,
        updated_by: int | None = None,
    ) -> SectionRecord | None:
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                "SELECT * FROM sections WHERE id = ?", (section_id,)
            ).fetchone()
            if existing is None:
                return None
            new_name = name.strip() if name else existing["name"]
            if name is not None and not new_name:
                raise ValueError("نام بخش نمی‌تواند خالی باشد")
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
            new_floor = floor if floor is not None else existing.get("floor")
            new_is_active = int(is_active) if is_active is not None else existing.get("is_active", 1)
            now = self._now()
            conn.execute(
                "UPDATE sections SET name=?, building_id=?, description=?, floor=?, is_active=?, updated_at_utc=?, updated_by=? WHERE id=?",
                (new_name, new_building_id, new_description, new_floor, new_is_active, now, updated_by, section_id),
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
        allowed_ids: set[int] | None = None,
    ) -> tuple[list[SectionRecord], int]:
        where_clauses: list[str] = []
        params: list[Any] = []
        if allowed_ids is not None:
            if not allowed_ids:
                return [], 0
            where_clauses.append("id IN (" + ", ".join("?" for _ in allowed_ids) + ")")
            params.extend(sorted(allowed_ids))
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

    def list_sections_for_buildings(
        self, building_ids: set[int]
    ) -> dict[int, list[SectionRecord]]:
        ids = sorted({int(value) for value in building_ids})
        if not ids:
            return {}
        placeholders = ", ".join("?" for _ in ids)
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM sections "
                f"WHERE building_id IN ({placeholders}) "
                "ORDER BY building_id ASC, id DESC",
                ids,
            ).fetchall()
        grouped: dict[int, list[SectionRecord]] = {building_id: [] for building_id in ids}
        for row in rows:
            record = self._row_to_section(row)
            if record.building_id is not None:
                grouped.setdefault(record.building_id, []).append(record)
        return grouped

    def get_building_names_by_ids(self, building_ids: set[int]) -> dict[int, str]:
        ids = sorted({int(value) for value in building_ids})
        if not ids:
            return {}
        placeholders = ", ".join("?" for _ in ids)
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                f"SELECT id, name FROM buildings WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        return {int(row["id"]): str(row["name"] or "") for row in rows}

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
    def _row_to_room(row: Row) -> RoomRecord:
        return RoomRecord(
            id=row["id"],
            section_id=row["section_id"],
            cam_id=row["cam_id"],
            name=row["name"],
            room_number=row["room_number"],
            room_type=row["room_type"],
            description=row["description"],
            is_active=bool(row["is_active"]),
            polygon_json=row["polygon_json"],
            created_at_utc=row["created_at_utc"],
            updated_at_utc=row["updated_at_utc"],
            created_by=row.get("created_by"),
            updated_by=row.get("updated_by"),
        )

    def create_room(
        self,
        name: str,
        section_id: int | None = None,
        cam_id: int | None = None,
        room_number: str | None = None,
        room_type: str | None = None,
        description: str | None = None,
        is_active: bool = True,
        polygon_json: str | None = None,
        created_by: int | None = None,
    ) -> RoomRecord:
        name = name.strip()
        if not name:
            raise ValueError("نام اتاق الزامی است")
        if polygon_json:
            points = parse_polygon(polygon_json)
            if len(points) < 3:
                raise ValueError("چندضلعی باید حداقل ۳ رأس داشته باشد")
        now = self._now()
        with self._lock, self._connection() as conn:
            effective_section_id = section_id
            if cam_id is not None:
                cam = conn.execute(
                    "SELECT id, section_id FROM cam WHERE id = ?", (cam_id,)
                ).fetchone()
                if cam is None:
                    raise ValueError(f"Cam not found: {cam_id}")
                cam_section_id = int(cam["section_id"])
                if section_id is not None and section_id != cam_section_id:
                    raise ValueError(
                        f"Cam {cam_id} does not belong to section {section_id}"
                    )
                effective_section_id = cam_section_id
            elif section_id is not None:
                # Legacy/internal callers may still create unassigned rooms by section.
                sec = conn.execute(
                    "SELECT id FROM sections WHERE id = ?", (section_id,)
                ).fetchone()
                if sec is None:
                    raise ValueError(f"Section not found: {section_id}")

            cursor = conn.execute(
                "INSERT INTO rooms (section_id, cam_id, name, room_number, room_type, description, is_active, polygon_json, created_at_utc, updated_at_utc, created_by) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (effective_section_id, cam_id, name, room_number, room_type, description, int(is_active), polygon_json, now, now, created_by),
            )
            row = conn.execute(
                "SELECT * FROM rooms WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            if row is None:
                raise RuntimeError("Failed to retrieve created room")
            self._invalidate_polygon_cache()
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
        cam_id: int | None | object = _UNSET,
        room_number: str | None | object = _UNSET,
        room_type: str | None | object = _UNSET,
        description: str | None | object = _UNSET,
        is_active: bool | None = None,
        polygon_json: str | None | object = _UNSET,
        updated_by: int | None = None,
    ) -> RoomRecord | None:
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                "SELECT * FROM rooms WHERE id = ?", (room_id,)
            ).fetchone()
            if existing is None:
                return None
            new_name = name.strip() if name else existing["name"]
            if name is not None and not new_name:
                raise ValueError("نام اتاق نمی‌تواند خالی باشد")

            new_cam_id = existing["cam_id"] if cam_id is _UNSET else cam_id
            new_section_id = existing["section_id"]
            if cam_id is not _UNSET:
                if cam_id is None:
                    new_section_id = None
                else:
                    cam = conn.execute(
                        "SELECT id, section_id FROM cam WHERE id = ?", (cam_id,)
                    ).fetchone()
                    if cam is None:
                        raise ValueError(f"Cam not found: {cam_id}")
                    new_section_id = int(cam["section_id"])
            elif section_id is not None:
                # Legacy/internal update path for unassigned rooms.
                sec = conn.execute(
                    "SELECT id FROM sections WHERE id = ?", (section_id,)
                ).fetchone()
                if sec is None:
                    raise ValueError(f"Section not found: {section_id}")
                if new_cam_id is not None:
                    cam = conn.execute(
                        "SELECT section_id FROM cam WHERE id = ?", (new_cam_id,)
                    ).fetchone()
                    if cam is None or int(cam["section_id"]) != section_id:
                        raise ValueError(
                            "Room section must match its assigned cam section"
                        )
                new_section_id = section_id

            new_description = existing["description"] if description is _UNSET else description
            new_room_number = existing["room_number"] if room_number is _UNSET else room_number
            new_room_type = existing["room_type"] if room_type is _UNSET else room_type
            new_is_active = int(is_active) if is_active is not None else existing["is_active"]
            new_polygon = existing["polygon_json"] if polygon_json is _UNSET else polygon_json
            if polygon_json is not _UNSET and polygon_json is not None:
                points = parse_polygon(polygon_json)
                if len(points) < 3:
                    raise ValueError("چندضلعی باید حداقل ۳ رأس داشته باشد")
            now = self._now()
            conn.execute(
                "UPDATE rooms SET name=?, section_id=?, cam_id=?, room_number=?, room_type=?, description=?, is_active=?, polygon_json=?, "
                "updated_at_utc=?, updated_by=? WHERE id=?",
                (new_name, new_section_id, new_cam_id, new_room_number, new_room_type, new_description, new_is_active, new_polygon, now, updated_by, room_id),
            )
            row = conn.execute(
                "SELECT * FROM rooms WHERE id = ?", (room_id,)
            ).fetchone()
            self._invalidate_polygon_cache()
            return self._row_to_room(row)

    def delete_room(self, room_id: int) -> bool:
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM rooms WHERE id = ?", (room_id,)
            )
            if cursor.rowcount > 0:
                self._invalidate_polygon_cache()
            return cursor.rowcount > 0

    def list_rooms(
        self,
        offset: int = 0,
        limit: int = 50,
        section_id: int | None = None,
        cam_id: int | None = None,
        search: str | None = None,
    ) -> tuple[list[RoomRecord], int]:
        where_clauses: list[str] = []
        params: list[Any] = []
        if section_id is not None:
            where_clauses.append("section_id = ?")
            params.append(section_id)
        if cam_id is not None:
            where_clauses.append("cam_id = ?")
            params.append(cam_id)
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
                row = conn.execute(
                    "INSERT INTO personnel_room_access (personnel_id, room_id, granted_at_utc, granted_by) "
                    "VALUES (?, ?, ?, ?) RETURNING *",
                    (personnel_id, room_id, now, granted_by),
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
            except IntegrityError:
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

    def resolve_access_for_pairs(
        self, pairs: set[tuple[int, int]] | list[tuple[int, int]]
    ) -> dict[tuple[int, int], bool]:
        """Resolve many personnel/room access checks with bounded queries."""
        unique_pairs = sorted({(int(pid), int(rid)) for pid, rid in pairs})
        if not unique_pairs:
            return {}
        room_ids = sorted({room_id for _, room_id in unique_pairs})
        room_rows: list[Row] = []
        granted_rows: list[Row] = []
        with self._lock, self._connection() as conn:
            for start in range(0, len(room_ids), 1000):
                room_batch = room_ids[start:start + 1000]
                room_placeholders = ", ".join("?" for _ in room_batch)
                room_rows.extend(
                    conn.execute(
                        f"SELECT id, name FROM rooms WHERE id IN ({room_placeholders})",
                        room_batch,
                    ).fetchall()
                )
            for start in range(0, len(unique_pairs), 1000):
                pair_batch = unique_pairs[start:start + 1000]
                values_sql = ", ".join("(?, ?)" for _ in pair_batch)
                pair_params = [value for pair in pair_batch for value in pair]
                granted_rows.extend(
                    conn.execute(
                        "WITH requested(personnel_id, room_id) AS (VALUES "
                        + values_sql
                        + ") SELECT a.personnel_id, a.room_id "
                        "FROM personnel_room_access a JOIN requested r "
                        "ON r.personnel_id = a.personnel_id AND r.room_id = a.room_id",
                        pair_params,
                    ).fetchall()
                )
        general_rooms = {
            int(row["id"])
            for row in room_rows
            if "general" in str(row.get("name") or "").lower()
            or "عمومی" in str(row.get("name") or "").lower()
        }
        granted = {
            (int(row["personnel_id"]), int(row["room_id"]))
            for row in granted_rows
        }
        return {
            pair: pair[1] in general_rooms or pair in granted
            for pair in unique_pairs
        }

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

    @staticmethod
    def _human_foot_point(bbox: list[float]) -> tuple[float, float]:
        """Compute the foot point (center bottom) of a human bounding box.

        The foot point is defined as ((x1 + x2) / 2, y2) where
        bbox = [x1, y1, x2, y2] in pixel coordinates.
        """
        if not bbox or len(bbox) < 4:
            return (0.0, 0.0)
        return ((float(bbox[0]) + float(bbox[2])) / 2.0, float(bbox[3]))

    def _resolve_transition(
        self,
        camera_id: str | None,
        track_id: int | None,
        room_id: int,
        is_inside: bool,
    ) -> str | None:
        """Determine entry/exit transition based on previous state.

        Returns 'entered', 'exited', or None if no transition.
        """
        if track_id is None or camera_id is None:
            # Without track_id we cannot track transitions
            return "entered" if is_inside else None
        key = (camera_id, track_id, room_id)
        was_inside = self._entry_state.get(key, False)
        if is_inside and not was_inside:
            self._entry_state[key] = True
            return "entered"
        elif not is_inside and was_inside:
            self._entry_state[key] = False
            return "exited"
        return None

    def match_detection_to_rooms(
        self,
        section_id: int,
        detection_type: str,
        detection_event_id: int,
        bbox_center_x: float,
        bbox_center_y: float,
        personnel_id: int | None = None,
        camera_id: str | None = None,
        *,
        track_id: int | None = None,
    ) -> list[DetectionRoomMatchRecord]:
        """Match a detection point to rooms via polygon containment.

        Checks every room in the given section for polygon containment
        of the specified point. The caller is responsible for resolving
        the section_id from the camera/source.

        When track_id is provided, entry/exit transition tracking is
        enabled: each result includes a transition_type field set to
        'entered', 'exited', or None.

        Returns a list of DetectionRoomMatchRecord for each matched room.
        """
        if section_id is None:
            return []
        if personnel_id is not None:
            try:
                personnel_id = int(personnel_id)
            except (TypeError, ValueError):
                personnel_id = None
        with self._lock, self._connection() as conn:
            rooms = conn.execute(
                "SELECT id, polygon_json FROM rooms "
                "WHERE section_id = ? AND is_active = 1 AND polygon_json IS NOT NULL",
                (section_id,),
            ).fetchall()
            now = self._now()
            values: list[tuple[Any, ...]] = []
            for room in rooms:
                polygon = parse_polygon(room["polygon_json"])
                if len(polygon) < 3:
                    continue
                is_inside = point_in_polygon(bbox_center_x, bbox_center_y, polygon)
                transition = self._resolve_transition(
                    camera_id=camera_id,
                    track_id=track_id,
                    room_id=room["id"],
                    is_inside=is_inside,
                )
                if not is_inside and transition is None:
                    continue
                values.append(
                    (
                        detection_type,
                        detection_event_id,
                        room["id"],
                        personnel_id,
                        camera_id,
                        track_id,
                        transition,
                        now,
                    )
                )
            if not values:
                return []
            value_sql = "(" + ", ".join("?" for _ in range(8)) + ")"
            rows = conn.execute(
                "INSERT INTO detection_room_matches "
                "(detection_type, detection_event_id, room_id, personnel_id, "
                "camera_id, track_id, transition_type, matched_at_utc) VALUES "
                + ", ".join(value_sql for _ in values)
                + " RETURNING id, detection_type, detection_event_id, room_id, "
                "personnel_id, camera_id, track_id, transition_type, matched_at_utc",
                [value for row in values for value in row],
            ).fetchall()
            return [
                DetectionRoomMatchRecord(
                    id=row["id"],
                    detection_type=row["detection_type"],
                    detection_event_id=row["detection_event_id"],
                    room_id=row["room_id"],
                    personnel_id=row["personnel_id"],
                    camera_id=row["camera_id"],
                    track_id=row["track_id"],
                    transition_type=row["transition_type"],
                    matched_at_utc=row["matched_at_utc"],
                )
                for row in rows
            ]

    def room_has_polygon(self, room_id: int | None) -> bool:
        return bool(self.get_polygon_for_room(room_id))

    def get_polygon_for_room(self, room_id: int | None) -> list[list[float]]:
        if room_id is None:
            return []
        with self._lock:
            cached = self._room_polygon_cache.get(room_id)
            if cached is not None:
                self._room_polygon_cache.move_to_end(room_id)
                return self._thaw_polygon(cached)
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT polygon_json FROM rooms WHERE id = ?", (room_id,)
            ).fetchone()
        points = parse_polygon(row["polygon_json"]) if row and row["polygon_json"] else []
        valid = points if len(points) >= 3 else []
        with self._lock:
            self._cache_put(
                self._room_polygon_cache,
                room_id,
                self._freeze_polygon(valid),
            )
        return valid

    def get_camera_polygon_rooms_for_room(
        self, room_id: int | None
    ) -> list[tuple[int, list[list[float]]]]:
        """Return every active polygon owned by the assigned room's camera.

        ``sources.room_id`` is singular, while ``rooms.cam_id`` intentionally
        permits a camera to own multiple rooms/zones.  Rooms without a camera
        link retain the legacy single-room behavior.
        """
        if room_id is None:
            return []
        with self._lock:
            cached = self._camera_polygon_cache.get(room_id)
            if cached is not None:
                self._camera_polygon_cache.move_to_end(room_id)
                return [
                    (cached_room_id, self._thaw_polygon(polygon))
                    for cached_room_id, polygon in cached
                ]
        with self._lock, self._connection() as conn:
            assigned = conn.execute(
                "SELECT cam_id, is_active, polygon_json FROM rooms WHERE id = ?",
                (room_id,),
            ).fetchone()
            if assigned is None or not bool(assigned["is_active"]):
                return []
            if assigned["cam_id"] is None:
                rows = [{"id": room_id, "polygon_json": assigned["polygon_json"]}]
            else:
                rows = conn.execute(
                    "SELECT id, polygon_json FROM rooms "
                    "WHERE cam_id = ? AND is_active = 1 ORDER BY id ASC",
                    (assigned["cam_id"],),
                ).fetchall()

        polygon_rooms: list[tuple[int, list[list[float]]]] = []
        for row in rows:
            points = parse_polygon(row["polygon_json"]) if row["polygon_json"] else []
            if len(points) >= 3:
                polygon_rooms.append((int(row["id"]), points))
        with self._lock:
            frozen = tuple(
                (cached_room_id, self._freeze_polygon(polygon))
                for cached_room_id, polygon in polygon_rooms
            )
            self._cache_put(self._camera_polygon_cache, room_id, frozen)
            for cached_room_id, polygon in frozen:
                self._cache_put(self._room_polygon_cache, cached_room_id, polygon)
        return polygon_rooms

    def get_camera_polygons_for_room(self, room_id: int | None) -> list[list[list[float]]]:
        return [
            polygon
            for _, polygon in self.get_camera_polygon_rooms_for_room(room_id)
        ]

    def match_detection_to_camera_rooms(
        self,
        anchor_room_id: int,
        detection_type: str,
        detection_event_id: int,
        bbox_center_x: float,
        bbox_center_y: float,
        personnel_id: int | None = None,
        camera_id: str | None = None,
        *,
        track_id: int | None = None,
    ) -> list[DetectionRoomMatchRecord]:
        """Match one detection independently against every zone of its camera."""
        if personnel_id is not None:
            try:
                personnel_id = int(personnel_id)
            except (TypeError, ValueError):
                personnel_id = None
        now = self._now()
        values: list[tuple[Any, ...]] = []
        for room_id, polygon in self.get_camera_polygon_rooms_for_room(anchor_room_id):
            is_inside = point_in_polygon(bbox_center_x, bbox_center_y, polygon)
            transition = self._resolve_transition(
                camera_id, track_id, room_id, is_inside
            )
            if not is_inside and transition is None:
                continue
            values.append(
                (
                    detection_type,
                    detection_event_id,
                    room_id,
                    personnel_id,
                    camera_id,
                    track_id,
                    transition,
                    now,
                )
            )
        if not values:
            return []
        value_sql = "(" + ", ".join("?" for _ in range(8)) + ")"
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "INSERT INTO detection_room_matches "
                "(detection_type, detection_event_id, room_id, personnel_id, "
                "camera_id, track_id, transition_type, matched_at_utc) VALUES "
                + ", ".join(value_sql for _ in values)
                + " RETURNING id, detection_type, detection_event_id, room_id, "
                "personnel_id, camera_id, track_id, transition_type, matched_at_utc",
                [value for row in values for value in row],
            ).fetchall()
        return [
            DetectionRoomMatchRecord(
                id=row["id"],
                detection_type=row["detection_type"],
                detection_event_id=row["detection_event_id"],
                room_id=row["room_id"],
                personnel_id=row["personnel_id"],
                camera_id=row["camera_id"],
                track_id=row["track_id"],
                transition_type=row["transition_type"],
                matched_at_utc=row["matched_at_utc"],
            )
            for row in rows
        ]

    def match_detection_to_room(
        self,
        room_id: int,
        detection_type: str,
        detection_event_id: int,
        bbox_center_x: float,
        bbox_center_y: float,
        personnel_id: int | None = None,
        camera_id: str | None = None,
        *,
        track_id: int | None = None,
        polygon: list[list[float]] | None = None,
    ) -> list[DetectionRoomMatchRecord]:
        polygon = polygon if polygon is not None else self.get_polygon_for_room(room_id)
        if not polygon:
            return []
        if personnel_id is not None:
            try:
                personnel_id = int(personnel_id)
            except (TypeError, ValueError):
                personnel_id = None
        is_inside = point_in_polygon(bbox_center_x, bbox_center_y, polygon)
        transition = self._resolve_transition(camera_id, track_id, room_id, is_inside)
        if not is_inside and transition is None:
            return []
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "INSERT INTO detection_room_matches "
                "(detection_type, detection_event_id, room_id, personnel_id, camera_id, track_id, transition_type, matched_at_utc) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "RETURNING id, detection_type, detection_event_id, room_id, personnel_id, camera_id, track_id, transition_type, matched_at_utc",
                (detection_type, detection_event_id, room_id, personnel_id, camera_id,
                 track_id, transition, self._now()),
            ).fetchone()
        if row is None:
            return []
        return [DetectionRoomMatchRecord(
            id=row["id"], detection_type=row["detection_type"],
            detection_event_id=row["detection_event_id"], room_id=row["room_id"],
            personnel_id=row["personnel_id"], camera_id=row["camera_id"],
            track_id=row["track_id"], transition_type=row["transition_type"],
            matched_at_utc=row["matched_at_utc"],
        )]

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
                    track_id=row["track_id"],
                    transition_type=row["transition_type"],
                    matched_at_utc=row["matched_at_utc"],
                )
                for row in rows
            ]

    def list_matches_for_room(
        self, room_id: int, limit: int = 50, offset: int = 0,
        transition_type: str | None = None,
    ) -> tuple[list[DetectionRoomMatchRecord], int]:
        where_clause = "WHERE room_id = ?"
        params: list[Any] = [room_id]
        if transition_type is not None:
            where_clause += " AND transition_type = ?"
            params.append(transition_type)
        with self._lock, self._connection() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM detection_room_matches {where_clause}",
                params,
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM detection_room_matches {where_clause} "
                "ORDER BY matched_at_utc DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            records = [
                DetectionRoomMatchRecord(
                    id=row["id"],
                    detection_type=row["detection_type"],
                    detection_event_id=row["detection_event_id"],
                    room_id=row["room_id"],
                    personnel_id=row["personnel_id"],
                    camera_id=row["camera_id"],
                    track_id=row["track_id"],
                    transition_type=row["transition_type"],
                    matched_at_utc=row["matched_at_utc"],
                )
                for row in rows
            ]
            return records, int(total)

    # ── Polygon zone helpers ──────────────────────────────────────────

    def section_has_polygons(self, section_id: int) -> bool:
        """Check if any rooms in this section have polygon zones defined."""
        if section_id is None:
            return False
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM rooms WHERE section_id = ? AND polygon_json IS NOT NULL",
                (section_id,),
            ).fetchone()
            count = int(row[0]) if row else 0
            # Also check if any polygon_json is valid (non-empty)
            if count == 0:
                return False
            # Verify at least one polygon has valid data
            rows = conn.execute(
                "SELECT polygon_json FROM rooms WHERE section_id = ? AND polygon_json IS NOT NULL LIMIT 1",
                (section_id,),
            ).fetchall()
            for r in rows:
                points = parse_polygon(r["polygon_json"])
                if len(points) >= 3:
                    return True
            return False

    def get_polygons_for_section(self, section_id: int) -> list[list[list[float]]]:
        """Return all valid polygon point lists for rooms in a section.

        Each entry is a polygon represented as [[x, y], [x, y], ...] with at
        least 3 points. Returns empty list if none exist or section_id is None.
        """
        if section_id is None:
            return []
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT polygon_json FROM rooms WHERE section_id = ? AND polygon_json IS NOT NULL",
                (section_id,),
            ).fetchall()
            result: list[list[list[float]]] = []
            for r in rows:
                points = parse_polygon(r["polygon_json"])
                if len(points) >= 3:
                    result.append(points)
            return result

    def get_default_polygon_entry_state(
        self, camera_id: str, track_id: int, foot_x: float, foot_y: float
    ) -> str | None:
        """Check transition against the default full-frame polygon.

        Returns 'entered', 'exited', or None if no transition.
        This does NOT insert into the database - it only tracks state
        in-memory for the default polygon case.
        """
        key = (camera_id, track_id, _DEFAULT_ROOM_ID)
        was_inside = self._entry_state.get(key, False)
        is_inside = point_in_polygon(foot_x, foot_y, DEFAULT_POLYGON)
        if is_inside and not was_inside:
            self._entry_state[key] = True
            return "entered"
        elif not is_inside and was_inside:
            self._entry_state[key] = False
            return "exited"
        return None

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
