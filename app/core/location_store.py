from __future__ import annotations

from app.database import Connection, Database, IntegrityError, OperationalError, Row, ensure_database
from app.time_utils import utc_now_text

import json
import logging
import threading
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


@dataclass(frozen=True, slots=True)
class RoomRecord:
    id: int
    section_id: int | None
    cam_id: int | None
    name: str
    description: str | None
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

    def _connection(self) -> Connection:
        return self.database.connection()

    def _init_db(self) -> None:
        # Alembic owns the PostgreSQL schema; runtime startup validates it.
        return None


    def _now(self) -> str:
        return utc_now_text()

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
            raise ValueError("Building name is required")
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
                raise ValueError("Building name cannot be blank")
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
        )

    def create_section(
        self,
        name: str,
        building_id: int | None = None,
        description: str | None = None,
        created_by: int | None = None,
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
                "INSERT INTO sections (building_id, name, description, created_at_utc, updated_at_utc, created_by) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (building_id, name, description, now, now, created_by),
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
                "UPDATE sections SET name=?, building_id=?, description=?, updated_at_utc=?, updated_by=? WHERE id=?",
                (new_name, new_building_id, new_description, now, updated_by, section_id),
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
    def _row_to_room(row: Row) -> RoomRecord:
        return RoomRecord(
            id=row["id"],
            section_id=row["section_id"],
            cam_id=row["cam_id"],
            name=row["name"],
            description=row["description"],
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
        description: str | None = None,
        polygon_json: str | None = None,
        created_by: int | None = None,
    ) -> RoomRecord:
        name = name.strip()
        if not name:
            raise ValueError("Room name is required")
        if polygon_json:
            points = parse_polygon(polygon_json)
            if len(points) < 3:
                raise ValueError("Polygon must have at least 3 vertices")
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
                "INSERT INTO rooms (section_id, cam_id, name, description, polygon_json, created_at_utc, updated_at_utc, created_by) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (effective_section_id, cam_id, name, description, polygon_json, now, now, created_by),
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
        cam_id: int | None | object = _UNSET,
        description: str | None = None,
        polygon_json: str | None = None,
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
                raise ValueError("Room name cannot be blank")

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

            new_description = description if description is not None else existing["description"]
            new_polygon = polygon_json if polygon_json is not None else existing["polygon_json"]
            if polygon_json is not None:
                points = parse_polygon(polygon_json)
                if len(points) < 3:
                    raise ValueError("Polygon must have at least 3 vertices")
            now = self._now()
            conn.execute(
                "UPDATE rooms SET name=?, section_id=?, cam_id=?, description=?, polygon_json=?, "
                "updated_at_utc=?, updated_by=? WHERE id=?",
                (new_name, new_section_id, new_cam_id, new_description, new_polygon, now, updated_by, room_id),
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
                "SELECT id, polygon_json FROM rooms WHERE section_id = ? AND polygon_json IS NOT NULL",
                (section_id,),
            ).fetchall()

            matched: list[DetectionRoomMatchRecord] = []
            now = self._now()

            for room in rooms:
                polygon = parse_polygon(room["polygon_json"])
                if len(polygon) < 3:
                    continue
                is_inside = point_in_polygon(bbox_center_x, bbox_center_y, polygon)
                # Determine transition type
                transition = self._resolve_transition(
                    camera_id=camera_id,
                    track_id=track_id,
                    room_id=room["id"],
                    is_inside=is_inside,
                )
                if not is_inside and transition is None:
                    # Point is outside and no transition (was outside before)
                    continue
                if is_inside and transition is None and track_id is not None:
                    # Still inside - record as heartbeat (no transition)
                    pass
                cursor = conn.execute(
                    "INSERT INTO detection_room_matches "
                    "(detection_type, detection_event_id, room_id, personnel_id, camera_id, track_id, transition_type, matched_at_utc) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (detection_type, detection_event_id, room["id"], personnel_id,
                     camera_id, track_id, transition, now),
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
                        track_id=row["track_id"],
                        transition_type=row["transition_type"],
                        matched_at_utc=row["matched_at_utc"],
                    ))
            return matched

    def room_has_polygon(self, room_id: int | None) -> bool:
        return bool(self.get_polygon_for_room(room_id))

    def get_polygon_for_room(self, room_id: int | None) -> list[list[float]]:
        if room_id is None:
            return []
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT polygon_json FROM rooms WHERE id = ?", (room_id,)
            ).fetchone()
        points = parse_polygon(row["polygon_json"]) if row and row["polygon_json"] else []
        return points if len(points) >= 3 else []

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
    ) -> list[DetectionRoomMatchRecord]:
        polygon = self.get_polygon_for_room(room_id)
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
            cursor = conn.execute(
                "INSERT INTO detection_room_matches "
                "(detection_type, detection_event_id, room_id, personnel_id, camera_id, track_id, transition_type, matched_at_utc) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (detection_type, detection_event_id, room_id, personnel_id, camera_id,
                 track_id, transition, self._now()),
            )
            row = conn.execute(
                "SELECT * FROM detection_room_matches WHERE id = ?", (cursor.lastrowid,)
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
