from __future__ import annotations

import json
import logging
import threading
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable

from app.core.types import TaskName
from app.database import Database, Row, ensure_database
from app.time_utils import utc_now_text


LOGGER = logging.getLogger(__name__)


def _utc_now() -> str:
    return utc_now_text()


RTSP = "rtsp"
STATIC_VIDEO = "static_video"
SOURCE_TYPES = frozenset({RTSP, STATIC_VIDEO})


@dataclass(slots=True)
class SourceRecord:
    id: int | None = None
    source_uri: str = ""
    name: str = ""
    enabled: bool = True
    tasks: set[TaskName] = field(default_factory=set)
    frame_width: int = 640
    frame_height: int = 640
    source_type: str = RTSP
    room_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    fps: float | None = None
    loop: bool = True
    draw_human: bool = True
    draw_zone: bool = True
    draw_fire: bool = True
    draw_smoke: bool = True
    draw_vehicle: bool = True
    draw_plate: bool = True
    created_at_utc: str = field(default_factory=_utc_now)
    updated_at_utc: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["tasks"] = sorted(task.value for task in self.tasks)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceRecord":
        source_uri = value.get("source_uri", value.get("camera_id", value.get("source_id")))
        if source_uri is None:
            raise ValueError("Source record requires source_uri")
        source_type = value.get("source_type", RTSP)
        if source_type not in SOURCE_TYPES:
            source_type = RTSP
        return cls(
            id=value.get("id"),
            source_uri=str(source_uri),
            name=str(value.get("name") or source_uri),
            enabled=bool(value.get("enabled", True)),
            tasks={TaskName(item) for item in value.get("tasks", [])},
            frame_width=int(value.get("frame_width", 640)),
            frame_height=int(value.get("frame_height", 640)),
            source_type=source_type,
            room_id=int(value["room_id"]) if value.get("room_id") is not None else None,
            metadata=dict(value.get("metadata") or {}),
            fps=float(value["fps"]) if value.get("fps") is not None else None,
            loop=bool(value.get("loop", True)),
            draw_human=bool(value.get("draw_human", True)),
            draw_zone=bool(value.get("draw_zone", True)),
            draw_fire=bool(value.get("draw_fire", True)),
            draw_smoke=bool(value.get("draw_smoke", True)),
            draw_vehicle=bool(value.get("draw_vehicle", True)),
            draw_plate=bool(value.get("draw_plate", True)),
            created_at_utc=str(value.get("created_at_utc") or _utc_now()),
            updated_at_utc=str(value.get("updated_at_utc") or _utc_now()),
        )


@dataclass(frozen=True, slots=True)
class SourceChange:
    action: str
    source_uri: str
    revision: int
    record: SourceRecord | None
    previous_source_uri: str | None = None


SourceChangeListener = Callable[[SourceChange], None]


class SourceRegistry:
    """Thread-safe source registry backed by PostgreSQL."""

    def __init__(self, database: Database | str) -> None:
        self._lock = threading.RLock()
        self.database = ensure_database(database)
        self._connection = self.database.connection()
        self._records: dict[str, SourceRecord] = {}
        self._listeners: set[SourceChangeListener] = set()
        self._revision = 0
        self._closed = False
        self._initialize()
        self._load_records()

    def _initialize(self) -> None:
        # Alembic owns the PostgreSQL schema; runtime startup validates it.
        return None

    def _load_records(self) -> None:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT id, source_uri, name, enabled, tasks_json,
                       frame_width, frame_height, room_id, source_type,
                       metadata_json, fps, loop, draw_human, draw_zone, draw_fire,
                       draw_smoke, draw_vehicle, draw_plate,
                       created_at_utc, updated_at_utc
                FROM sources
                WHERE name IS NOT NULL
                ORDER BY COALESCE(created_at_utc, updated_at_utc), source_uri
                """
            ).fetchall()
            self._records = {
                str(row["source_uri"]): self._row_to_record(row) for row in rows
            }

    def _next_id(self) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(id), 0) + 1 AS next_id FROM sources WHERE id IS NOT NULL"
        ).fetchone()
        return int(row["next_id"]) if row is not None else 1

    @staticmethod
    def _normalized(record: SourceRecord) -> SourceRecord:
        value = deepcopy(record)
        value.source_uri = value.source_uri.strip()
        value.name = value.name.strip()
        if not value.source_uri:
            raise ValueError("source_uri cannot be blank")
        if not value.name:
            raise ValueError("name cannot be blank")
        value.tasks = {TaskName(task) for task in value.tasks}
        value.metadata = dict(value.metadata)
        value.fps = float(value.fps) if value.fps is not None else None
        value.frame_width = int(value.frame_width)
        value.frame_height = int(value.frame_height)
        if value.fps is not None and not 0 < value.fps <= 240:
            raise ValueError("fps must be between 0 and 240")
        if not 16 <= value.frame_width <= 4096:
            raise ValueError("frame_width must be between 16 and 4096")
        if not 16 <= value.frame_height <= 4096:
            raise ValueError("frame_height must be between 16 and 4096")
        if value.source_type not in SOURCE_TYPES:
            value.source_type = RTSP
        return value

    @staticmethod
    def _row_to_record(row: Row) -> SourceRecord:
        metadata = dict(json.loads(row["metadata_json"]))
        return SourceRecord(
            id=int(row["id"]),
            source_uri=str(row["source_uri"]),
            name=str(row["name"]),
            enabled=bool(row["enabled"]),
            tasks={TaskName(item) for item in json.loads(row["tasks_json"])},
            frame_width=int(row["frame_width"]),
            frame_height=int(row["frame_height"]),
            source_type=str(row["source_type"]) if row["source_type"] else RTSP,
            room_id=int(row["room_id"]) if row["room_id"] is not None else None,
            metadata=metadata,
            fps=float(row["fps"]) if row["fps"] is not None else None,
            loop=bool(row["loop"]) if row["loop"] is not None else True,
            draw_human=bool(row["draw_human"]) if row["draw_human"] is not None else True,
            draw_zone=bool(row["draw_zone"]) if row["draw_zone"] is not None else True,
            draw_fire=bool(row["draw_fire"]) if row["draw_fire"] is not None else True,
            draw_smoke=bool(row["draw_smoke"]) if row["draw_smoke"] is not None else True,
            draw_vehicle=bool(row["draw_vehicle"]) if row["draw_vehicle"] is not None else True,
            draw_plate=bool(row["draw_plate"]) if row["draw_plate"] is not None else True,
            created_at_utc=str(row["created_at_utc"]),
            updated_at_utc=str(row["updated_at_utc"]),
        )

    @staticmethod
    def _parameters(record: SourceRecord) -> tuple[Any, ...]:
        metadata = dict(record.metadata)
        metadata.pop("section_id", None)
        metadata.pop("room_id", None)
        return (
            record.id,
            record.source_uri,
            record.name,
            int(record.enabled),
            json.dumps(sorted(task.value for task in record.tasks)),
            record.frame_width,
            record.frame_height,
            record.room_id,
            record.source_type,
            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            record.fps,
            int(record.loop),
            int(record.draw_human),
            int(record.draw_zone),
            int(record.draw_fire),
            int(record.draw_smoke),
            int(record.draw_vehicle),
            int(record.draw_plate),
            record.created_at_utc,
            record.updated_at_utc,
        )

    def _next_change(
        self,
        action: str,
        source_uri: str,
        record: SourceRecord | None,
        *,
        previous_source_uri: str | None = None,
    ) -> tuple[SourceChange, tuple[SourceChangeListener, ...]]:
        self._revision += 1
        change = SourceChange(
            action=action,
            source_uri=source_uri,
            revision=self._revision,
            record=deepcopy(record),
            previous_source_uri=previous_source_uri,
        )
        return change, tuple(self._listeners)

    @staticmethod
    def _notify(
        change: SourceChange,
        listeners: tuple[SourceChangeListener, ...],
    ) -> None:
        for listener in listeners:
            try:
                listener(change)
            except Exception:
                LOGGER.exception(
                    "Source registry listener failed: action=%s source_uri=%s",
                    change.action,
                    change.source_uri,
                )

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def add_listener(self, listener: SourceChangeListener) -> None:
        with self._lock:
            self._listeners.add(listener)

    def remove_listener(self, listener: SourceChangeListener) -> None:
        with self._lock:
            self._listeners.discard(listener)

    def import_if_empty(self, records: Iterable[SourceRecord]) -> int:
        prepared = [self._normalized(record) for record in records]
        if not prepared:
            return 0
        with self._lock:
            if self._records:
                return 0
            imported_records: dict[str, SourceRecord] = {}
            for record in prepared:
                if record.id is None:
                    record.id = self._next_id()
                self._connection.execute(
                    """
                    INSERT INTO sources (
                        id, source_uri, name, enabled, tasks_json,
                        frame_width, frame_height, room_id, source_type,
                        metadata_json, fps, loop, draw_human, draw_zone, draw_fire,
                        draw_smoke, draw_vehicle, draw_plate,
                        created_at_utc, updated_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_uri) DO UPDATE SET
                        id = COALESCE(sources.id, excluded.id),
                        name = excluded.name,
                        enabled = excluded.enabled,
                        tasks_json = excluded.tasks_json,
                        frame_width = excluded.frame_width,
                        frame_height = excluded.frame_height,
                        room_id = excluded.room_id,
                        source_type = excluded.source_type,
                        metadata_json = excluded.metadata_json,
                        fps = excluded.fps,
                        loop = excluded.loop,
                        draw_human = excluded.draw_human,
                        draw_zone = excluded.draw_zone,
                        draw_fire = excluded.draw_fire,
                        draw_smoke = excluded.draw_smoke,
                        draw_vehicle = excluded.draw_vehicle,
                        draw_plate = excluded.draw_plate,
                        updated_at_utc = excluded.updated_at_utc
                    """,
                    self._parameters(record),
                )
                imported_records[record.source_uri] = deepcopy(record)
            self._connection.commit()
            self._records = imported_records
            self._revision += 1
            return len(prepared)

    def list(self) -> list[SourceRecord]:
        with self._lock:
            return deepcopy(list(self._records.values()))

    def list_by_type(self, source_type: str) -> list[SourceRecord]:
        if source_type not in SOURCE_TYPES:
            raise ValueError(f"Unknown source_type: {source_type}")
        with self._lock:
            return deepcopy([
                r for r in self._records.values() if r.source_type == source_type
            ])

    def active_sources(self) -> list[SourceRecord]:
        """Return all enabled sources, regardless of type."""
        with self._lock:
            return deepcopy([
                r for r in self._records.values() if r.enabled
            ])

    def get(self, source_uri: str) -> SourceRecord | None:
        with self._lock:
            value = self._records.get(source_uri)
            return deepcopy(value) if value is not None else None

    def get_by_id(self, source_id: int) -> SourceRecord | None:
        with self._lock:
            for record in self._records.values():
                if record.id == source_id:
                    return deepcopy(record)
            return None

    def require(self, source_uri: str) -> SourceRecord:
        value = self.get(source_uri)
        if value is None:
            raise KeyError(source_uri)
        return value

    def create(self, record: SourceRecord) -> SourceRecord:
        record = self._normalized(record)
        with self._lock:
            if self._records.get(record.source_uri) is not None:
                raise ValueError(f"Source already exists: {record.source_uri}")
            if record.id is None:
                record.id = self._next_id()
            self._connection.execute(
                """
                INSERT INTO sources (
                    id, source_uri, name, enabled, tasks_json,
                    frame_width, frame_height, room_id, source_type,
                    metadata_json, fps, loop, draw_human, draw_zone, draw_fire,
                    draw_smoke, draw_vehicle, draw_plate,
                    created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_uri) DO UPDATE SET
                    id = COALESCE(sources.id, excluded.id),
                    name = excluded.name,
                    enabled = excluded.enabled,
                    tasks_json = excluded.tasks_json,
                    frame_width = excluded.frame_width,
                    frame_height = excluded.frame_height,
                    room_id = excluded.room_id,
                    source_type = excluded.source_type,
                    metadata_json = excluded.metadata_json,
                    fps = excluded.fps,
                    loop = excluded.loop,
                    draw_human = excluded.draw_human,
                    draw_zone = excluded.draw_zone,
                    draw_fire = excluded.draw_fire,
                    draw_smoke = excluded.draw_smoke,
                    draw_vehicle = excluded.draw_vehicle,
                    draw_plate = excluded.draw_plate,
                    updated_at_utc = excluded.updated_at_utc
                """,
                self._parameters(record),
            )
            self._connection.commit()
            self._records[record.source_uri] = deepcopy(record)
            change, listeners = self._next_change("created", record.source_uri, record)
        self._notify(change, listeners)
        return deepcopy(record)

    def upsert(self, record: SourceRecord) -> SourceRecord:
        record = self._normalized(record)
        with self._lock:
            existing = self._records.get(record.source_uri)
            action = "created" if existing is None else "updated"
            if existing is not None:
                record.created_at_utc = existing.created_at_utc
                record.id = existing.id
            elif record.id is None:
                record.id = self._next_id()
            record.updated_at_utc = _utc_now()
            self._connection.execute(
                """
                INSERT INTO sources (
                    id, source_uri, name, enabled, tasks_json,
                    frame_width, frame_height, room_id, source_type,
                    metadata_json, loop, created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_uri) DO UPDATE SET
                    id = COALESCE(sources.id, excluded.id),
                    name = excluded.name,
                    enabled = excluded.enabled,
                    tasks_json = excluded.tasks_json,
                    frame_width = excluded.frame_width,
                    frame_height = excluded.frame_height,
                    room_id = excluded.room_id,
                    source_type = excluded.source_type,
                    metadata_json = excluded.metadata_json,
                    loop = excluded.loop,
                    updated_at_utc = excluded.updated_at_utc
                """,
                self._parameters(record),
            )
            self._connection.commit()
            self._records[record.source_uri] = deepcopy(record)
            change, listeners = self._next_change(action, record.source_uri, record)
        self._notify(change, listeners)
        return deepcopy(record)

    def update(
        self,
        source_uri: str,
        *,
        source_uri_new: str | None = None,
        name: str | None = None,
        enabled: bool | None = None,
        tasks: Iterable[TaskName] | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
        source_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        room_id: int | None | object = ...,
        fps: float | None | object = ...,
        loop: bool | None = None,
        draw_human: bool | None = None,
        draw_zone: bool | None = None,
        draw_fire: bool | None = None,
        draw_smoke: bool | None = None,
        draw_vehicle: bool | None = None,
        draw_plate: bool | None = None,
    ) -> SourceRecord:
        with self._lock:
            existing = self._records.get(source_uri)
            record = deepcopy(existing) if existing is not None else None
            if record is None:
                raise KeyError(source_uri)
            old_source_uri = record.source_uri
            if source_uri_new is not None:
                source_uri_new = source_uri_new.strip()
                if not source_uri_new:
                    raise ValueError("source_uri cannot be blank")
                if source_uri_new != old_source_uri and (
                    source_uri_new in self._records
                    or self._connection.execute(
                        "SELECT 1 FROM sources WHERE source_uri = ?",
                        (source_uri_new,),
                    ).fetchone()
                    is not None
                ):
                    raise ValueError(f"Source already exists: {source_uri_new}")
                record.source_uri = source_uri_new
            if name is not None:
                record.name = name
            if enabled is not None:
                record.enabled = enabled
            if tasks is not None:
                record.tasks = {TaskName(task) for task in tasks}
            if frame_width is not None:
                record.frame_width = frame_width
            if frame_height is not None:
                record.frame_height = frame_height
            if source_type is not None:
                if source_type not in SOURCE_TYPES:
                    raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}")
                record.source_type = source_type
            if metadata is not None:
                record.metadata = dict(metadata)
            if fps is not ...:
                record.fps = float(fps) if fps is not None else None
            if loop is not None:
                record.loop = loop
            if draw_human is not None:
                record.draw_human = draw_human
            if draw_zone is not None:
                record.draw_zone = draw_zone
            if draw_fire is not None:
                record.draw_fire = draw_fire
            if draw_smoke is not None:
                record.draw_smoke = draw_smoke
            if draw_vehicle is not None:
                record.draw_vehicle = draw_vehicle
            if draw_plate is not None:
                record.draw_plate = draw_plate
            if room_id is not ...:
                record.room_id = int(room_id) if room_id is not None else None
            record.updated_at_utc = _utc_now()
            record = self._normalized(record)
            room_id_val = record.room_id
            self._connection.execute(
                """
                UPDATE sources
                SET name = ?, enabled = ?, tasks_json = ?,
                    frame_width = ?, frame_height = ?,
                    room_id = ?, source_type = ?,
                    metadata_json = ?, fps = ?, loop = ?, draw_human = ?, draw_zone = ?,
                    draw_fire = ?, draw_smoke = ?, draw_vehicle = ?, draw_plate = ?,
                    updated_at_utc = ?
                WHERE source_uri = ?
                """,
                (
                    record.name,
                    int(record.enabled),
                    json.dumps(sorted(task.value for task in record.tasks)),
                    record.frame_width,
                    record.frame_height,
                    room_id_val,
                    record.source_type,
                    json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
                    record.fps,
                    int(record.loop),
                    int(record.draw_human),
                    int(record.draw_zone),
                    int(record.draw_fire),
                    int(record.draw_smoke),
                    int(record.draw_vehicle),
                    int(record.draw_plate),
                    record.updated_at_utc,
                    source_uri,
                ),
            )
            if source_uri_new is not None and source_uri_new != old_source_uri:
                self._connection.execute(
                    "UPDATE sources SET source_uri = ? WHERE source_uri = ?",
                    (source_uri_new, old_source_uri),
                )
            self._connection.commit()
            if source_uri_new is not None and source_uri_new != old_source_uri:
                self._records.pop(old_source_uri, None)
                self._records[source_uri_new] = deepcopy(record)
                change, listeners = self._next_change(
                    "updated",
                    source_uri_new,
                    record,
                    previous_source_uri=old_source_uri,
                )
            else:
                self._records[source_uri] = deepcopy(record)
                change, listeners = self._next_change("updated", source_uri, record)
        self._notify(change, listeners)
        return deepcopy(record)

    def rename(self, source_uri: str, new_source_uri: str) -> SourceRecord:
        return self.update(source_uri, source_uri_new=new_source_uri)

    def delete(self, source_uri: str) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM sources WHERE source_uri = ?",
                (source_uri,),
            )
            existed = cursor.rowcount > 0
            if not existed:
                return False
            self._connection.commit()
            self._records.pop(source_uri, None)
            change, listeners = self._next_change("deleted", source_uri, None)
        self._notify(change, listeners)
        return True

    def enabled_source_ids(self) -> list[str]:
        # Legacy — returns source_uri values; kept for backward compat as internal
        with self._lock:
            return [
                record.source_uri for record in self._records.values() if record.enabled
            ]

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._listeners.clear()
            self._connection.close()
