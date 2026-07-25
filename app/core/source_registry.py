from __future__ import annotations

import json
import logging
import threading
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from app.core.types import TaskName
from app.database import Database, IntegrityError, Row, ensure_database
from app.time_utils import utc_now_text


LOGGER = logging.getLogger(__name__)


def _utc_now() -> str:
    return utc_now_text()


RTSP = "rtsp"
STATIC_VIDEO = "static_video"
SOURCE_TYPES = frozenset({RTSP, STATIC_VIDEO})


@dataclass(slots=True)
class SourceRecord:
    source_id: str
    name: str
    enabled: bool = True
    tasks: set[TaskName] = field(default_factory=set)
    source_uri: str | None = None
    frame_width: int = 640
    frame_height: int = 640
    source_type: str = RTSP
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = field(default_factory=_utc_now)
    updated_at_utc: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["tasks"] = sorted(task.value for task in self.tasks)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceRecord":
        source_id = value.get("source_id", value.get("camera_id"))
        if source_id is None:
            raise ValueError("Camera record requires source_id or camera_id")
        source_type = value.get("source_type", RTSP)
        if source_type not in SOURCE_TYPES:
            source_type = RTSP
        return cls(
            source_id=str(source_id),
            name=str(value.get("name") or source_id),
            enabled=bool(value.get("enabled", True)),
            tasks={TaskName(item) for item in value.get("tasks", [])},
            source_uri=value.get("source_uri"),
            frame_width=int(value.get("frame_width", 640)),
            frame_height=int(value.get("frame_height", 640)),
            source_type=source_type,
            metadata=dict(value.get("metadata") or {}),
            created_at_utc=str(value.get("created_at_utc") or _utc_now()),
            updated_at_utc=str(value.get("updated_at_utc") or _utc_now()),
        )


@dataclass(frozen=True, slots=True)
class SourceChange:
    action: str
    source_id: str
    revision: int
    record: SourceRecord | None


SourceChangeListener = Callable[[SourceChange], None]


class SourceRegistry:
    """Thread-safe camera registry backed by PostgreSQL."""

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
                SELECT camera_id, name, enabled, tasks_json, source_uri,
                       frame_width, frame_height, section_id, source_type,
                       metadata_json, created_at_utc, updated_at_utc
                FROM cameras
                ORDER BY created_at_utc, camera_id
                """
            ).fetchall()
            self._records = {
                str(row["camera_id"]): self._row_to_record(row) for row in rows
            }

    @staticmethod
    def _normalized(record: SourceRecord) -> SourceRecord:
        value = deepcopy(record)
        value.source_id = value.source_id.strip()
        value.name = value.name.strip()
        if not value.source_id:
            raise ValueError("source_id cannot be blank")
        if not value.name:
            raise ValueError("name cannot be blank")
        value.tasks = {TaskName(task) for task in value.tasks}
        value.metadata = dict(value.metadata)
        value.frame_width = int(value.frame_width)
        value.frame_height = int(value.frame_height)
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
        # If section_id is stored as a dedicated column, merge it into metadata for backward compat
        section_id = row["section_id"]
        if section_id is not None and "section_id" not in metadata:
            metadata["section_id"] = int(section_id)
        return SourceRecord(
            source_id=str(row["camera_id"]),
            name=str(row["name"]),
            enabled=bool(row["enabled"]),
            tasks={TaskName(item) for item in json.loads(row["tasks_json"])},
            source_uri=row["source_uri"],
            frame_width=int(row["frame_width"]),
            frame_height=int(row["frame_height"]),
            source_type=str(row["source_type"]) if row["source_type"] else RTSP,
            metadata=metadata,
            created_at_utc=str(row["created_at_utc"]),
            updated_at_utc=str(row["updated_at_utc"]),
        )

    @staticmethod
    def _parameters(record: SourceRecord) -> tuple[Any, ...]:
        # Extract section_id from metadata if present, for the dedicated column
        metadata = dict(record.metadata)
        section_id = metadata.pop("section_id", None)
        return (
            record.source_id,
            record.name,
            int(record.enabled),
            json.dumps(sorted(task.value for task in record.tasks)),
            record.source_uri,
            record.frame_width,
            record.frame_height,
            section_id,
            record.source_type,
            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            record.created_at_utc,
            record.updated_at_utc,
        )

    def _next_change(
        self,
        action: str,
        source_id: str,
        record: SourceRecord | None,
    ) -> tuple[SourceChange, tuple[SourceChangeListener, ...]]:
        self._revision += 1
        change = SourceChange(
            action=action,
            source_id=source_id,
            revision=self._revision,
            record=deepcopy(record),
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
                    "Camera registry listener failed: action=%s camera_id=%s",
                    change.action,
                    change.source_id,
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
            self._connection.executemany(
                """
                INSERT INTO cameras (
                    camera_id, name, enabled, tasks_json, source_uri,
                    frame_width, frame_height, section_id, source_type,
                    metadata_json, created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [self._parameters(record) for record in prepared],
            )
            self._connection.commit()
            self._records = {
                record.source_id: deepcopy(record) for record in prepared
            }
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

    def get(self, source_id: str) -> SourceRecord | None:
        with self._lock:
            value = self._records.get(source_id)
            return deepcopy(value) if value is not None else None

    def require(self, source_id: str) -> SourceRecord:
        value = self.get(source_id)
        if value is None:
            raise KeyError(source_id)
        return value

    def create(self, record: SourceRecord) -> SourceRecord:
        record = self._normalized(record)
        with self._lock:
            try:
                self._connection.execute(
                    """
                    INSERT INTO cameras (
                        camera_id, name, enabled, tasks_json, source_uri,
                        frame_width, frame_height, section_id, source_type,
                        metadata_json, created_at_utc, updated_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    self._parameters(record),
                )
                self._connection.commit()
            except IntegrityError as exc:
                self._connection.rollback()
                raise ValueError(f"Source already exists: {record.source_id}") from exc
            self._records[record.source_id] = deepcopy(record)
            change, listeners = self._next_change("created", record.source_id, record)
        self._notify(change, listeners)
        return deepcopy(record)

    def upsert(self, record: SourceRecord) -> SourceRecord:
        record = self._normalized(record)
        with self._lock:
            existing = self._records.get(record.source_id)
            action = "created" if existing is None else "updated"
            if existing is not None:
                record.created_at_utc = existing.created_at_utc
            record.updated_at_utc = _utc_now()
            self._connection.execute(
                """
                INSERT INTO cameras (
                    camera_id, name, enabled, tasks_json, source_uri,
                    frame_width, frame_height, section_id, source_type,
                    metadata_json, created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(camera_id) DO UPDATE SET
                    name = excluded.name,
                    enabled = excluded.enabled,
                    tasks_json = excluded.tasks_json,
                    source_uri = excluded.source_uri,
                    frame_width = excluded.frame_width,
                    frame_height = excluded.frame_height,
                    section_id = excluded.section_id,
                    source_type = excluded.source_type,
                    metadata_json = excluded.metadata_json,
                    updated_at_utc = excluded.updated_at_utc
                """,
                self._parameters(record),
            )
            self._connection.commit()
            self._records[record.source_id] = deepcopy(record)
            change, listeners = self._next_change(action, record.source_id, record)
        self._notify(change, listeners)
        return deepcopy(record)

    def update(
        self,
        source_id: str,
        *,
        name: str | None = None,
        enabled: bool | None = None,
        tasks: Iterable[TaskName] | None = None,
        source_uri: str | None | object = ...,
        frame_width: int | None = None,
        frame_height: int | None = None,
        source_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        section_id: int | None | object = ...,
    ) -> SourceRecord:
        with self._lock:
            existing = self._records.get(source_id)
            record = deepcopy(existing) if existing is not None else None
            if record is None:
                raise KeyError(source_id)
            if name is not None:
                record.name = name
            if enabled is not None:
                record.enabled = enabled
            if tasks is not None:
                record.tasks = {TaskName(task) for task in tasks}
            if source_uri is not ...:
                record.source_uri = source_uri  # type: ignore[assignment]
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
            if section_id is not ...:
                # Handle section_id: store in metadata as well for backward compat
                record.metadata = dict(record.metadata)
                if section_id is not None:
                    record.metadata["section_id"] = section_id
                else:
                    record.metadata.pop("section_id", None)
            record.updated_at_utc = _utc_now()
            record = self._normalized(record)
            # Build parameters with section_id extracted from metadata
            params = list(self._parameters(record))
            # _parameters already extracts section_id from metadata
            section_id_val = params[7]  # section_id is at index 7
            self._connection.execute(
                """
                UPDATE cameras
                SET name = ?, enabled = ?, tasks_json = ?, source_uri = ?,
                    frame_width = ?, frame_height = ?,
                    section_id = ?, source_type = ?,
                    metadata_json = ?, updated_at_utc = ?
                WHERE camera_id = ?
                """,
                (
                    record.name,
                    int(record.enabled),
                    json.dumps(sorted(task.value for task in record.tasks)),
                    record.source_uri,
                    record.frame_width,
                    record.frame_height,
                    section_id_val,
                    record.source_type,
                    json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
                    record.updated_at_utc,
                    source_id,
                ),
            )
            self._connection.commit()
            self._records[source_id] = deepcopy(record)
            change, listeners = self._next_change("updated", source_id, record)
        self._notify(change, listeners)
        return deepcopy(record)

    def delete(self, source_id: str) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM cameras WHERE camera_id = ?",
                (source_id,),
            )
            existed = cursor.rowcount > 0
            if not existed:
                return False
            self._connection.commit()
            self._records.pop(source_id, None)
            change, listeners = self._next_change("deleted", source_id, None)
        self._notify(change, listeners)
        return True

    def enabled_source_ids(self) -> list[str]:
        with self._lock:
            return [
                record.source_id for record in self._records.values() if record.enabled
            ]

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._listeners.clear()
            self._connection.close()
