from __future__ import annotations

import json
import threading
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.core.types import TaskName


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class SourceRecord:
    source_id: str
    name: str
    enabled: bool = True
    tasks: set[TaskName] = field(default_factory=set)
    source_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = field(default_factory=_utc_now)
    updated_at_utc: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["tasks"] = sorted(task.value for task in self.tasks)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceRecord":
        return cls(
            source_id=str(value["source_id"]),
            name=str(value.get("name") or value["source_id"]),
            enabled=bool(value.get("enabled", True)),
            tasks={TaskName(item) for item in value.get("tasks", [])},
            source_uri=value.get("source_uri"),
            metadata=dict(value.get("metadata") or {}),
            created_at_utc=str(value.get("created_at_utc") or _utc_now()),
            updated_at_utc=str(value.get("updated_at_utc") or _utc_now()),
        )


class SourceRegistry:
    def __init__(self, persistence_path: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, SourceRecord] = {}
        self.persistence_path = persistence_path
        if persistence_path is not None and persistence_path.is_file():
            self.load()

    def load(self) -> None:
        assert self.persistence_path is not None
        payload = json.loads(self.persistence_path.read_text(encoding="utf-8"))
        with self._lock:
            self._records = {
                item["source_id"]: SourceRecord.from_dict(item)
                for item in payload
            }

    def _persist(self) -> None:
        if self.persistence_path is None:
            return
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.persistence_path.with_suffix(self.persistence_path.suffix + ".tmp")
        payload = [item.to_dict() for item in self._records.values()]
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.persistence_path)

    def list(self) -> list[SourceRecord]:
        with self._lock:
            return deepcopy(list(self._records.values()))

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
        with self._lock:
            if record.source_id in self._records:
                raise ValueError(f"Source already exists: {record.source_id}")
            record.tasks = {TaskName(task) for task in record.tasks}
            self._records[record.source_id] = deepcopy(record)
            self._persist()
            return deepcopy(record)

    def upsert(self, record: SourceRecord) -> SourceRecord:
        with self._lock:
            existing = self._records.get(record.source_id)
            if existing is not None:
                record.created_at_utc = existing.created_at_utc
            record.updated_at_utc = _utc_now()
            record.tasks = {TaskName(task) for task in record.tasks}
            self._records[record.source_id] = deepcopy(record)
            self._persist()
            return deepcopy(record)

    def update(
        self,
        source_id: str,
        *,
        name: str | None = None,
        enabled: bool | None = None,
        tasks: Iterable[TaskName] | None = None,
        source_uri: str | None | object = ...,
        metadata: dict[str, Any] | None = None,
    ) -> SourceRecord:
        with self._lock:
            record = self._records.get(source_id)
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
            if metadata is not None:
                record.metadata = dict(metadata)
            record.updated_at_utc = _utc_now()
            self._persist()
            return deepcopy(record)

    def delete(self, source_id: str) -> bool:
        with self._lock:
            existed = self._records.pop(source_id, None) is not None
            if existed:
                self._persist()
            return existed

    def enabled_source_ids(self) -> list[str]:
        with self._lock:
            return [record.source_id for record in self._records.values() if record.enabled]
