from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.database import Database, Row, ensure_database
from app.time_utils import utc_now_text


STATIC_VIDEO_STATUSES = frozenset(
    {"uploaded", "queued", "processing", "completed", "failed"}
)
STATIC_VIDEO_COLUMNS = (
    "id, source_uri, name, source_type, processing_status, processing_error, "
    "processing_started_at, processing_completed_at, processing_attempts, "
    "is_processed, loop, source_config_json"
)


@dataclass(frozen=True, slots=True)
class StaticVideoRecord:
    id: int
    source_uri: str
    name: str
    source_type: str = "static_video"
    processing_status: str = "uploaded"
    processing_error: str | None = None
    processing_started_at: str | None = None
    processing_completed_at: str | None = None
    processing_attempts: int = 0
    is_processed: bool = False
    loop: bool = False
    source_config: dict[str, Any] | None = None


class StaticVideoStore:
    def __init__(self, database: Database | str) -> None:
        self.database = ensure_database(database)

    def _connect(self):
        return self.database.connection()

    def list(self) -> list[StaticVideoRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT {STATIC_VIDEO_COLUMNS} FROM static_videos ORDER BY id ASC"
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get(self, source_uri: str) -> StaticVideoRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {STATIC_VIDEO_COLUMNS} FROM static_videos WHERE source_uri = ?",
                (source_uri,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def get_by_id(self, video_id: int) -> StaticVideoRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {STATIC_VIDEO_COLUMNS} FROM static_videos WHERE id = ?",
                (int(video_id),),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def create(
        self,
        name: str,
        source_uri: str,
        source_type: str = "static_video",
        *,
        loop: bool = False,
        source_config: dict[str, Any] | None = None,
    ) -> StaticVideoRecord:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO static_videos "
                "(name, source_uri, source_type, processing_status, processing_attempts, loop, source_config_json) "
                "VALUES (?, ?, ?, 'uploaded', 0, ?, ?)",
                (name, source_uri, source_type, int(loop), self._encode_config(source_config)),
            )
        record = self.get(source_uri)
        if record is None:
            raise RuntimeError("Failed to create static video")
        return record

    def update(self, source_uri: str, **changes: Any) -> StaticVideoRecord:
        current = self.get(source_uri)
        if current is None:
            raise KeyError(source_uri)
        allowed = {"name", "source_uri", "source_type", "loop", "source_config"}
        unexpected = set(changes) - allowed
        if unexpected:
            raise ValueError(f"Unsupported fields: {sorted(unexpected)}")
        if not changes:
            return current
        sql_changes = dict(changes)
        if "loop" in sql_changes:
            sql_changes["loop"] = int(bool(sql_changes["loop"]))
        if "source_config" in sql_changes:
            sql_changes["source_config_json"] = self._encode_config(sql_changes.pop("source_config"))
        set_parts = ", ".join(f"{key} = ?" for key in sql_changes)
        values = [sql_changes[key] for key in sql_changes] + [source_uri]
        with self._connect() as connection:
            connection.execute(f"UPDATE static_videos SET {set_parts} WHERE source_uri = ?", values)
        updated_uri = str(changes.get("source_uri", source_uri))
        record = self.get(updated_uri)
        if record is None:
            raise RuntimeError("Failed to update static video")
        return record

    def mark_processing(self, source_uri: str) -> StaticVideoRecord | None:
        now = utc_now_text()
        with self._connect() as connection:
            connection.execute(
                "UPDATE static_videos SET processing_status = 'processing', "
                "is_processed = FALSE, processing_error = NULL, processing_started_at = ?, "
                "processing_completed_at = NULL, processing_attempts = processing_attempts + 1 "
                "WHERE source_uri = ? AND processing_status = 'queued'",
                (now, source_uri),
            )
        return self.get(source_uri)

    def mark_queued(
        self,
        source_uri: str,
        *,
        source_config: dict[str, Any] | None = None,
    ) -> StaticVideoRecord | None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE static_videos SET processing_status = 'queued', "
                "processing_error = NULL, processing_started_at = NULL, "
                "processing_completed_at = NULL, is_processed = FALSE, "
                "source_config_json = COALESCE(?, source_config_json) "
                "WHERE source_uri = ? AND processing_status = 'uploaded'",
                (self._encode_config(source_config), source_uri),
            )
        return self.get(source_uri)

    def mark_uploaded(self, source_uri: str) -> StaticVideoRecord | None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE static_videos SET processing_status = 'uploaded', "
                "processing_error = NULL, processing_started_at = NULL, "
                "processing_completed_at = NULL, is_processed = FALSE "
                "WHERE source_uri = ?",
                (source_uri,),
            )
        return self.get(source_uri)

    def mark_completed(
        self, source_uri: str, *, source_config: dict[str, Any] | None = None
    ) -> StaticVideoRecord | None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE static_videos SET processing_status = 'completed', is_processed = TRUE, processing_error = NULL, "
                "processing_completed_at = ?, source_config_json = COALESCE(?, source_config_json) "
                "WHERE source_uri = ?",
                (utc_now_text(), self._encode_config(source_config), source_uri),
            )
        return self.get(source_uri)

    def mark_failed(
        self, source_uri: str, error: str, *, source_config: dict[str, Any] | None = None
    ) -> StaticVideoRecord | None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE static_videos SET processing_status = 'failed', is_processed = FALSE, processing_error = ?, "
                "processing_completed_at = ?, source_config_json = COALESCE(?, source_config_json) "
                "WHERE source_uri = ?",
                (error[:2000], utc_now_text(), self._encode_config(source_config), source_uri),
            )
        return self.get(source_uri)

    def reset_for_retry(self, source_uri: str, *, loop: bool | None = None) -> StaticVideoRecord | None:
        sets = [
            "processing_status = 'queued'",
            "processing_error = NULL",
            "processing_started_at = NULL",
            "processing_completed_at = NULL",
            "is_processed = FALSE",
        ]
        params: list[Any] = []
        if loop is not None:
            sets.append("loop = ?")
            params.append(int(loop))
        params.append(source_uri)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE static_videos SET {', '.join(sets)} WHERE source_uri = ?", params
            )
        return self.get(source_uri)

    def recover_interrupted(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE static_videos SET processing_status = 'queued', is_processed = FALSE, processing_error = NULL, "
                "processing_started_at = NULL, processing_completed_at = NULL "
                "WHERE processing_status = 'processing'"
            )
        return cursor.rowcount

    def delete(self, source_uri: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM static_videos WHERE source_uri = ?", (source_uri,))
        return cursor.rowcount > 0

    @staticmethod
    def _encode_config(value: dict[str, Any] | None) -> str | None:
        return json.dumps(value, ensure_ascii=False) if value is not None else None

    @staticmethod
    def _row_to_record(row: Row) -> StaticVideoRecord:
        raw_config = row.get("source_config_json")
        return StaticVideoRecord(
            id=int(row["id"]),
            source_uri=str(row["source_uri"]),
            name=str(row["name"]),
            source_type=str(row["source_type"]),
            processing_status=str(row.get("processing_status") or "uploaded"),
            processing_error=str(row["processing_error"]) if row.get("processing_error") else None,
            processing_started_at=str(row["processing_started_at"]) if row.get("processing_started_at") else None,
            processing_completed_at=str(row["processing_completed_at"]) if row.get("processing_completed_at") else None,
            processing_attempts=int(row.get("processing_attempts") or 0),
            is_processed=bool(row.get("is_processed")),
            loop=bool(row.get("loop")),
            source_config=json.loads(str(raw_config)) if raw_config else None,
        )
