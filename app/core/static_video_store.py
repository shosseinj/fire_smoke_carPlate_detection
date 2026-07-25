from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.database import Database, Row, ensure_database


@dataclass(frozen=True, slots=True)
class StaticVideoRecord:
    source_uri: str
    name: str
    source_type: str = "static_video"


class StaticVideoStore:
    """Persistent store for uploaded static video file metadata.

    Each record tracks a video file uploaded to the media store. The
    actual video processing is driven by a corresponding ``SourceRecord``
    in the cameras table (created separately), so the ingestor continues
    to work unchanged. This store is the API-facing management layer.
    """

    def __init__(self, database: Database | str) -> None:
        self.database = ensure_database(database)

    def _connect(self):
        return self.database.connection()

    def list(self) -> list[StaticVideoRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT source_uri, name, source_type "
                "FROM static_videos ORDER BY source_uri ASC"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get(self, source_uri: str) -> StaticVideoRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT source_uri, name, source_type "
                "FROM static_videos WHERE source_uri = ?",
                (source_uri,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def create(
        self,
        name: str,
        source_uri: str,
        source_type: str = "static_video",
    ) -> StaticVideoRecord:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO static_videos (name, source_uri, source_type) "
                "VALUES (?, ?, ?)",
                (name, source_uri, source_type),
            )
        return StaticVideoRecord(
            name=name,
            source_uri=source_uri,
            source_type=source_type,
        )

    def update(
        self,
        source_uri: str,
        **changes: Any,
    ) -> StaticVideoRecord:
        current = self.get(source_uri)
        if current is None:
            raise KeyError(source_uri)

        allowed = {"name", "source_uri", "source_type"}
        unexpected = set(changes) - allowed
        if unexpected:
            raise ValueError(f"Unsupported fields: {sorted(unexpected)}")

        if not changes:
            return current

        set_parts = ", ".join(f"{k} = ?" for k in changes)
        values = [changes[k] for k in changes] + [source_uri]
        with self._connect() as connection:
            connection.execute(
                f"UPDATE static_videos SET {set_parts} WHERE source_uri = ?",
                values,
            )
        return StaticVideoRecord(
            name=changes.get("name", current.name),
            source_uri=changes.get("source_uri", current.source_uri),
            source_type=changes.get("source_type", current.source_type),
        )

    def delete(self, source_uri: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM static_videos WHERE source_uri = ?",
                (source_uri,),
            )
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_record(row: Row) -> StaticVideoRecord:
        return StaticVideoRecord(
            source_uri=str(row["source_uri"]),
            name=str(row["name"]),
            source_type=str(row["source_type"]),
        )
