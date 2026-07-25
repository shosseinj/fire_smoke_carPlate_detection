from __future__ import annotations

import threading
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from app.core.operational_settings import OperationalSettings
from app.database import Database, ensure_database

# Well-known key for the global-defaults row
_DEFAULT_KEY = "__default__"

_SOURCE_COLUMNS = (
    "source_uri, fire_confidence, smoke_confidence, plate_confidence, plate_iou, "
    "vehicle_confidence, vehicle_iou, face_human_confidence, "
    "face_detection_confidence, face_recognition_threshold, updated_at_utc"
)

# Confidence fields stored in the sources table
SOURCE_FIELDS: tuple[str, ...] = (
    "fire_confidence",
    "smoke_confidence",
    "plate_confidence",
    "plate_iou",
    "vehicle_confidence",
    "vehicle_iou",
    "face_human_confidence",
    "face_detection_confidence",
    "face_recognition_threshold",
)

# Column name → source field (they're the same for this table)
_COLUMN_TO_FIELD: dict[str, str] = {f: f for f in SOURCE_FIELDS}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SourceSettingsStore:
    """Per-source confidence threshold overrides.

    The ``sources`` table stores per-source overrides for the 9 confidence
    fields, keyed by ``source_uri`` (no ``camera_id``).

    A special row with ``source_uri = '__default__'`` holds the global
    defaults.  When a per-source row has NULL for a field the global default
    from the ``__default__`` row (or the env fallback) applies.
    """

    def __init__(self, database: Database | str, env_defaults: OperationalSettings | None = None) -> None:
        self.database = ensure_database(database)
        self._lock = threading.RLock()
        self._env_defaults = (env_defaults or OperationalSettings()).to_dict()
        self._ensure_default()

    def _connect(self):
        return self.database.connection()

    def _ensure_default(self) -> None:
        """Create the ``__default__`` row on first startup if missing."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT source_uri FROM sources WHERE source_uri = ?",
                (_DEFAULT_KEY,),
            ).fetchone()
            if row is None:
                now = _utc_now()
                cols = ", ".join(SOURCE_FIELDS)
                placeholders = ", ".join("?" for _ in SOURCE_FIELDS)
                values = [self._env_defaults.get(f) for f in SOURCE_FIELDS]
                conn.execute(
                    f"INSERT INTO sources (source_uri, {cols}, updated_at_utc) "
                    f"VALUES (?, {placeholders}, ?)",
                    [_DEFAULT_KEY, *values, now],
                )
                conn.commit()

    def _row_to_dict(self, row: Any) -> dict[str, Any]:
        """Convert a sources row to a dict with source_uri + confidence fields."""
        result: dict[str, Any] = {"source_uri": str(row["source_uri"])}
        for field in SOURCE_FIELDS:
            val = row[field]
            result[field] = float(val) if val is not None else None
        result["updated_at_utc"] = str(row["updated_at_utc"])
        return result

    # ── Global defaults ──────────────────────────────────────────────

    def get_default(self) -> OperationalSettings:
        """Return the global defaults as an ``OperationalSettings``.

        NULL fields fall back to env-startup values.
        """
        with self._lock, self._connect() as conn:
            row = conn.execute(
                f"SELECT {_SOURCE_COLUMNS} FROM sources WHERE source_uri = ?",
                (_DEFAULT_KEY,),
            ).fetchone()
        if row is None:
            return OperationalSettings(**self._env_defaults)
        merged = dict(self._env_defaults)
        for field in SOURCE_FIELDS:
            val = row[field]
            if val is not None:
                merged[field] = float(val)
        return OperationalSettings(**merged)

    def set_default(self, changes: dict[str, Any]) -> OperationalSettings:
        """Update global defaults (upsert the ``__default__`` row)."""
        self._ensure_default()
        unknown = set(changes) - set(SOURCE_FIELDS)
        if unknown:
            raise ValueError(f"Unknown source setting(s): {sorted(unknown)}")
        now = _utc_now()
        with self._lock, self._connect() as conn:
            set_parts = ", ".join(f"{f} = ?" for f in changes)
            values = [changes[f] for f in changes] + [now, _DEFAULT_KEY]
            conn.execute(
                f"UPDATE sources SET {set_parts}, updated_at_utc = ? "
                f"WHERE source_uri = ?",
                values,
            )
            conn.commit()
        return self.get_default()

    def reset_default(self) -> OperationalSettings:
        """Reset global defaults back to env-startup values."""
        with self._lock, self._connect() as conn:
            now = _utc_now()
            set_parts = ", ".join(f"{f} = ?" for f in SOURCE_FIELDS)
            values = [self._env_defaults.get(f) for f in SOURCE_FIELDS] + [now, _DEFAULT_KEY]
            conn.execute(
                f"UPDATE sources SET {set_parts}, updated_at_utc = ? "
                f"WHERE source_uri = ?",
                values,
            )
            conn.commit()
        return self.get_default()

    # ── Per-source overrides ─────────────────────────────────────────

    def get(self, source_uri: str) -> dict[str, float | None]:
        """Return per-source override values for *source_uri*.

        Only fields that have been explicitly set (non-NULL) are included.
        """
        with self._lock, self._connect() as conn:
            row = conn.execute(
                f"SELECT {_SOURCE_COLUMNS} FROM sources WHERE source_uri = ?",
                (source_uri,),
            ).fetchone()
        if row is None:
            return {}
        result: dict[str, float | None] = {}
        for field in SOURCE_FIELDS:
            val = row[field]
            if val is not None:
                result[field] = float(val)
        return result

    def set(self, source_uri: str, overrides: dict[str, float]) -> None:
        """Upsert per-source overrides for *source_uri*."""
        if source_uri == _DEFAULT_KEY:
            raise ValueError("Use set_default() for global defaults")
        unknown = set(overrides) - set(SOURCE_FIELDS)
        if unknown:
            raise ValueError(f"Unknown source setting(s): {sorted(unknown)}")
        now = _utc_now()
        with self._lock, self._connect() as conn:
            existing = self.get(source_uri)
            merged = dict(existing)
            merged.update(overrides)
            cols = ", ".join(SOURCE_FIELDS)
            placeholders = ", ".join("?" for _ in SOURCE_FIELDS)
            values = [merged.get(f) for f in SOURCE_FIELDS]
            update_parts = ", ".join(
                f"{f} = COALESCE(excluded.{f}, sources.{f})"
                if f not in overrides
                else f"{f} = excluded.{f}"
                for f in SOURCE_FIELDS
            )
            conn.execute(
                f"""
                INSERT INTO sources (source_uri, {cols}, updated_at_utc)
                VALUES (?, {placeholders}, ?)
                ON CONFLICT(source_uri) DO UPDATE SET
                    {update_parts},
                    updated_at_utc = excluded.updated_at_utc
                """,
                [source_uri, *values, now],
            )
            conn.commit()

    def delete(self, source_uri: str) -> bool:
        """Remove all overrides for *source_uri* (not the default row)."""
        if source_uri == _DEFAULT_KEY:
            raise ValueError("Cannot delete the global-default row")
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM sources WHERE source_uri = ?",
                (source_uri,),
            )
            conn.commit()
        return cursor.rowcount > 0

    def rename(self, old_source_uri: str, new_source_uri: str) -> bool:
        """Move overrides from *old_source_uri* to *new_source_uri*."""
        if old_source_uri == _DEFAULT_KEY or new_source_uri == _DEFAULT_KEY:
            raise ValueError("Cannot rename the global-default row")
        if old_source_uri == new_source_uri:
            return True
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "UPDATE sources SET source_uri = ? WHERE source_uri = ?",
                (new_source_uri, old_source_uri),
            )
            conn.commit()
        return cursor.rowcount > 0

    def resolve(
        self,
        source_uri: str,
        global_settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Merge global defaults with per-source overrides.

        Returns a dict with all 9 confidence fields.  *global_settings*
        is typically ``get_default().to_dict()`` but can be overridden.
        """
        base = self.get_default().to_dict()
        if global_settings is not None:
            base.update(global_settings)
        overrides = self.get(source_uri)
        base.update(overrides)
        return base

    def all_settings(self) -> list[dict[str, Any]]:
        """Return all rows (global default + per-source) for admin API."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_SOURCE_COLUMNS} FROM sources ORDER BY source_uri"
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]
