from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.database import Database, Row, ensure_database
from app.time_utils import utc_now_text

QUALITY_PRESETS = {
    "low": {"width": 640, "height": 360, "fps": 10, "bitrate_bps": 750_000},
    "medium": {"width": 1280, "height": 720, "fps": 15, "bitrate_bps": 2_000_000},
    "high": {"width": 1920, "height": 1080, "fps": 25, "bitrate_bps": 5_000_000},
    "original": {"width": None, "height": None, "fps": None, "bitrate_bps": 8_000_000},
}
SEGMENT_SECONDS = frozenset({60, 120, 300, 900})
RETENTION_DAYS = frozenset({7, 30, 90, 365})


@dataclass(frozen=True, slots=True)
class RecordingPolicy:
    continuous_enabled: bool = False
    quality_preset: str = "medium"
    segment_seconds: int = 120
    retention_days: int = 30

    def __post_init__(self) -> None:
        if self.quality_preset not in QUALITY_PRESETS:
            raise ValueError("invalid recording quality preset")
        if self.segment_seconds not in SEGMENT_SECONDS:
            raise ValueError("invalid recording segment duration")
        if self.retention_days not in RETENTION_DAYS:
            raise ValueError("invalid recording retention period")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["output"] = dict(QUALITY_PRESETS[self.quality_preset])
        return values


class RecordingSettingsStore:
    def __init__(self, database: Database | str) -> None:
        self.database = ensure_database(database)
        self._ensure_global()

    def _ensure_global(self) -> None:
        now = utc_now_text()
        with self.database.connection() as connection:
            connection.execute(
                "INSERT INTO recording_settings (id, continuous_enabled, quality_preset, segment_seconds, "
                "retention_days, created_at_utc, updated_at_utc) VALUES (1, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (id) DO NOTHING",
                (False, "medium", 120, 30, now, now),
            )

    @staticmethod
    def _policy(row: Row) -> RecordingPolicy:
        return RecordingPolicy(bool(row["continuous_enabled"]), str(row["quality_preset"]),
                               int(row["segment_seconds"]), int(row["retention_days"]))

    def global_policy(self) -> RecordingPolicy:
        with self.database.connection() as connection:
            row = connection.execute("SELECT * FROM recording_settings WHERE id = 1").fetchone()
        if row is None:
            raise RuntimeError("recording settings singleton not found")
        return self._policy(row)

    def camera_override(self, source_uri: str) -> RecordingPolicy | None:
        with self.database.connection() as connection:
            row = connection.execute("SELECT * FROM recording_camera_settings WHERE source_uri = ?", (source_uri,)).fetchone()
        return self._policy(row) if row is not None else None

    def effective(self, source_uri: str) -> RecordingPolicy:
        return self.camera_override(source_uri) or self.global_policy()

    @staticmethod
    def _merge(current: RecordingPolicy, changes: dict[str, Any]) -> RecordingPolicy:
        values = asdict(current); values.update(changes)
        return RecordingPolicy(**values)

    def update_global(self, changes: dict[str, Any], updated_by: int | None) -> RecordingPolicy:
        policy = self._merge(self.global_policy(), changes)
        with self.database.connection() as connection:
            connection.execute(
                "UPDATE recording_settings SET continuous_enabled = ?, quality_preset = ?, segment_seconds = ?, "
                "retention_days = ?, updated_by = ?, updated_at_utc = ? WHERE id = 1",
                (policy.continuous_enabled, policy.quality_preset, policy.segment_seconds,
                 policy.retention_days, updated_by, utc_now_text()),
            )
        return policy

    def update_camera(self, source_uri: str, changes: dict[str, Any], updated_by: int | None) -> RecordingPolicy:
        policy = self._merge(self.camera_override(source_uri) or self.global_policy(), changes)
        now = utc_now_text()
        with self.database.connection() as connection:
            if connection.execute("SELECT source_uri FROM sources WHERE source_uri = ? AND source_uri <> '__default__'", (source_uri,)).fetchone() is None:
                raise KeyError(source_uri)
            connection.execute(
                "INSERT INTO recording_camera_settings (source_uri, continuous_enabled, quality_preset, segment_seconds, "
                "retention_days, created_by, updated_by, created_at_utc, updated_at_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (source_uri) DO UPDATE SET continuous_enabled = EXCLUDED.continuous_enabled, "
                "quality_preset = EXCLUDED.quality_preset, segment_seconds = EXCLUDED.segment_seconds, "
                "retention_days = EXCLUDED.retention_days, updated_by = EXCLUDED.updated_by, updated_at_utc = EXCLUDED.updated_at_utc",
                (source_uri, policy.continuous_enabled, policy.quality_preset, policy.segment_seconds,
                 policy.retention_days, updated_by, updated_by, now, now),
            )
        return policy

    def reset_global(self, updated_by: int | None) -> RecordingPolicy:
        return self.update_global(asdict(RecordingPolicy()), updated_by)

    def reset_camera(self, source_uri: str) -> bool:
        with self.database.connection() as connection:
            cursor = connection.execute("DELETE FROM recording_camera_settings WHERE source_uri = ?", (source_uri,))
        return cursor.rowcount == 1

    def enabled_sources(self) -> list[str]:
        with self.database.connection() as connection:
            if self.global_policy().continuous_enabled:
                rows = connection.execute(
                    "SELECT source_uri FROM sources WHERE source_uri <> '__default__' AND enabled = 1 AND source_type = 'rtsp' "
                    "AND source_uri NOT IN (SELECT source_uri FROM recording_camera_settings WHERE continuous_enabled = FALSE)"
                ).fetchall()
            else:
                rows = connection.execute("SELECT source_uri FROM recording_camera_settings WHERE continuous_enabled = TRUE").fetchall()
        return [str(row["source_uri"]) for row in rows]
