from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2

from app.core.types import FramePacket, TaskName, TaskResult
from app.fire_core.policy import FireSmokePolicyConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _PendingEvent:
    frame: Any
    result: TaskResult


class FireSmokeLogStore:
    """Persistent hazard events with non-blocking, background snapshot writes."""

    def __init__(
        self,
        database_path: Path,
        media_root: Path,
        *,
        default_policy: FireSmokePolicyConfig = FireSmokePolicyConfig(),
        queue_size: int = 64,
    ) -> None:
        self.database_path = database_path
        self.media_root = media_root
        self.snapshot_dir = media_root / "fire_smoke_snapshots"
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._policy_revision = 0
        self._queue: queue.Queue[_PendingEvent | None] = queue.Queue(maxsize=queue_size)
        self._dropped = 0
        self._saved = 0
        self._closed = False
        self._initialize(default_policy.validated())
        self._policy = self._load_policy()
        self._thread = threading.Thread(
            target=self._run,
            name="fire-smoke-log-writer",
            daemon=True,
        )
        self._thread.start()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self, default: FireSmokePolicyConfig) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=10000")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS fire_smoke_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera TEXT NOT NULL,
                    time TEXT NOT NULL,
                    incident_id TEXT,
                    severity TEXT NOT NULL,
                    fire_count INTEGER NOT NULL,
                    smoke_count INTEGER NOT NULL,
                    fire_confidence REAL NOT NULL,
                    smoke_confidence REAL NOT NULL,
                    window_seconds REAL NOT NULL,
                    snapshot_url TEXT NOT NULL DEFAULT '',
                    details_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS fire_smoke_settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    window_seconds REAL NOT NULL,
                    low_count INTEGER NOT NULL,
                    medium_count INTEGER NOT NULL,
                    high_count INTEGER NOT NULL,
                    updated_at_utc TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO fire_smoke_settings (
                    id, window_seconds, low_count, medium_count, high_count, updated_at_utc
                ) VALUES (1, ?, ?, ?, ?, ?)
                """,
                (
                    default.window_seconds,
                    default.low_count,
                    default.medium_count,
                    default.high_count,
                    datetime.utcnow().isoformat() + "Z",
                ),
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_fire_smoke_logs_time ON fire_smoke_logs(time)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_fire_smoke_logs_camera ON fire_smoke_logs(camera)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_fire_smoke_logs_severity ON fire_smoke_logs(severity)"
            )
            connection.commit()

    def _load_policy(self) -> FireSmokePolicyConfig:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT window_seconds, low_count, medium_count, high_count "
                "FROM fire_smoke_settings WHERE id = 1"
            ).fetchone()
        assert row is not None
        return FireSmokePolicyConfig(**dict(row)).validated()

    def policy_snapshot(self) -> tuple[int, FireSmokePolicyConfig]:
        with self._lock:
            return self._policy_revision, self._policy

    def update_policy(self, policy: FireSmokePolicyConfig) -> dict[str, Any]:
        policy = policy.validated()
        updated_at = datetime.utcnow().isoformat() + "Z"
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE fire_smoke_settings
                SET window_seconds = ?, low_count = ?, medium_count = ?,
                    high_count = ?, updated_at_utc = ?
                WHERE id = 1
                """,
                (
                    policy.window_seconds,
                    policy.low_count,
                    policy.medium_count,
                    policy.high_count,
                    updated_at,
                ),
            )
            connection.commit()
            self._policy = policy
            self._policy_revision += 1
        return {**asdict(policy), "updated_at_utc": updated_at}

    def settings(self) -> dict[str, Any]:
        revision, policy = self.policy_snapshot()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT updated_at_utc FROM fire_smoke_settings WHERE id = 1"
            ).fetchone()
        return {
            **asdict(policy),
            "revision": revision,
            "updated_at_utc": str(row["updated_at_utc"]),
        }

    def observe_result(self, packet: FramePacket, result: TaskResult) -> None:
        if (
            self._closed
            or result.error
            or result.task != TaskName.FIRE_SMOKE
            or result.data.get("severity") == "none"
            or not result.data.get("severity_changed", False)
        ):
            return
        try:
            # Copy once while the ingest buffer is still valid; all drawing,
            # JPEG encoding, and SQLite I/O happens on the writer thread.
            self._queue.put_nowait(_PendingEvent(packet.frame.copy(), result))
        except queue.Full:
            self._dropped += 1
            LOGGER.warning("Fire/smoke log queue is full; newest event was dropped")

    @staticmethod
    def _incident_id(result: TaskResult) -> str | None:
        for event in result.data.get("events", []):
            if event.get("incident_id"):
                return str(event["incident_id"])
        return None

    def _save(self, pending: _PendingEvent) -> None:
        result = pending.result
        fire = dict(result.data.get("fire") or {})
        smoke = dict(result.data.get("smoke") or {})
        frame = pending.frame
        for track in result.data.get("tracks", []):
            if track.get("label") not in {"fire", "smoke"}:
                continue
            x1, y1, x2, y2 = (int(round(float(v))) for v in track.get("bbox", [0, 0, 0, 0]))
            color = (0, 0, 255) if track.get("label") == "fire" else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{track.get('label')} {float(track.get('confidence', 0.0)):.2f}",
                (x1, max(20, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        timestamp = result.processed_at_utc.replace(":", "-").replace("+", "_")
        filename = f"{result.source_id}_{timestamp}_{uuid4().hex[:8]}.jpg"
        snapshot_path = self.snapshot_dir / filename
        if not cv2.imwrite(str(snapshot_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
            raise RuntimeError(f"Could not save fire/smoke snapshot: {snapshot_path}")
        snapshot_url = f"/media/fire_smoke_snapshots/{filename}"
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO fire_smoke_logs (
                    camera, time, incident_id, severity, fire_count, smoke_count,
                    fire_confidence, smoke_confidence, window_seconds,
                    snapshot_url, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.source_id,
                    result.processed_at_utc,
                    self._incident_id(result),
                    result.data["severity"],
                    int(fire.get("positive_count", 0)),
                    int(smoke.get("positive_count", 0)),
                    float(fire.get("max_confidence", 0.0)),
                    float(smoke.get("max_confidence", 0.0)),
                    float(result.data.get("severity_window_seconds", 3.0)),
                    snapshot_url,
                    json.dumps(result.data, ensure_ascii=False, sort_keys=True),
                ),
            )
            connection.commit()
            self._saved += 1

    def _run(self) -> None:
        while True:
            pending = self._queue.get()
            try:
                if pending is None:
                    return
                self._save(pending)
            except Exception:
                LOGGER.exception("Fire/smoke event persistence failed")
            finally:
                self._queue.task_done()

    def list(
        self,
        *,
        camera: str | None = None,
        severity: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if camera:
            clauses.append("camera = ?")
            parameters.append(camera.strip())
        if severity:
            clauses.append("severity = ?")
            parameters.append(severity.strip().lower())
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(max(1, min(int(limit), 1000)))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT id, camera, time, incident_id, severity, fire_count, "
                "smoke_count, fire_confidence, smoke_confidence, window_seconds, "
                f"snapshot_url FROM fire_smoke_logs{where} ORDER BY id DESC LIMIT ?",
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM fire_smoke_logs").fetchone()
        return int(row["count"] if row else 0)

    def status(self) -> dict[str, Any]:
        return {
            "count": self.count(),
            "queued": self._queue.qsize(),
            "saved": self._saved,
            "dropped": self._dropped,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=10.0)
