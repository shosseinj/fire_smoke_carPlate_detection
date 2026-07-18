from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.types import FramePacket, TaskResult


LOGGER = logging.getLogger(__name__)
_STOP = object()


@dataclass(slots=True)
class HumanLogEvent:
    session_id: str
    camera: str
    track_id: int
    name: str
    captured_at_utc: str
    recognition_score: float
    ref_img_id: str | int | None
    bbox: list[float]
    frame: np.ndarray


class HumanLogStore:
    """Asynchronous one-row-per-ByteTrack human history and snapshots."""

    def __init__(
        self,
        database_path: Path,
        saved_media_path: Path,
        *,
        queue_size: int = 256,
    ) -> None:
        self.database_path = database_path.resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = (saved_media_path / "human_snapshots").resolve()
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[HumanLogEvent | object] = queue.Queue(
            maxsize=max(8, int(queue_size))
        )
        self._observed: dict[tuple[str, str, int], str] = {}
        self._lock = threading.RLock()
        self._dropped_events = 0
        self._saved_snapshots = 0
        self._last_error: str | None = None
        self._closed = False
        self._create_schema()
        self._thread = threading.Thread(
            target=self._run,
            name="human-log-writer",
            daemon=True,
        )
        self._thread.start()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _create_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS human_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    camera TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    name TEXT NOT NULL DEFAULT 'Unknown',
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    recognition_score REAL NOT NULL DEFAULT 0,
                    ref_img_id TEXT,
                    snapshot_url TEXT NOT NULL DEFAULT '',
                    UNIQUE(session_id, camera, track_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_human_logs_camera ON human_logs(camera)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_human_logs_name ON human_logs(name)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_human_logs_last_seen ON human_logs(last_seen)"
            )

    def observe_result(self, packet: FramePacket, result: TaskResult) -> None:
        if result.error:
            return
        session_id = str(result.data.get("tracking_session_id") or "unknown-session")
        for human in result.data.get("humans", []):
            track_id = human.get("track_id")
            if track_id is None:
                continue
            name = str(human.get("person") or "Unknown").strip() or "Unknown"
            key = (session_id, packet.source_id, int(track_id))
            with self._lock:
                previous_name = self._observed.get(key)
                should_capture = previous_name is None or (
                    previous_name == "Unknown" and name != "Unknown"
                )
                if not should_capture:
                    continue
                self._observed[key] = name
            event = HumanLogEvent(
                session_id=session_id,
                camera=packet.source_id,
                track_id=int(track_id),
                name=name,
                captured_at_utc=packet.captured_at_utc,
                recognition_score=float(human.get("recognition_score", 0.0) or 0.0),
                ref_img_id=human.get("ref_img_id"),
                bbox=[float(value) for value in human.get("bbox", [0, 0, 0, 0])[:4]],
                frame=packet.frame.copy(),
            )
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                with self._lock:
                    self._dropped_events += 1
                    # Permit a later frame to retry this important state change.
                    if previous_name is None:
                        self._observed.pop(key, None)
                    else:
                        self._observed[key] = previous_name
                LOGGER.warning("Human log queue is full; newest event was dropped")

    @staticmethod
    def _bounded_box(frame: np.ndarray, bbox: list[float]) -> tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
        return (
            max(0, min(x1, width - 1)),
            max(0, min(y1, height - 1)),
            max(1, min(x2, width)),
            max(1, min(y2, height)),
        )

    def _save_snapshot(self, event: HumanLogEvent) -> tuple[str, Path]:
        frame = event.frame.copy()
        x1, y1, x2, y2 = self._bounded_box(frame, event.bbox)
        color = (70, 230, 100) if event.name != "Unknown" else (0, 190, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = f"{event.name} #{event.track_id}"
        cv2.putText(
            frame,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
        filename = (
            f"{event.camera}_{event.track_id}_{uuid.uuid4().hex[:12]}.jpg"
            .replace("/", "_")
            .replace("\\", "_")
        )
        path = self.snapshot_dir / filename
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
            raise RuntimeError(f"Could not save human snapshot: {path}")
        return f"/media/human_snapshots/{filename}", path

    def _write(self, event: HumanLogEvent) -> None:
        snapshot_url, snapshot_path = self._save_snapshot(event)
        old_snapshot_url = ""
        try:
            with self._connect() as connection:
                existing = connection.execute(
                    """
                    SELECT snapshot_url FROM human_logs
                    WHERE session_id = ? AND camera = ? AND track_id = ?
                    """,
                    (event.session_id, event.camera, event.track_id),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO human_logs (
                            session_id, camera, track_id, name, first_seen,
                            last_seen, recognition_score, ref_img_id, snapshot_url
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event.session_id,
                            event.camera,
                            event.track_id,
                            event.name,
                            event.captured_at_utc,
                            event.captured_at_utc,
                            event.recognition_score,
                            None if event.ref_img_id is None else str(event.ref_img_id),
                            snapshot_url,
                        ),
                    )
                else:
                    old_snapshot_url = str(existing["snapshot_url"] or "")
                    connection.execute(
                        """
                        UPDATE human_logs
                        SET name = ?, last_seen = ?, recognition_score = ?,
                            ref_img_id = ?, snapshot_url = ?
                        WHERE session_id = ? AND camera = ? AND track_id = ?
                        """,
                        (
                            event.name,
                            event.captured_at_utc,
                            event.recognition_score,
                            None if event.ref_img_id is None else str(event.ref_img_id),
                            snapshot_url,
                            event.session_id,
                            event.camera,
                            event.track_id,
                        ),
                    )
            if old_snapshot_url and old_snapshot_url != snapshot_url:
                (self.snapshot_dir / Path(old_snapshot_url).name).unlink(missing_ok=True)
            with self._lock:
                self._saved_snapshots += 1
                self._last_error = None
        except Exception:
            snapshot_path.unlink(missing_ok=True)
            raise

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                assert isinstance(item, HumanLogEvent)
                self._write(item)
            except Exception as exc:
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                LOGGER.exception("Human log write failed")
            finally:
                self._queue.task_done()

    def list(
        self,
        *,
        camera: str | None = None,
        name: str | None = None,
        track_id: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        values: list[Any] = []
        for column, value in (("camera", camera), ("name", name), ("track_id", track_id)):
            if value is not None:
                conditions.append(f"{column} = ?")
                values.append(value)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        values.append(max(1, min(int(limit), 1000)))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, session_id, camera, track_id, name, first_seen,
                       last_seen, recognition_score, ref_img_id, snapshot_url
                FROM human_logs
                """
                + where
                + " ORDER BY id DESC LIMIT ?",
                values,
            ).fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM human_logs").fetchone()
        return int(row["count"])

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "count": self.count(),
                "queued": self._queue.qsize(),
                "queue_capacity": self._queue.maxsize,
                "dropped_events": self._dropped_events,
                "saved_snapshots": self._saved_snapshots,
                "snapshot_directory": str(self.snapshot_dir),
                "last_error": self._last_error,
            }

    def flush(self) -> None:
        """Wait until all currently queued writes are durable (primarily for tests)."""
        self._queue.join()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(_STOP)
        self._thread.join(timeout=10.0)
