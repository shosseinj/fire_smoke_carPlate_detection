from __future__ import annotations

from app.database import Connection, Database, IntegrityError, OperationalError, Row, ensure_database
from app.time_utils import utc_now_text

import json
import logging
import queue
import threading
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2

from app.core.media_utils import save_video_frames
from app.core.types import FramePacket, TaskName, TaskResult
from app.fire_core.policy import FireSmokePolicyConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _PendingEvent:
    frame: Any
    result: TaskResult
    video_frames: tuple[Any, ...]
    video_only: bool = False


class FireSmokeLogStore:
    """Persistent hazard events with bounded background media writes."""

    def __init__(
        self,
        database: Database | str,
        media_root: Path,
        *,
        default_policy: FireSmokePolicyConfig = FireSmokePolicyConfig(),
        queue_size: int = 64,
        video_fps: float = 5.0,
        video_max_frames: int = 30,
        video_update_interval_frames: int = 5,
    ) -> None:
        self.database = ensure_database(database)
        self.media_root = media_root.resolve()
        self.snapshot_dir = self.media_root / "fire_smoke_snapshots"
        self.video_dir = self.media_root / "fire_smoke_videos"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._policy_revision = 0
        self._queue: queue.Queue[_PendingEvent | None] = queue.Queue(maxsize=queue_size)
        self.video_fps = max(0.1, float(video_fps))
        self.video_max_frames = max(1, int(video_max_frames))
        self.video_update_interval_frames = max(1, int(video_update_interval_frames))
        self._video_buffers: dict[tuple[str, str], deque[Any]] = {}
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

    def _connect(self) -> Connection:
        return self.database.connection()

    def _initialize(self, default: FireSmokePolicyConfig) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO fire_smoke_settings (
                    id, window_seconds, low_count, medium_count, high_count, updated_at_utc
                ) VALUES (1, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING
                """,
                (
                    default.window_seconds,
                    default.low_count,
                    default.medium_count,
                    default.high_count,
                    utc_now_text(),
                ),
            )

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
        if self._closed or result.error or result.task != TaskName.FIRE_SMOKE:
            return
        ended_incidents = {
            str(event["incident_id"])
            for event in result.data.get("events", [])
            if event.get("event_type") == "incident_ended" and event.get("incident_id")
        }
        with self._lock:
            for incident_id in ended_incidents:
                frames = self._video_buffers.pop((result.source_id, incident_id), None)
                if frames:
                    self._queue_video_only(result, tuple(frames))

        incident_id = result.data.get("incident_id")
        if (
            result.data.get("severity") not in {"medium", "high"}
            or not incident_id
            or not self._has_hazard_detection(result)
        ):
            return
        key = (result.source_id, str(incident_id))
        frame = self._annotate_frame(packet.frame.copy(), result)
        with self._lock:
            buffer = self._video_buffers.setdefault(
                key,
                deque(maxlen=self.video_max_frames),
            )
            buffer.append(frame)
            should_queue = (
                bool(result.data.get("severity_changed", False))
                or len(buffer) % self.video_update_interval_frames == 0
            )
            video_frames = tuple(buffer)
        if not should_queue:
            return
        try:
            self._queue.put_nowait(_PendingEvent(frame, result, video_frames))
        except queue.Full:
            self._dropped += 1
            LOGGER.warning("Fire/smoke log queue is full; newest event was dropped")

    @staticmethod
    def _has_hazard_detection(result: TaskResult) -> bool:
        items = [
            *result.data.get("tracks", []),
            *result.data.get("detections", []),
        ]
        return any(item.get("label") in {"fire", "smoke"} for item in items)

    @staticmethod
    def _annotate_frame(frame: Any, result: TaskResult) -> Any:
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
        return frame

    def _queue_video_only(
        self,
        result: TaskResult,
        video_frames: tuple[Any, ...],
    ) -> None:
        try:
            self._queue.put_nowait(
                _PendingEvent(video_frames[-1], result, video_frames, video_only=True)
            )
        except queue.Full:
            self._dropped += 1
            LOGGER.warning("Fire/smoke video queue is full; final frames were dropped")

    @staticmethod
    def _incident_id(result: TaskResult) -> str | None:
        if result.data.get("incident_id"):
            return str(result.data["incident_id"])
        for event in result.data.get("events", []):
            if event.get("incident_id"):
                return str(event["incident_id"])
        return None

    def _save(self, pending: _PendingEvent) -> None:
        result = pending.result
        fire = dict(result.data.get("fire") or {})
        smoke = dict(result.data.get("smoke") or {})
        incident_id = self._incident_id(result)
        existing: Row | None = None
        if incident_id:
            with self._lock, self._connect() as connection:
                existing = connection.execute(
                    "SELECT id, severity, snapshot_url, video_url FROM fire_smoke_logs "
                    "WHERE incident_id = ? ORDER BY id LIMIT 1",
                    (incident_id,),
                ).fetchone()
        if pending.video_only:
            if existing is None or not pending.video_frames:
                return
            camera_stem = result.source_id.replace("/", "_").replace("\\", "_")
            timestamp = result.processed_at_utc.replace(":", "-").replace("+", "_")
            video_filename = f"{camera_stem}_{timestamp}_{uuid4().hex[:8]}.mp4"
            video_path = self.video_dir / video_filename
            save_video_frames(pending.video_frames, video_path, self.video_fps)
            video_url = f"/media/fire_smoke_videos/{video_filename}"
            with self._lock, self._connect() as connection:
                connection.execute(
                    "UPDATE fire_smoke_logs SET video_url = ? WHERE id = ?",
                    (video_url, int(existing["id"])),
                )
                connection.commit()
            if existing["video_url"]:
                old_video = self.video_dir / Path(str(existing["video_url"])).name
                if old_video != video_path:
                    old_video.unlink(missing_ok=True)
            return
        severity_rank = {"medium": 2, "high": 3}
        if existing is not None:
            existing_rank = severity_rank.get(str(existing["severity"]), 0)
            result_rank = severity_rank.get(str(result.data.get("severity")), 0)
            if existing_rank > result_rank:
                return
            if existing_rank == result_rank and len(pending.video_frames) <= 1:
                return
        frame = pending.frame
        timestamp = result.processed_at_utc.replace(":", "-").replace("+", "_")
        camera_stem = result.source_id.replace("/", "_").replace("\\", "_")
        filename = f"{camera_stem}_{timestamp}_{uuid4().hex[:8]}.jpg"
        snapshot_path = self.snapshot_dir / filename
        if not cv2.imwrite(str(snapshot_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
            raise RuntimeError(f"Could not save fire/smoke snapshot: {snapshot_path}")
        snapshot_url = f"/media/fire_smoke_snapshots/{filename}"
        video_filename = f"{Path(filename).stem}.mp4"
        video_path = self.video_dir / video_filename
        try:
            save_video_frames(pending.video_frames or (frame,), video_path, self.video_fps)
        except Exception:
            snapshot_path.unlink(missing_ok=True)
            raise
        video_url = f"/media/fire_smoke_videos/{video_filename}"
        values = (
            result.source_id,
            result.processed_at_utc,
            incident_id,
            result.data["severity"],
            int(fire.get("positive_count", 0)),
            int(smoke.get("positive_count", 0)),
            float(fire.get("max_confidence", 0.0)),
            float(smoke.get("max_confidence", 0.0)),
            float(result.data.get("severity_window_seconds", 3.0)),
            snapshot_url,
            video_url,
            json.dumps(result.data, ensure_ascii=False, sort_keys=True),
        )
        with self._lock, self._connect() as connection:
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO fire_smoke_logs (
                        camera, time, incident_id, severity, fire_count, smoke_count,
                        fire_confidence, smoke_confidence, window_seconds,
                        snapshot_url, video_url, details_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
            else:
                connection.execute(
                    """
                    UPDATE fire_smoke_logs
                    SET camera = ?, time = ?, incident_id = ?, severity = ?,
                        fire_count = ?, smoke_count = ?, fire_confidence = ?,
                        smoke_confidence = ?, window_seconds = ?, snapshot_url = ?,
                        video_url = ?, details_json = ?
                    WHERE id = ?
                    """,
                    (*values, int(existing["id"])),
                )
            connection.commit()
            self._saved += 1
        if existing is not None and existing["snapshot_url"]:
            old_path = self.snapshot_dir / Path(str(existing["snapshot_url"])).name
            if old_path != snapshot_path:
                old_path.unlink(missing_ok=True)
        if existing is not None and existing["video_url"]:
            old_video = self.video_dir / Path(str(existing["video_url"])).name
            if old_video != video_path:
                old_video.unlink(missing_ok=True)

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
        hazard_type: str | None = None,
        detected_from: str | None = None,
        detected_to: str | None = None,
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
        if hazard_type:
            clauses.append("details_json LIKE ?")
            parameters.append(f'%"hazard_type": "{hazard_type.strip()}"%')
        if detected_from:
            clauses.append("time >= ?")
            parameters.append(detected_from)
        if detected_to:
            clauses.append("time <= ?")
            parameters.append(detected_to)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(max(1, min(int(limit), 1000)))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT id, camera, time, incident_id, severity, fire_count, "
                "smoke_count, fire_confidence, smoke_confidence, window_seconds, "
                f"snapshot_url, video_url FROM fire_smoke_logs{where} ORDER BY id DESC LIMIT ?",
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM fire_smoke_logs").fetchone()
        return int(row["count"] if row else 0)

    def get(self, log_id: int) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT id, camera, time, incident_id, severity, fire_count, smoke_count, "
                "fire_confidence, smoke_confidence, window_seconds, snapshot_url, video_url "
                "FROM fire_smoke_logs WHERE id = ?",
                (log_id,),
            ).fetchone()
        return dict(row) if row else None

    def create_manual(self, values: dict[str, Any]) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO fire_smoke_logs (camera, time, severity, fire_count, smoke_count, "
                "fire_confidence, smoke_confidence, window_seconds, snapshot_url, video_url, details_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
                (
                    str(values["camera_id"]), values["detection_time"].isoformat(), values["severity"],
                    int(values.get("fire_count", 0)), int(values.get("smoke_count", 0)),
                    float(values.get("confidence") or 0.0), float(values.get("confidence") or 0.0),
                    0.0, values.get("snapshot_url") or "", values.get("video_url") or "", "{}",
                ),
            )
            log_id = int(cursor.fetchone()[0])
            connection.commit()
        return self.get(log_id) or {}

    def update_manual(self, log_id: int, values: dict[str, Any]) -> dict[str, Any] | None:
        mapping = {
            "camera_id": "camera", "detection_time": "time", "severity": "severity",
            "snapshot_url": "snapshot_url", "video_url": "video_url",
        }
        updates = [(mapping[key], value.isoformat() if isinstance(value, datetime) else value) for key, value in values.items() if key in mapping]
        if not updates:
            return self.get(log_id)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE fire_smoke_logs SET {', '.join(f'{key} = ?' for key, _ in updates)} WHERE id = ?",
                [value for _, value in updates] + [log_id],
            )
            connection.commit()
            if cursor.rowcount == 0:
                return None
        return self.get(log_id)

    def delete(self, log_id: int) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM fire_smoke_logs WHERE id = ?", (log_id,))
            connection.commit()
            return cursor.rowcount > 0

    def status(self) -> dict[str, Any]:
        return {
            "count": self.count(),
            "queued": self._queue.qsize(),
            "video_buffer_sources": len(self._video_buffers),
            "video_buffer_frames": sum(len(value) for value in self._video_buffers.values()),
            "video_max_frames": self.video_max_frames,
            "saved": self._saved,
            "dropped": self._dropped,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            self._video_buffers.clear()
        self._queue.put(None)
        self._thread.join(timeout=10.0)
