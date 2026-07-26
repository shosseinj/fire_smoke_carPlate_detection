from __future__ import annotations

from app.database import Connection, Database, IntegrityError, OperationalError, Row, ensure_database

import logging
import queue
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from app.core.types import TaskName, TaskResult
from fastapi import APIRouter, Depends, HTTPException, status
from app.core.types import FramePacket
from uuid import uuid4

import cv2

from app.core.types import FramePacket, TaskName, TaskResult
from app.core.media_utils import save_single_frame_video

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _PendingPlateEvent:
    packet: FramePacket
    result: TaskResult



class PlateLogStore:
    """Persistent plate detection logs backed by PostgreSQL."""

    def __init__(
        self,
        database: Database | str,
        draw_info: bool,
        save_plate_snapshot: bool,
        queue_size: int = 128,
        media_root: Path | None = None,
    ) -> None:
        self.database = ensure_database(database)
        self.draw_info = draw_info
        self.save_plate_snapshot = save_plate_snapshot
        self.media_root = (media_root or Path("saved_media")).resolve()
        self._lock = threading.RLock()
        self._queue: queue.Queue[_PendingPlateEvent | None] = queue.Queue(
            maxsize=max(8, int(queue_size))
        )
        self._dropped = 0
        self._saved = 0
        self._last_error: str | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="plate-log-writer",
            daemon=True,
        )
        self._initialize()
        self._thread.start()

    def _connect(self) -> Connection:
        return self.database.connection()

    def _initialize(self) -> None:
        # Alembic owns the PostgreSQL schema; runtime startup validates it.
        return None

    def insert(
        self,
        *,
        camera: str,
        time: str,
        plate: str,
        snapshot_url: str,
        video_url: str = "",
    ) -> dict[str, str]:
        camera = camera.strip()
        detected_at = time.strip()
        plate = plate.strip()
        snapshot_url = snapshot_url.strip()

        if not camera:
            raise ValueError("camera must not be empty")

        if not detected_at:
            raise ValueError("time must not be empty")

        if not plate:
            raise ValueError("plate must not be empty")

        if not snapshot_url:
            raise ValueError("snapshot_url must not be empty")

        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO plate_logs (
                    camera,
                    time,
                    plate,
                    snapshot_url,
                    video_url
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    camera,
                    detected_at,
                    plate,
                    snapshot_url,
                    video_url,
                ),
            )

            connection.commit()

        return {
            "camera": camera,
            "time": detected_at,
            "plate": plate,
            "snapshot_url": snapshot_url,
            "video_url": video_url,
        }

    
    

    def insert_result(
        self,
        packet: FramePacket,
        result: TaskResult,
    ) -> int:
        if result.error:
            return 0

        if result.task != TaskName.PLATE_RECOGNITION:
            return 0

        plates = result.data.get("plates", [])

        if not plates:
            return 0

        detected_datetime = datetime.fromisoformat(
            result.processed_at_utc.replace("Z", "+00:00")
        )
        timestamp = detected_datetime.strftime("%Y%m%d_%H%M%S_%f")

        safe_camera_id = "".join(
            character
            if character.isalnum() or character in "-_"
            else "_"
            for character in str(result.source_id)
        )

        file_name = (
            f"{safe_camera_id}_"
            f"{result.frame_index}_"
            f"{timestamp}_"
            f"{uuid4().hex[:8]}.jpg"
        )

        snapshot_directory = self.media_root / "plate_snapshots"
        video_directory = self.media_root / "plate_videos"
        snapshot_directory.mkdir(parents=True, exist_ok=True)

        snapshot_path = snapshot_directory / file_name
        snapshot_url = f"/media/plate_snapshots/{file_name}"
        video_path = video_directory / f"{Path(file_name).stem}.mp4"
        video_url = f"/media/plate_videos/{video_path.name}"

        if packet.frame is None or packet.frame.size == 0:
            raise ValueError("Snapshot frame is empty")

        # Copy prevents changing the original frame.
        snapshot_frame = packet.frame.copy()

        records: list[tuple[str, str, str, str, str]] = []

        for item in plates:
            if isinstance(item, dict):
                plate = str(item.get("plate") or "").strip()
                bbox = item.get("bbox")
            else:
                plate = str(item or "").strip()
                bbox = None

            if not plate:
                continue

            if self.draw_info and bbox and len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)

                frame_height, frame_width = snapshot_frame.shape[:2]

                x1 = max(0, min(x1, frame_width - 1))
                y1 = max(0, min(y1, frame_height - 1))
                x2 = max(0, min(x2, frame_width - 1))
                y2 = max(0, min(y2, frame_height - 1))

                cv2.rectangle(
                    snapshot_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    snapshot_frame,
                    plate,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            records.append(
                (
                    result.source_id,
                    result.processed_at_utc,
                    plate,
                    snapshot_url,
                    video_url,
                )
            )

        if not records:
            return 0
        if self.save_plate_snapshot:
            snapshot_saved = cv2.imwrite(
                str(snapshot_path),
                snapshot_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            print('Plate snapshot saved')
            if not snapshot_saved:
                raise RuntimeError(
                    f"Plate snapshot could not be saved: {snapshot_path}"
                )
        try:
            save_single_frame_video(snapshot_frame, video_path)
        except Exception:
            snapshot_path.unlink(missing_ok=True)
            raise

        try:
            with self._lock, self._connect() as connection:
                connection.executemany(
                    """
                    INSERT INTO plate_logs (
                        camera,
                        time,
                        plate,
                        snapshot_url,
                        video_url
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    records,
                )
                connection.commit()

        except Exception:
            snapshot_path.unlink(missing_ok=True)
            video_path.unlink(missing_ok=True)
            raise

        return len(records)

    def observe_result(self, packet: FramePacket, result: TaskResult) -> None:
        """Queue plate persistence without blocking the inference worker."""
        if (
            self._closed
            or result.error
            or result.task != TaskName.PLATE_RECOGNITION
            or not result.data.get("plates")
        ):
            return
        try:
            queued_packet = replace(packet, frame=packet.frame.copy())
            self._queue.put_nowait(_PendingPlateEvent(queued_packet, result))
        except queue.Full:
            self._dropped += 1
            LOGGER.warning("Plate log queue is full; newest result was dropped")

    def _run(self) -> None:
        while True:
            pending = self._queue.get()
            try:
                if pending is None:
                    return
                self._saved += self.insert_result(pending.packet, pending.result)
            except Exception:
                self._last_error = "plate persistence failed"
                LOGGER.exception("Plate persistence failed")
            finally:
                self._queue.task_done()

    def status(self) -> dict[str, Any]:
        return {
            "queued": self._queue.qsize(),
            "queue_capacity": self._queue.maxsize,
            "saved": self._saved,
            "dropped": self._dropped,
            "last_error": self._last_error,
        }
    def list(
        self,
        *,
        camera: str | None = None,
        plate: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, str]]:
        clauses: list[str] = []
        parameters: list[Any] = []

        if camera:
            clauses.append("camera = ?")
            parameters.append(camera.strip())

        if plate:
            clauses.append("plate LIKE ?")
            parameters.append(f"%{plate.strip()}%")

        where_clause = (
            f" WHERE {' AND '.join(clauses)}"
            if clauses
            else ""
        )

        safe_limit = max(1, min(int(limit), 1000))
        parameters.append(safe_limit)

        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    camera,
                    time,
                    plate,
                    snapshot_url,
                    video_url
                FROM plate_logs
                {where_clause}
                ORDER BY id DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()

        return [dict(row) for row in rows]

    def count(self) -> int:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM plate_logs
                """
            ).fetchone()

        return int(
            row["count"]
            if row is not None
            else 0
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=10.0)

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).isoformat()

    def list_logs(
        self,
        *,
        plate_id: int | None = None,
        plate_full_number: str | None = None,
        camera_id: str | None = None,
        direction: str | None = None,
        source_type: str | None = None,
        is_verified: bool | None = None,
        detected_from: str | None = None,
        detected_to: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = ["1=1"]
        params: list[Any] = []
        if plate_id is not None:
            clauses.append("plate_id = ?")
            params.append(plate_id)
        if plate_full_number:
            clauses.append("plate_full_number = ?")
            params.append(plate_full_number)
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if direction:
            clauses.append("direction = ?")
            params.append(direction)
        if source_type:
            clauses.append("source_type = ?")
            params.append(source_type)
        if is_verified is not None:
            clauses.append("is_verified = ?")
            params.append(1 if is_verified else 0)
        if detected_from:
            clauses.append("detection_time >= ?")
            params.append(detected_from)
        if detected_to:
            clauses.append("detection_time <= ?")
            params.append(detected_to)
        params.extend([max(0, skip), max(1, min(limit, 500))])
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM plate_logs WHERE " + " AND ".join(clauses) + " ORDER BY detection_time DESC, id DESC OFFSET ? LIMIT ?",
                params,
            ).fetchall()
        return [self._serialize_plate_log(dict(row)) for row in rows]

    def get_log(self, log_id: int) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM plate_logs WHERE id = ?", (log_id,)).fetchone()
        return self._serialize_plate_log(dict(row)) if row else None

    def create_log(self, values: dict[str, Any]) -> dict[str, Any]:
        now = self._now_utc()
        fields = dict(values)
        fields.setdefault("source_type", "camera")
        fields.setdefault("direction", "unknown")
        fields.setdefault("created_at", now)
        fields.setdefault("updated_at", now)
        columns = list(fields)
        params = [fields[name] for name in fields]
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                f"INSERT INTO plate_logs ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)}) RETURNING id",
                params,
            )
            log_id = int(cursor.fetchone()[0])
            connection.commit()
        return self.get_log(log_id) or {}

    def update_log(self, log_id: int, values: dict[str, Any]) -> dict[str, Any] | None:
        if not values:
            return self.get_log(log_id)
        assignments = [f"{name} = ?" for name in values] + ["updated_at = ?"]
        params = [values[name] for name in values] + [self._now_utc(), log_id]
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE plate_logs SET {', '.join(assignments)} WHERE id = ?",
                params,
            )
            connection.commit()
            if cursor.rowcount == 0:
                return None
        return self.get_log(log_id)

    def delete_log(self, log_id: int) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM plate_logs WHERE id = ?", (log_id,))
            connection.commit()
            return cursor.rowcount > 0

    @staticmethod
    def _serialize_plate_log(value: dict[str, Any]) -> dict[str, Any]:
        legacy_aliases = {
            "plate_full_number": "plate",
            "camera_id": "camera",
            "detection_time": "time",
            "snapshot_path": "snapshot_url",
        }
        for target, source in legacy_aliases.items():
            if value.get(target) is None:
                value[target] = value.get(source)
        if "is_verified" in value:
            value["is_verified"] = bool(value["is_verified"])
        return value
