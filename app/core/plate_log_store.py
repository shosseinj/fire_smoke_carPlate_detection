from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2

from app.core.media_utils import save_single_frame_video
from app.core.plate_constants import normalize_plate_full_number
from app.core.types import FramePacket, TaskName, TaskResult
from app.database import Connection, Database, ensure_database


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _PendingPlateEvent:
    packet: FramePacket
    result: TaskResult


class PlateLogStore:
    """Bounded asynchronous persistence for normalized plate detections."""

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
            target=self._run, name="plate-log-writer", daemon=True
        )
        self._thread.start()

    def _connect(self) -> Connection:
        return self.database.connection()

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _media_key(value: str | None) -> str | None:
        text = str(value or "").strip().replace("\\", "/")
        if not text:
            return None
        if text.startswith("/media/"):
            return text[len("/media/") :]
        return text.lstrip("/")

    @staticmethod
    def _media_url(value: str | None) -> str | None:
        key = PlateLogStore._media_key(value)
        return f"/media/{key}" if key else None

    def _source_identity(self, source_uri: str) -> tuple[str, str | None, int | None]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM static_videos WHERE source_uri = ?", (source_uri,)
            ).fetchone()
        if row is not None:
            return "static_video", None, int(row["id"])
        return "camera", source_uri, None

    def _registered_plate_id(self, plate_number: str) -> int | None:
        """Resolve an exact normalized OCR value to one active registered plate."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id
                FROM car_plates
                WHERE deleted_at_utc IS NULL
                  AND is_active = 1
                  AND CONCAT(left_digits, plate_alphabet, right_digits, iran_code) = ?
                ORDER BY id
                LIMIT 1
                """,
                (plate_number,),
            ).fetchone()
        return int(row["id"]) if row is not None else None

    def insert_result(self, packet: FramePacket, result: TaskResult) -> int:
        if result.error or result.task != TaskName.PLATE_RECOGNITION:
            return 0
        ocr_results = result.data.get("plate_ocr_results")
        candidates = ocr_results if ocr_results is not None else result.data.get("plates", [])
        if not candidates:
            return 0
        if packet.frame is None or packet.frame.size == 0:
            raise ValueError("فریم تصویر خالی است")

        source_type, source_uri, static_video_id = self._source_identity(result.source_id)
        captured_at = packet.captured_at_utc
        detected_datetime = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        timestamp = detected_datetime.strftime("%Y%m%d_%H%M%S_%f")
        safe_source = "".join(
            char if char.isalnum() or char in "-_" else "_"
            for char in str(result.source_id)
        )
        stem = f"{safe_source}_{result.frame_index}_{timestamp}_{uuid4().hex[:8]}"
        snapshot_dir = self.media_root / "plate" / "snapshots"
        video_dir = self.media_root / "plate" / "videos"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / f"{stem}.jpg"
        video_path = video_dir / f"{stem}.mp4"
        snapshot_key = f"plate/snapshots/{snapshot_path.name}" if self.save_plate_snapshot else None
        video_key = f"plate/videos/{video_path.name}"
        snapshot_frame = packet.frame.copy()

        records: list[tuple[Any, ...]] = []
        now = self._now_utc()
        for item in candidates:
            if isinstance(item, dict):
                plate_value = (
                    item.get("plate_number")
                    if "plate_number" in item
                    else item.get("plate")
                )
                raw_value = (
                    item.get("raw_plate_text")
                    if "raw_plate_text" in item
                    else item.get("plate")
                )
                raw_text = None if raw_value is None else str(raw_value)
                bbox = item.get("bbox")
                confidence = item.get("recognizer_confidence")
            else:
                plate_value = item
                raw_text = str(item) if item is not None else None
                bbox = None
                confidence = None
            plate_number = (
                normalize_plate_full_number(str(plate_value))
                if plate_value is not None and str(plate_value).strip()
                else None
            )
            if plate_number is None and raw_text is None:
                continue
            plate_id = (
                self._registered_plate_id(plate_number) if plate_number else None
            )
            if self.draw_info and bbox and len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                height, width = snapshot_frame.shape[:2]
                x1, x2 = sorted((max(0, min(x1, width - 1)), max(0, min(x2, width - 1))))
                y1, y2 = sorted((max(0, min(y1, height - 1)), max(0, min(y2, height - 1))))
                cv2.rectangle(snapshot_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    snapshot_frame,
                    plate_number or raw_text or "",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
                )
            records.append(
                (
                    source_type, source_uri, static_video_id, plate_id,
                    plate_number, raw_text,
                    float(confidence) if confidence is not None else None, captured_at,
                    snapshot_key, video_key, now, now,
                )
            )
        if not records:
            return 0

        if self.save_plate_snapshot and not cv2.imwrite(
            str(snapshot_path), snapshot_frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
        ):
            raise RuntimeError(f"Plate snapshot could not be saved: {snapshot_path}")
        try:
            save_single_frame_video(snapshot_frame, video_path)
            with self._lock, self._connect() as connection:
                connection.executemany(
                    """
                    INSERT INTO plate_logs (
                        source_type, source_uri, static_video_id, plate_id,
                        plate_number, raw_plate_text, confidence, detection_time,
                        snapshot_key, video_key, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        if (
            self._closed
            or result.error
            or result.task != TaskName.PLATE_RECOGNITION
            or not (
                result.data.get("plate_ocr_results")
                or result.data.get("plates")
            )
        ):
            return
        try:
            self._queue.put_nowait(
                _PendingPlateEvent(replace(packet, frame=packet.frame.copy()), result)
            )
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
                self._last_error = None
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

    def count(self) -> int:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM plate_logs").fetchone()
        return int(row["count"] if row is not None else 0)

    def list_logs(
        self,
        *,
        plate_id: int | None = None,
        plate_number: str | None = None,
        source_uri: str | None = None,
        static_video_id: int | None = None,
        source_type: str | None = None,
        detected_from: str | None = None,
        detected_to: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses = ["1=1"]
        params: list[Any] = []
        for column, value in (
            ("plate_id", plate_id), ("plate_number", plate_number),
            ("source_uri", source_uri), ("static_video_id", static_video_id),
            ("source_type", source_type),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if detected_from:
            clauses.append("detection_time >= ?")
            params.append(detected_from)
        if detected_to:
            clauses.append("detection_time <= ?")
            params.append(detected_to)
        params.extend([max(0, skip), max(1, min(limit, 500))])
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM plate_logs WHERE " + " AND ".join(clauses)
                + " ORDER BY detection_time DESC, id DESC OFFSET ? LIMIT ?",
                params,
            ).fetchall()
        return [self._serialize(dict(row)) for row in rows]

    def get_log(self, log_id: int) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM plate_logs WHERE id = ?", (log_id,)
            ).fetchone()
        return self._serialize(dict(row)) if row else None

    @staticmethod
    def _validate_source_identity(fields: dict[str, Any]) -> None:
        source_type = fields.get("source_type")
        source_uri = fields.get("source_uri")
        static_video_id = fields.get("static_video_id")
        created_by = fields.get("created_by_user_id")
        valid = (
            source_type == "camera" and bool(source_uri) and static_video_id is None and created_by is None
        ) or (
            source_type == "static_video" and not source_uri and static_video_id is not None and created_by is None
        ) or (
            source_type == "manual" and not source_uri and static_video_id is None and created_by is not None
        )
        if not valid:
            raise ValueError("Invalid plate log source identity")

    def create_log(self, values: dict[str, Any]) -> dict[str, Any]:
        fields = dict(values)
        self._validate_source_identity(fields)
        now = self._now_utc()
        fields.setdefault("created_at", now)
        fields.setdefault("updated_at", now)
        fields["snapshot_key"] = self._media_key(fields.get("snapshot_key"))
        fields["video_key"] = self._media_key(fields.get("video_key"))
        columns = list(fields)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"INSERT INTO plate_logs ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)}) RETURNING id",
                [fields[name] for name in columns],
            ).fetchone()
            connection.commit()
        return self.get_log(int(row["id"])) or {}

    def update_log(self, log_id: int, values: dict[str, Any]) -> dict[str, Any] | None:
        if not values:
            return self.get_log(log_id)
        allowed = {
            "plate_id", "plate_number", "raw_plate_text", "confidence",
            "detection_time", "snapshot_key", "video_key", "notes",
            "updated_by_user_id",
        }
        unexpected = set(values) - allowed
        if unexpected:
            raise ValueError(f"Unsupported plate-log fields: {sorted(unexpected)}")
        fields = dict(values)
        for name in ("snapshot_key", "video_key"):
            if name in fields:
                fields[name] = self._media_key(fields[name])
        assignments = [f"{name} = ?" for name in fields] + ["updated_at = ?"]
        params = [fields[name] for name in fields] + [self._now_utc(), log_id]
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE plate_logs SET {', '.join(assignments)} WHERE id = ?", params
            )
            connection.commit()
        return self.get_log(log_id) if cursor.rowcount else None

    def delete_log(self, log_id: int) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM plate_logs WHERE id = ?", (log_id,))
            connection.commit()
        return cursor.rowcount > 0

    @classmethod
    def _serialize(cls, value: dict[str, Any]) -> dict[str, Any]:
        value["snapshot_url"] = cls._media_url(value.get("snapshot_key"))
        value["video_url"] = cls._media_url(value.get("video_key"))
        return value

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=10.0)
