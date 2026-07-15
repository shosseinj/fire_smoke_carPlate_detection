from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any
from datetime import datetime
from app.core.types import TaskName, TaskResult
from fastapi import APIRouter, Depends, HTTPException, status
from app.core.types import FramePacket
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2

from app.core.types import FramePacket, TaskName, TaskResult



class PlateLogStore:
    """Persistent plate detection logs backed by SQLite."""

    def __init__(self, database_path: Path, draw_info:bool, save_plate_snapshot:bool) -> None:
        self.database_path = database_path
        self.draw_info = draw_info
        self.save_plate_snapshot = save_plate_snapshot
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=10.0,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS plate_logs (
                    camera TEXT NOT NULL,
                    time TEXT NOT NULL,
                    plate TEXT NOT NULL,
                    snapshot_url TEXT NOT NULL DEFAULT ''
                )
                """
            )

            # Add snapshot_url when the database was created
            # using the older three-column schema.
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(plate_logs)"
                ).fetchall()
            }

            if "snapshot_url" not in columns:
                connection.execute(
                    """
                    ALTER TABLE plate_logs
                    ADD COLUMN snapshot_url TEXT NOT NULL DEFAULT ''
                    """
                )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_plate_logs_time
                ON plate_logs(time)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_plate_logs_camera
                ON plate_logs(camera)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_plate_logs_plate
                ON plate_logs(plate)
                """
            )

            connection.commit()

    def insert(
        self,
        *,
        camera: str,
        time: str,
        plate: str,
        snapshot_url: str,
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
                    snapshot_url
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    camera,
                    detected_at,
                    plate,
                    snapshot_url,
                ),
            )

            connection.commit()

        return {
            "camera": camera,
            "time": detected_at,
            "plate": plate,
            "snapshot_url": snapshot_url,
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

        snapshot_directory = Path("saved_media") / "plate_snapshots"
        snapshot_directory.mkdir(parents=True, exist_ok=True)

        snapshot_path = snapshot_directory / file_name
        snapshot_url = f"/media/plate_snapshots/{file_name}"

        if packet.frame is None or packet.frame.size == 0:
            raise ValueError("Snapshot frame is empty")

        # Copy prevents changing the original frame.
        snapshot_frame = packet.frame.copy()

        records: list[tuple[str, str, str, str]] = []

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
            with self._lock, self._connect() as connection:
                connection.executemany(
                    """
                    INSERT INTO plate_logs (
                        camera,
                        time,
                        plate,
                        snapshot_url
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    records,
                )
                connection.commit()

        except Exception:
            snapshot_path.unlink(missing_ok=True)
            raise

        return len(records)
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
                    snapshot_url
                FROM plate_logs
                {where_clause}
                ORDER BY rowid DESC
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