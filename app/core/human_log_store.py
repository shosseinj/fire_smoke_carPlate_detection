from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
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
class HumanMediaEvent:
    session_id: str
    camera: str
    track_id: int
    name: str
    captured_at_utc: str
    recognition_score: float
    ref_img_id: str | int | None
    snapshot_frame: np.ndarray | None
    snapshot_quality: float
    human_video_frame: np.ndarray | None
    face_video_frame: np.ndarray | None
    face_quality: float
    face_yaw: float | None
    face_pitch: float | None
    face_roll: float | None


@dataclass(slots=True)
class TrackMediaState:
    human_writer: cv2.VideoWriter | None = None
    face_writer: cv2.VideoWriter | None = None
    video_url: str = ""
    face_video_url: str = ""
    last_event_monotonic: float = 0.0


class HumanLogStore:
    """Non-blocking per-ByteTrack database, best snapshot, and video recorder."""

    HUMAN_VIDEO_SIZE = (256, 384)
    FACE_VIDEO_SIZE = (224, 224)

    def __init__(
        self,
        database_path: Path,
        saved_media_path: Path,
        *,
        queue_size: int = 128,
        video_fps: float = 10.0,
        video_idle_seconds: float = 5.0,
        snapshot_min_improvement: float = 0.01,
    ) -> None:
        self.database_path = database_path.resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        media_root = saved_media_path.resolve()
        self.snapshot_dir = media_root / "human_snapshots"
        self.video_dir = media_root / "human_videos"
        self.face_video_dir = media_root / "human_face_videos"
        for directory in (self.snapshot_dir, self.video_dir, self.face_video_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.video_fps = max(1.0, float(video_fps))
        self.video_idle_seconds = max(1.0, float(video_idle_seconds))
        self.snapshot_min_improvement = max(0.0, float(snapshot_min_improvement))
        self._queue: queue.Queue[HumanMediaEvent | object] = queue.Queue(
            maxsize=max(8, int(queue_size))
        )
        self._observed_names: dict[tuple[str, str, int], str] = {}
        self._candidate_scores: dict[tuple[str, str, int], float] = {}
        self._last_video_at: dict[tuple[str, str, int], float] = {}
        self._media: dict[tuple[str, str, int], TrackMediaState] = {}
        self._lock = threading.RLock()
        self._dropped_events = 0
        self._saved_snapshots = 0
        self._human_video_frames = 0
        self._accepted_face_video_frames = 0
        self._created_human_videos = 0
        self._created_face_videos = 0
        self._last_error: str | None = None
        self._closed = False
        self._create_schema()
        self._thread = threading.Thread(
            target=self._run,
            name="human-media-writer",
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
                    video_url TEXT NOT NULL DEFAULT '',
                    face_video_url TEXT NOT NULL DEFAULT '',
                    snapshot_quality REAL NOT NULL DEFAULT 0,
                    best_face_quality REAL NOT NULL DEFAULT 0,
                    best_face_yaw REAL,
                    best_face_pitch REAL,
                    best_face_roll REAL,
                    human_video_frames INTEGER NOT NULL DEFAULT 0,
                    accepted_face_frames INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(session_id, camera, track_id)
                )
                """
            )
            existing = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(human_logs)").fetchall()
            }
            additions = {
                "video_url": "TEXT NOT NULL DEFAULT ''",
                "face_video_url": "TEXT NOT NULL DEFAULT ''",
                "snapshot_quality": "REAL NOT NULL DEFAULT 0",
                "best_face_quality": "REAL NOT NULL DEFAULT 0",
                "best_face_yaw": "REAL",
                "best_face_pitch": "REAL",
                "best_face_roll": "REAL",
                "human_video_frames": "INTEGER NOT NULL DEFAULT 0",
                "accepted_face_frames": "INTEGER NOT NULL DEFAULT 0",
            }
            for column, definition in additions.items():
                if column not in existing:
                    connection.execute(
                        f"ALTER TABLE human_logs ADD COLUMN {column} {definition}"
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

    @staticmethod
    def _bounded_box(
        frame: np.ndarray,
        bbox: list[float],
        *,
        padding_ratio: float = 0.0,
    ) -> tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (float(value) for value in bbox)
        pad_x = (x2 - x1) * padding_ratio
        pad_y = (y2 - y1) * padding_ratio
        return (
            max(0, min(int(round(x1 - pad_x)), width - 1)),
            max(0, min(int(round(y1 - pad_y)), height - 1)),
            max(1, min(int(round(x2 + pad_x)), width)),
            max(1, min(int(round(y2 + pad_y)), height)),
        )

    @classmethod
    def _human_crop(cls, frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
        x1, y1, x2, y2 = cls._bounded_box(frame, bbox, padding_ratio=0.05)
        crop = frame[y1:y2, x1:x2]
        return crop.copy() if crop.size else None

    @staticmethod
    def _letterbox(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        target_width, target_height = size
        height, width = image.shape[:2]
        scale = min(target_width / max(width, 1), target_height / max(height, 1))
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height))
        output = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        x = (target_width - resized_width) // 2
        y = (target_height - resized_height) // 2
        output[y : y + resized_height, x : x + resized_width] = resized
        return output

    @classmethod
    def _aligned_face(
        cls,
        frame: np.ndarray,
        bbox: list[float],
        landmarks: list[list[float]],
    ) -> np.ndarray | None:
        values = np.asarray(landmarks, dtype=np.float32)
        if values.shape[0] < 5:
            return None
        reference = 2.0 * np.asarray(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        transform, _ = cv2.estimateAffinePartial2D(
            values[:5],
            reference,
            method=cv2.LMEDS,
        )
        if transform is None:
            return None
        return cv2.warpAffine(frame, transform, cls.FACE_VIDEO_SIZE)

    def observe_result(self, packet: FramePacket, result: TaskResult) -> None:
        if result.error:
            return
        session_id = str(result.data.get("tracking_session_id") or "unknown-session")
        faces_by_track: dict[int, dict[str, Any]] = {}
        for face in result.data.get("faces", []):
            track_id = face.get("track_id")
            if track_id is None or not bool(face.get("quality_valid")):
                continue
            existing = faces_by_track.get(int(track_id))
            if existing is None or float(face.get("quality_score", 0.0)) > float(
                existing.get("quality_score", 0.0)
            ):
                faces_by_track[int(track_id)] = face

        for human in result.data.get("humans", []):
            track_id = human.get("track_id")
            if track_id is None:
                continue
            track_id = int(track_id)
            key = (session_id, packet.source_id, track_id)
            name = str(human.get("person") or "Unknown").strip() or "Unknown"
            bbox = [float(value) for value in human.get("bbox", [0, 0, 0, 0])[:4]]
            human_crop = self._human_crop(packet.frame, bbox)
            if human_crop is None:
                continue
            face = faces_by_track.get(track_id)
            face_quality = float(face.get("quality_score", 0.0)) if face else 0.0
            frame_area = max(1.0, float(packet.frame.shape[0] * packet.frame.shape[1]))
            human_area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            if face is not None:
                snapshot_quality = 0.70 + 0.30 * face_quality
            else:
                snapshot_quality = min(
                    0.60,
                    0.35 * float(human.get("confidence", 0.0) or 0.0)
                    + 0.25 * min(1.0, human_area / frame_area * 5.0),
                )
            now = float(packet.captured_monotonic)
            with self._lock:
                previous_name = self._observed_names.get(key)
                previous_score = self._candidate_scores.get(key, -1.0)
                previous_video_at = self._last_video_at.get(key, float("-inf"))
                identity_changed = previous_name is None or (
                    previous_name == "Unknown" and name != "Unknown"
                )
                better_snapshot = (
                    snapshot_quality
                    >= previous_score + self.snapshot_min_improvement
                )
                video_due = now - previous_video_at >= 1.0 / self.video_fps
                if not (identity_changed or better_snapshot or video_due):
                    continue
                self._observed_names[key] = name
                if better_snapshot:
                    self._candidate_scores[key] = snapshot_quality
                if video_due:
                    self._last_video_at[key] = now

            quality_metrics = dict(face.get("quality_metrics", {})) if face else {}
            face_frame = None
            if face is not None and video_due:
                face_frame = self._aligned_face(
                    packet.frame,
                    [float(value) for value in face.get("bbox", [0, 0, 0, 0])[:4]],
                    list(face.get("landmarks", [])),
                )
            event = HumanMediaEvent(
                session_id=session_id,
                camera=packet.source_id,
                track_id=track_id,
                name=name,
                captured_at_utc=packet.captured_at_utc,
                recognition_score=float(human.get("recognition_score", 0.0) or 0.0),
                ref_img_id=human.get("ref_img_id"),
                snapshot_frame=human_crop if better_snapshot else None,
                snapshot_quality=snapshot_quality,
                human_video_frame=(
                    self._letterbox(human_crop, self.HUMAN_VIDEO_SIZE)
                    if video_due
                    else None
                ),
                face_video_frame=face_frame,
                face_quality=face_quality,
                face_yaw=self._optional_float(quality_metrics.get("yaw")),
                face_pitch=self._optional_float(quality_metrics.get("pitch")),
                face_roll=self._optional_float(quality_metrics.get("roll")),
            )
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                with self._lock:
                    self._dropped_events += 1
                    if previous_name is None:
                        self._observed_names.pop(key, None)
                    else:
                        self._observed_names[key] = previous_name
                    if better_snapshot:
                        if previous_score < 0:
                            self._candidate_scores.pop(key, None)
                        else:
                            self._candidate_scores[key] = previous_score
                    if video_due:
                        if previous_video_at == float("-inf"):
                            self._last_video_at.pop(key, None)
                        else:
                            self._last_video_at[key] = previous_video_at
                LOGGER.warning("Human media queue is full; newest frame was dropped")

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        return None if value is None else float(value)

    @staticmethod
    def _safe_stem(camera: str, track_id: int) -> str:
        return f"{camera}_{track_id}_{uuid.uuid4().hex[:12]}".replace(
            "/", "_"
        ).replace("\\", "_")

    def _new_writer(
        self,
        directory: Path,
        url_prefix: str,
        stem: str,
        size: tuple[int, int],
    ) -> tuple[cv2.VideoWriter, str]:
        filename = f"{stem}.mp4"
        path = directory / filename
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.video_fps,
            size,
        )
        if not writer.isOpened():
            writer.release()
            path.unlink(missing_ok=True)
            raise RuntimeError(f"Could not create human video: {path}")
        return writer, f"/media/{url_prefix}/{filename}"

    def _media_state(self, event: HumanMediaEvent) -> TrackMediaState:
        key = (event.session_id, event.camera, event.track_id)
        state = self._media.get(key)
        if state is None:
            state = TrackMediaState()
            self._media[key] = state
        state.last_event_monotonic = time.monotonic()
        stem = self._safe_stem(event.camera, event.track_id)
        if event.human_video_frame is not None and state.human_writer is None:
            state.human_writer, state.video_url = self._new_writer(
                self.video_dir,
                "human_videos",
                stem,
                self.HUMAN_VIDEO_SIZE,
            )
            self._created_human_videos += 1
        if event.face_video_frame is not None and state.face_writer is None:
            state.face_writer, state.face_video_url = self._new_writer(
                self.face_video_dir,
                "human_face_videos",
                f"{stem}_faces",
                self.FACE_VIDEO_SIZE,
            )
            self._created_face_videos += 1
        return state

    def _save_snapshot(self, event: HumanMediaEvent) -> tuple[str, Path]:
        assert event.snapshot_frame is not None
        filename = f"{self._safe_stem(event.camera, event.track_id)}.jpg"
        path = self.snapshot_dir / filename
        if not cv2.imwrite(
            str(path),
            event.snapshot_frame,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        ):
            raise RuntimeError(f"Could not save human snapshot: {path}")
        return f"/media/human_snapshots/{filename}", path

    def _write(self, event: HumanMediaEvent) -> None:
        state = self._media_state(event)
        wrote_human = False
        wrote_face = False
        if event.human_video_frame is not None and state.human_writer is not None:
            state.human_writer.write(event.human_video_frame)
            wrote_human = True
        if event.face_video_frame is not None and state.face_writer is not None:
            state.face_writer.write(event.face_video_frame)
            wrote_face = True

        old_snapshot_url = ""
        new_snapshot_url = ""
        new_snapshot_path: Path | None = None
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT snapshot_url, snapshot_quality, best_face_quality, name,
                       recognition_score, ref_img_id
                FROM human_logs
                WHERE session_id = ? AND camera = ? AND track_id = ?
                """,
                (event.session_id, event.camera, event.track_id),
            ).fetchone()
            current_snapshot_quality = (
                float(existing["snapshot_quality"] or 0.0) if existing else -1.0
            )
            should_replace_snapshot = (
                event.snapshot_frame is not None
                and event.snapshot_quality
                >= current_snapshot_quality + self.snapshot_min_improvement
            )
            if existing is None and event.snapshot_frame is not None:
                should_replace_snapshot = True
            if should_replace_snapshot:
                old_snapshot_url = str(existing["snapshot_url"] or "") if existing else ""
                new_snapshot_url, new_snapshot_path = self._save_snapshot(event)
            snapshot_url = (
                new_snapshot_url
                if new_snapshot_url
                else str(existing["snapshot_url"] or "") if existing else ""
            )
            snapshot_quality = (
                event.snapshot_quality
                if should_replace_snapshot
                else max(current_snapshot_quality, 0.0)
            )
            current_face_quality = (
                float(existing["best_face_quality"] or 0.0) if existing else 0.0
            )
            better_face = event.face_quality > current_face_quality
            stored_name = event.name
            if existing and event.name == "Unknown" and existing["name"] != "Unknown":
                stored_name = str(existing["name"])
            stored_recognition_score = event.recognition_score
            stored_ref_img_id = event.ref_img_id
            if existing and float(existing["recognition_score"] or 0.0) > event.recognition_score:
                stored_recognition_score = float(existing["recognition_score"])
                stored_ref_img_id = existing["ref_img_id"]

            if existing is None:
                connection.execute(
                    """
                    INSERT INTO human_logs (
                        session_id, camera, track_id, name, first_seen, last_seen,
                        recognition_score, ref_img_id, snapshot_url, video_url,
                        face_video_url, snapshot_quality, best_face_quality,
                        best_face_yaw, best_face_pitch, best_face_roll,
                        human_video_frames, accepted_face_frames
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.session_id,
                        event.camera,
                        event.track_id,
                        stored_name,
                        event.captured_at_utc,
                        event.captured_at_utc,
                        stored_recognition_score,
                        None if stored_ref_img_id is None else str(stored_ref_img_id),
                        snapshot_url,
                        state.video_url,
                        state.face_video_url,
                        snapshot_quality,
                        event.face_quality,
                        event.face_yaw,
                        event.face_pitch,
                        event.face_roll,
                        int(wrote_human),
                        int(wrote_face),
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE human_logs
                    SET name = ?, last_seen = ?, recognition_score = ?, ref_img_id = ?,
                        snapshot_url = ?, video_url = ?, face_video_url = ?,
                        snapshot_quality = ?,
                        best_face_quality = CASE WHEN ? THEN ? ELSE best_face_quality END,
                        best_face_yaw = CASE WHEN ? THEN ? ELSE best_face_yaw END,
                        best_face_pitch = CASE WHEN ? THEN ? ELSE best_face_pitch END,
                        best_face_roll = CASE WHEN ? THEN ? ELSE best_face_roll END,
                        human_video_frames = human_video_frames + ?,
                        accepted_face_frames = accepted_face_frames + ?
                    WHERE session_id = ? AND camera = ? AND track_id = ?
                    """,
                    (
                        stored_name,
                        event.captured_at_utc,
                        max(stored_recognition_score, 0.0),
                        None if stored_ref_img_id is None else str(stored_ref_img_id),
                        snapshot_url,
                        state.video_url,
                        state.face_video_url,
                        snapshot_quality,
                        int(better_face),
                        event.face_quality,
                        int(better_face),
                        event.face_yaw,
                        int(better_face),
                        event.face_pitch,
                        int(better_face),
                        event.face_roll,
                        int(wrote_human),
                        int(wrote_face),
                        event.session_id,
                        event.camera,
                        event.track_id,
                    ),
                )
        if old_snapshot_url and old_snapshot_url != new_snapshot_url:
            (self.snapshot_dir / Path(old_snapshot_url).name).unlink(missing_ok=True)
        with self._lock:
            self._saved_snapshots += int(bool(new_snapshot_path))
            self._human_video_frames += int(wrote_human)
            self._accepted_face_video_frames += int(wrote_face)
            self._last_error = None

    @staticmethod
    def _release_state(state: TrackMediaState) -> None:
        if state.human_writer is not None:
            state.human_writer.release()
        if state.face_writer is not None:
            state.face_writer.release()

    def _close_idle_media(self) -> None:
        now = time.monotonic()
        for key, state in list(self._media.items()):
            if now - state.last_event_monotonic < self.video_idle_seconds:
                continue
            self._release_state(state)
            self._media.pop(key, None)

    def _close_all_media(self) -> None:
        for state in self._media.values():
            self._release_state(state)
        self._media.clear()

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                self._close_idle_media()
                continue
            try:
                if item is _STOP:
                    self._close_all_media()
                    return
                assert isinstance(item, HumanMediaEvent)
                self._write(item)
                self._close_idle_media()
            except Exception as exc:
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                LOGGER.exception("Human media write failed")
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
                       last_seen, recognition_score, ref_img_id, snapshot_url,
                       video_url, face_video_url, snapshot_quality,
                       best_face_quality, best_face_yaw, best_face_pitch,
                       best_face_roll, human_video_frames, accepted_face_frames
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
                "human_video_frames": self._human_video_frames,
                "accepted_face_video_frames": self._accepted_face_video_frames,
                "created_human_videos": self._created_human_videos,
                "created_face_videos": self._created_face_videos,
                "open_track_recorders": len(self._media),
                "video_fps": self.video_fps,
                "snapshot_directory": str(self.snapshot_dir),
                "video_directory": str(self.video_dir),
                "face_video_directory": str(self.face_video_dir),
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
