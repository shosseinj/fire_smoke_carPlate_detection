from __future__ import annotations

from app.database import Connection, Database, IntegrityError, OperationalError, Row, ensure_database

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.types import FramePacket, TaskResult
from app.core.detection_log_store import DetectionLogStore
from app.core.personnel_store import normalize_national_code


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
    personnel_id: int | None
    snapshot_frame: np.ndarray | None
    face_image_frame: np.ndarray | None
    snapshot_quality: float
    full_frame_video_frame: np.ndarray | None
    face_video_frame: np.ndarray | None
    face_quality: float
    finalize_detection_log: bool = False
    persist_human_log: bool = True


@dataclass(slots=True)
class TrackMediaState:
    full_frame_writer: cv2.VideoWriter | None = None
    face_writer: cv2.VideoWriter | None = None
    full_frame_size: tuple[int, int] | None = None
    face_size: tuple[int, int] | None = None
    video_url: str = ""
    face_video_url: str = ""
    last_event_monotonic: float = 0.0


class HumanLogStore:
    """Non-blocking per-ByteTrack best snapshot and independent video recorder."""

    def __init__(
        self,
        database: Database | str,
        saved_media_path: Path,
        *,
        queue_size: int = 128,
        video_fps: float = 10.0,
        video_idle_seconds: float = 5.0,
        snapshot_min_improvement: float = 0.01,
        detection_log_store: DetectionLogStore | None = None,
    ) -> None:
        self.database = ensure_database(database)
        media_root = saved_media_path.resolve()
        self.snapshot_dir = media_root / "human_snapshots"
        self.detected_face_dir = media_root / "detected_faces"
        self.video_dir = media_root / "human_videos"
        self.face_video_dir = media_root / "human_face_videos"
        for directory in (
            self.snapshot_dir,
            self.detected_face_dir,
            self.video_dir,
            self.face_video_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.video_fps = max(1.0, float(video_fps))
        self.video_idle_seconds = max(1.0, float(video_idle_seconds))
        self.snapshot_min_improvement = max(0.0, float(snapshot_min_improvement))
        self.detection_log_store = detection_log_store
        self._personnel_identity_cache: dict[str, tuple[str, int] | None] = {}
        self._reference_image_cache: dict[str, np.ndarray | None] = {}
        self._queue: queue.Queue[HumanMediaEvent | object] = queue.Queue(
            maxsize=max(8, int(queue_size))
        )
        self._observed_names: dict[tuple[str, str, int], str] = {}
        self._candidate_scores: dict[tuple[str, str, int], float] = {}
        self._last_full_frame_at: dict[tuple[str, str, int], float] = {}
        self._last_face_video_at: dict[tuple[str, str, int], float] = {}
        self._best_face_scores: dict[tuple[str, str, int], float] = {}
        self._media: dict[tuple[str, str, int], TrackMediaState] = {}
        self._last_face_images: dict[tuple[str, str, int], np.ndarray] = {}
        self._lock = threading.RLock()
        self._dropped_events = 0
        self._saved_snapshots = 0
        self._full_frame_video_frames = 0
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

    def _connect(self) -> Connection:
        return self.database.connection()

    def _create_schema(self) -> None:
        # Alembic owns the PostgreSQL schema; runtime startup validates it.
        return None

    def _resolve_personnel_identity(
        self,
        person: str,
        ref_img_id: str | int | None,
    ) -> tuple[str, int | None]:
        """Resolve Qdrant's national-code identity to the stored full name."""
        candidates: list[tuple[str, str]] = []
        person_code = normalize_national_code(person)
        if len(person_code) == 10:
            candidates.append(("national_code", person_code))
        ref_text = str(ref_img_id).strip() if ref_img_id is not None else ""
        if ref_text.startswith("personnel_"):
            candidates.append(("id", ref_text[len("personnel_") :]))
        else:
            ref_code = normalize_national_code(ref_text)
            if len(ref_code) == 10 and ref_code != person_code:
                candidates.append(("national_code", ref_code))

        for lookup_type, lookup_value in candidates:
            cache_key = f"{lookup_type}:{lookup_value}"
            if cache_key in self._personnel_identity_cache:
                cached = self._personnel_identity_cache[cache_key]
                if cached is not None:
                    return cached
                continue
            if lookup_type == "id":
                query = "SELECT id, fname, lname FROM personnel WHERE id = ?"
            else:
                query = "SELECT id, fname, lname FROM personnel WHERE national_code = ?"
            with self._connect() as connection:
                row = connection.execute(query, (lookup_value,)).fetchone()
            if row is not None:
                identity = (
                    f"{str(row['fname']).strip()} {str(row['lname']).strip()}".strip(),
                    int(row["id"]),
                )
                self._personnel_identity_cache[cache_key] = identity
                return identity
            self._personnel_identity_cache[cache_key] = None
        return person, None

    def _reference_image(self, ref_img_id: str | int | None) -> np.ndarray | None:
        if ref_img_id is None:
            return None
        ref_text = str(ref_img_id).strip()
        if not ref_text:
            return None
        cached = self._reference_image_cache.get(ref_text)
        if cached is not None:
            return cached.copy()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT storage_key FROM personnel_images WHERE id = ?",
                (int(ref_text),),
            ).fetchone() if ref_text.isdigit() else None
            if row is None and ref_text.isdigit():
                row = connection.execute(
                    "SELECT storage_key FROM personnel_images "
                    "WHERE personnel_id = ? ORDER BY is_primary DESC, id DESC LIMIT 1",
                    (int(ref_text),),
                ).fetchone()
            if row is None and ref_text.startswith("personnel_"):
                row = connection.execute(
                    "SELECT storage_key FROM personnel_images "
                    "WHERE personnel_id = ? ORDER BY is_primary DESC, id DESC LIMIT 1",
                    (int(ref_text[len("personnel_"):]),),
                ).fetchone()
            if row is None:
                return None
        storage_key = Path(str(row["storage_key"])).as_posix()
        path = (self.snapshot_dir.parent / storage_key).resolve()
        media_root = self.snapshot_dir.parent.resolve()
        if media_root not in path.parents or not path.is_file():
            return None
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return None
        self._reference_image_cache[ref_text] = image.copy()
        return image

    @staticmethod
    def _concat_reference_and_face(
        reference_image: np.ndarray,
        face_image: np.ndarray,
    ) -> np.ndarray | None:
        if reference_image.size == 0 or face_image.size == 0:
            return None
        reference = reference_image
        face = face_image
        if reference.ndim == 2:
            reference = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
        if face.ndim == 2:
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
        target_height = min(max(reference.shape[0], face.shape[0]), 512)
        parts: list[np.ndarray] = []
        for image in (reference, face):
            width = max(1, int(round(image.shape[1] * target_height / image.shape[0])))
            parts.append(cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA))
        return cv2.hconcat(parts)

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
        x1, y1, x2, y2 = (float(value) for value in bbox[:4])
        side = max(112, int(round(max(x2 - x1, y2 - y1))))
        if side % 2:
            side += 1
        reference = (side / 112.0) * np.asarray(
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
        return cv2.warpAffine(frame, transform, (side, side))

    def observe_result(
        self,
        packet: FramePacket,
        result: TaskResult,
        *,
        persist_human_log: bool = True,
    ) -> None:
        if result.error:
            return
        source_frame = packet.source_frame
        session_id = str(result.data.get("tracking_session_id") or "unknown-session")
        faces_by_track: dict[int, dict[str, Any]] = {}
        for face in result.data.get("faces", []):
            track_id = face.get("track_id")
            if track_id is None:
                continue
            existing = faces_by_track.get(int(track_id))
            if existing is None or float(face.get("quality_score", 0.0)) > float(
                existing.get("quality_score", 0.0)
            ):
                faces_by_track[int(track_id)] = face

        human_entries = [
            (human, False) for human in result.data.get("humans", [])
        ] + [
            (human, True) for human in result.data.get("disappeared_humans", [])
        ]
        for human, disappeared in human_entries:
            track_id = human.get("track_id")
            if track_id is None:
                continue
            track_id = int(track_id)
            key = (session_id, packet.source_id, track_id)
            raw_name = str(human.get("person") or "Unknown").strip() or "Unknown"
            raw_ref = human.get("ref_img_id")
            name, resolved_personnel_id = self._resolve_personnel_identity(
                raw_name, raw_ref
            )
            bbox = [
                float(value)
                for value in human.get(
                    "source_bbox", human.get("bbox", [0, 0, 0, 0])
                )[:4]
            ]
            human_crop = self._human_crop(source_frame, bbox)
            face = faces_by_track.get(track_id)
            face_quality = float(face.get("quality_score", 0.0)) if face else 0.0
            frame_area = max(1.0, float(source_frame.shape[0] * source_frame.shape[1]))
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
            face_frame = None
            if face is not None:
                face_frame = self._aligned_face(
                    source_frame,
                    [
                        float(value)
                        for value in face.get(
                            "source_bbox", face.get("bbox", [0, 0, 0, 0])
                        )[:4]
                    ],
                    list(face.get("source_landmarks", face.get("landmarks", []))),
                )
                if face_frame is None:
                    face_frame = self._human_crop(
                        source_frame,
                        [
                            float(value)
                            for value in face.get(
                                "source_bbox", face.get("bbox", [0, 0, 0, 0])
                            )[:4]
                        ],
                    )
            with self._lock:
                remembered_face = self._last_face_images.get(key)
            face_image_frame = (
                face_frame
                if face_frame is not None
                else remembered_face
                if remembered_face is not None
                else None
            )
            if human_crop is None and disappeared:
                human_crop = source_frame.copy()
            if human_crop is None:
                continue
            if not persist_human_log and (disappeared or face_frame is None):
                continue
            current_face_or_human = face_image_frame if face_image_frame is not None else human_crop
            reference_frame = (
                self._reference_image(raw_ref) if name != "Unknown" else None
            )
            snapshot_image = current_face_or_human
            if reference_frame is not None and face_image_frame is not None:
                snapshot_image = self._concat_reference_and_face(
                    reference_frame,
                    face_image_frame,
                )
            with self._lock:
                if face_frame is not None:
                    self._last_face_images[key] = face_frame.copy()
                previous_name = self._observed_names.get(key)
                previous_score = self._candidate_scores.get(key, -1.0)
                previous_full_frame_at = self._last_full_frame_at.get(
                    key, float("-inf")
                )
                previous_face_video_at = self._last_face_video_at.get(
                    key, float("-inf")
                )
                previous_best_face = self._best_face_scores.get(key, -1.0)
                identity_changed = previous_name is None or (
                    previous_name == "Unknown" and name != "Unknown"
                )
                better_snapshot = (
                    snapshot_quality
                    >= previous_score + self.snapshot_min_improvement
                )
                full_frame_due = not disappeared and (
                    now - previous_full_frame_at >= 1.0 / self.video_fps
                )
                better_face = (
                    face_frame is not None
                    and face_quality
                    >= previous_best_face + self.snapshot_min_improvement
                )
                face_due = face_frame is not None and (
                    better_face
                    or now - previous_face_video_at >= 1.0 / self.video_fps
                )
                if persist_human_log is False and not disappeared and not (
                    identity_changed or better_snapshot or full_frame_due or face_due
                ):
                    continue
                self._observed_names[key] = name
                if better_snapshot:
                    self._candidate_scores[key] = snapshot_quality
                if full_frame_due:
                    self._last_full_frame_at[key] = now
                if face_due:
                    self._last_face_video_at[key] = now
                if better_face:
                    self._best_face_scores[key] = face_quality

            raw_personnel_id = resolved_personnel_id
            if raw_personnel_id is None and raw_ref is not None:
                ref_str = str(raw_ref)
                if ref_str.startswith("personnel_"):
                    try:
                        raw_personnel_id = int(ref_str[len("personnel_"):])
                    except (ValueError, IndexError):
                        pass
            event = HumanMediaEvent(
                session_id=session_id,
                camera=packet.source_id,
                track_id=track_id,
                name=name,
                captured_at_utc=packet.captured_at_utc,
                recognition_score=float(human.get("recognition_score", 0.0) or 0.0),
                ref_img_id=raw_ref,
                personnel_id=raw_personnel_id,
                snapshot_frame=(
                    snapshot_image
                    if persist_human_log and (better_snapshot or disappeared)
                    else None
                ),
                face_image_frame=face_image_frame,
                snapshot_quality=snapshot_quality,
                full_frame_video_frame=(
                    source_frame.copy() if persist_human_log and full_frame_due else None
                ),
                face_video_frame=face_frame if persist_human_log and face_due else None,
                face_quality=face_quality,
                finalize_detection_log=persist_human_log and disappeared,
                persist_human_log=persist_human_log,
            )
            try:
                self._queue.put_nowait(event)
                if disappeared:
                    with self._lock:
                        self._last_face_images.pop(key, None)
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
                    if full_frame_due:
                        if previous_full_frame_at == float("-inf"):
                            self._last_full_frame_at.pop(key, None)
                        else:
                            self._last_full_frame_at[key] = previous_full_frame_at
                    if face_due:
                        if previous_face_video_at == float("-inf"):
                            self._last_face_video_at.pop(key, None)
                        else:
                            self._last_face_video_at[key] = previous_face_video_at
                    if better_face:
                        if previous_best_face < 0:
                            self._best_face_scores.pop(key, None)
                        else:
                            self._best_face_scores[key] = previous_best_face
                LOGGER.warning("Human media queue is full; newest frame was dropped")
            if disappeared and face_image_frame is None:
                LOGGER.warning(
                    "HUMAN_FACE_IMAGE_UNAVAILABLE camera=%s track_id=%s",
                    packet.source_id,
                    track_id,
                )

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
        if (
            event.full_frame_video_frame is not None
            and state.full_frame_writer is None
        ):
            height, width = event.full_frame_video_frame.shape[:2]
            state.full_frame_size = (width, height)
            state.full_frame_writer, state.video_url = self._new_writer(
                self.video_dir,
                "human_videos",
                stem,
                state.full_frame_size,
            )
            self._created_human_videos += 1
        if event.face_video_frame is not None and state.face_writer is None:
            face_height, face_width = event.face_video_frame.shape[:2]
            state.face_size = (face_width, face_height)
            state.face_writer, state.face_video_url = self._new_writer(
                self.face_video_dir,
                "human_face_videos",
                f"{stem}_faces",
                state.face_size,
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

    def _save_face_image(self, event: HumanMediaEvent) -> tuple[str, Path]:
        assert event.face_image_frame is not None
        filename = f"{self._safe_stem(event.camera, event.track_id)}_face.jpg"
        path = self.detected_face_dir / filename
        if not cv2.imwrite(
            str(path),
            event.face_image_frame,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        ):
            raise RuntimeError(f"Could not save human face image: {path}")
        LOGGER.info(
            "HUMAN_FACE_IMAGE_SAVED camera=%s track_id=%s path=%s",
            event.camera,
            event.track_id,
            path.relative_to(self.snapshot_dir.parent),
        )
        return f"/media/detected_faces/{filename}", path

    def _write(self, event: HumanMediaEvent) -> None:
        state = self._media_state(event)
        wrote_full_frame = False
        wrote_face = False
        face_image_url = ""
        if event.face_image_frame is not None:
            face_image_url, _ = self._save_face_image(event)
        if not event.persist_human_log:
            with self._lock:
                self._last_error = None
            return
        if (
            event.full_frame_video_frame is not None
            and state.full_frame_writer is not None
            and state.full_frame_size is not None
        ):
            frame = event.full_frame_video_frame
            if (frame.shape[1], frame.shape[0]) != state.full_frame_size:
                frame = cv2.resize(frame, state.full_frame_size)
            state.full_frame_writer.write(frame)
            wrote_full_frame = True
        if event.face_video_frame is not None and state.face_writer is not None:
            face_frame = event.face_video_frame
            if (
                state.face_size is not None
                and (face_frame.shape[1], face_frame.shape[0]) != state.face_size
            ):
                face_frame = cv2.resize(face_frame, state.face_size)
            state.face_writer.write(face_frame)
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
                        full_frame_video_frames, accepted_face_frames,
                        personnel_id, counts_for_attendance
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        int(wrote_full_frame),
                        int(wrote_face),
                        event.personnel_id,
                        1,
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
                        full_frame_video_frames = full_frame_video_frames + ?,
                        accepted_face_frames = accepted_face_frames + ?,
                        personnel_id = CASE WHEN ? THEN ? ELSE personnel_id END
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
                        bool(better_face),
                        event.face_quality,
                        int(wrote_full_frame),
                        int(wrote_face),
                        bool(event.personnel_id is not None),
                        event.personnel_id,
                        event.session_id,
                        event.camera,
                        event.track_id,
                    ),
                )
            human_row = connection.execute(
                "SELECT id FROM human_logs WHERE session_id = ? AND camera = ? AND track_id = ?",
                (event.session_id, event.camera, event.track_id),
            ).fetchone()
        if event.finalize_detection_log and self.detection_log_store is not None:
            source_event_key = (
                f"human-track:{event.session_id}:{event.camera}:{event.track_id}"
            )
            existing_detection = self.detection_log_store.get_by_source_event_key(
                source_event_key
            )
            if existing_detection is None:
                self.detection_log_store.create(
                    source_system="face_recognition",
                    source_event_key=source_event_key,
                    source_human_log_id=(int(human_row["id"]) if human_row else None),
                    personnel_id=event.personnel_id,
                    person=event.name,
                    confidence=event.recognition_score,
                    detection_time=event.captured_at_utc,
                    ref_img_id=(
                        None if event.ref_img_id is None else str(event.ref_img_id)
                    ),
                    camera_id=event.camera,
                    access_granted=event.personnel_id is not None,
                    counts_for_attendance=True,
                    log_type="camera_rtsp",
                    face_image=face_image_url,
                    snapshot_image=snapshot_url,
                    video=state.video_url,
                    face_video_or_unknown_faces=state.face_video_url,
                )
            elif not existing_detection.face_image and face_image_url:
                self.detection_log_store.update(
                    existing_detection.id,
                    face_image=face_image_url,
                    snapshot_image=snapshot_url or existing_detection.snapshot_image,
                    video=state.video_url or existing_detection.video,
                    face_video_or_unknown_faces=(
                        state.face_video_url
                        or existing_detection.face_video_or_unknown_faces
                    ),
                )
        if old_snapshot_url and old_snapshot_url != new_snapshot_url:
            (self.snapshot_dir / Path(old_snapshot_url).name).unlink(missing_ok=True)
        with self._lock:
            self._saved_snapshots += int(bool(new_snapshot_path))
            self._full_frame_video_frames += int(wrote_full_frame)
            self._accepted_face_video_frames += int(wrote_face)
            self._last_error = None

    @staticmethod
    def _release_state(state: TrackMediaState) -> None:
        if state.full_frame_writer is not None:
            state.full_frame_writer.release()
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
                       best_face_quality, full_frame_video_frames,
                       accepted_face_frames, personnel_id, counts_for_attendance
                FROM human_logs
                """
                + where
                + " ORDER BY id DESC LIMIT ?",
                values,
            ).fetchall()
        items = [dict(row) for row in rows]
        for item in items:
            full_name = str(item.get("name") or "Unknown").strip() or "Unknown"
            if full_name == "Unknown":
                first_name, last_name = "Unknown", ""
            else:
                parts = full_name.split(maxsplit=1)
                first_name = parts[0]
                last_name = parts[1] if len(parts) > 1 else ""
            item["first_name"] = first_name
            item["last_name"] = last_name
            item["image_url"] = item.get("snapshot_url") or ""
        return items

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
                "full_frame_video_frames": self._full_frame_video_frames,
                "accepted_face_video_frames": self._accepted_face_video_frames,
                "created_human_videos": self._created_human_videos,
                "created_face_videos": self._created_face_videos,
                "open_track_recorders": len(self._media),
                "video_fps": self.video_fps,
                "snapshot_directory": str(self.snapshot_dir),
                "detected_face_directory": str(self.detected_face_dir),
                "video_directory": str(self.video_dir),
                "face_video_directory": str(self.face_video_dir),
                "last_error": self._last_error,
            }

    def get_logs_for_personnel(
        self, personnel_id: int, utc_start: str, utc_end: str,
    ) -> list[dict[str, Any]]:
        """Return human_log rows for a personnel within a UTC range."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, session_id, camera, track_id, name, first_seen,
                       last_seen, recognition_score, ref_img_id, snapshot_url,
                       video_url, face_video_url, snapshot_quality,
                       best_face_quality, full_frame_video_frames,
                       accepted_face_frames, personnel_id, counts_for_attendance
                FROM human_logs
                WHERE personnel_id = ?
                  AND last_seen >= ?
                  AND last_seen <= ?
                  AND counts_for_attendance = 1
                ORDER BY last_seen ASC
                """,
                (personnel_id, utc_start, utc_end),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_logs_in_range(
        self, utc_start: str, utc_end: str,
        personnel_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return human_log rows in a UTC range, optionally filtered by personnel."""
        where = "last_seen >= ? AND last_seen <= ? AND counts_for_attendance = 1"
        params: list[Any] = [utc_start, utc_end]
        if personnel_id is not None:
            where += " AND personnel_id = ?"
            params.append(personnel_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, session_id, camera, track_id, name, first_seen,
                       last_seen, recognition_score, ref_img_id, snapshot_url,
                       video_url, face_video_url, snapshot_quality,
                       best_face_quality, full_frame_video_frames,
                       accepted_face_frames, personnel_id, counts_for_attendance
                FROM human_logs
                WHERE """ + where + """
                ORDER BY last_seen ASC
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def set_counts_for_attendance(self, log_id: int, counts: bool) -> dict[str, Any] | None:
        """Toggle counts_for_attendance on a human_log row."""
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id FROM human_logs WHERE id = ?", (log_id,)
            ).fetchone()
            if existing is None:
                return None
            connection.execute(
                "UPDATE human_logs SET counts_for_attendance = ? WHERE id = ?",
                (1 if counts else 0, log_id),
            )
            row = connection.execute(
                "SELECT * FROM human_logs WHERE id = ?", (log_id,)
            ).fetchone()
            return dict(row) if row else None

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
