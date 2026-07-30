from __future__ import annotations

from app.database import Connection, Database, IntegrityError, OperationalError, Row, ensure_database

import logging
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.types import FramePacket, TaskResult
from app.core.detection_log_store import DetectionLogStore
from app.core.detection_media import DetectionMediaStorage, MEDIA_STATUS_READY
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
    evidence_captured_at_utc: str
    recognition_score: float
    ref_img_id: str | int | None
    personnel_id: int | None
    room_id: int | None
    snapshot_frame: np.ndarray | None
    whole_snapshot_frame: np.ndarray | None
    face_image_frame: np.ndarray | None
    snapshot_quality: float
    # A sampled full camera frame associated with this person's track.
    full_frame_video_frame: np.ndarray | None
    face_video_frame: np.ndarray | None
    face_quality: float
    room_ids: tuple[int, ...] = ()
    pre_roll_frames: tuple[np.ndarray, ...] = ()
    evidence_frame_index: int | None = None
    ranked_face_candidates: tuple[RankedFaceCandidate, ...] = ()
    finalize_detection_log: bool = False
    persist_human_log: bool = True
    counts_for_attendance: bool = True


@dataclass(slots=True)
class TrackMediaState:
    full_frame_writer: cv2.VideoWriter | None = None
    face_writer: cv2.VideoWriter | None = None
    full_frame_size: tuple[int, int] | None = None
    face_size: tuple[int, int] | None = None
    video_key: str = ""
    face_video_key: str = ""
    last_event_monotonic: float = 0.0
    full_frame_frames: int = 0
    face_frames: int = 0


@dataclass(slots=True)
class PendingFinalization:
    event: HumanMediaEvent
    remaining_frames: int
    last_frame_index: int | None = None


@dataclass(slots=True)
class StillEvidenceBundle:
    """One frame's atomic still-image evidence for a tracked human."""

    captured_at_utc: str
    frame_index: int
    body_frame: np.ndarray
    full_frame: np.ndarray
    face_frame: np.ndarray | None
    quality: float
    face_quality: float


@dataclass(slots=True)
class RankedFaceCandidate:
    """A bounded face crop ranked by the existing Face Quality score."""

    captured_at_utc: str
    frame_index: int
    face_frame: np.ndarray
    face_quality: float


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
        video_pre_roll_frames: int = 0,
        video_post_roll_frames: int = 0,
        video_pre_roll_max_bytes: int = 128 * 1024 * 1024,
        snapshot_min_improvement: float = 0.01,
        face_candidate_limit: int = 5,
        detection_log_store: DetectionLogStore | None = None,
    ) -> None:
        self.database = ensure_database(database)
        media_root = saved_media_path.resolve()
        self.media_storage = DetectionMediaStorage(media_root)
        human_media_root = media_root / "human"
        self.snapshot_dir = human_media_root / "body_images"
        self.whole_snapshot_dir = human_media_root / "full_frame_images"
        self.detected_face_dir = human_media_root / "detected_faces"
        self.face_thumbnail_dir = human_media_root / "face_thumbnails"
        self.video_dir = human_media_root / "videos"
        self.face_video_dir = human_media_root / "face_videos"
        for directory in (
            self.snapshot_dir,
            self.whole_snapshot_dir,
            self.detected_face_dir,
            self.face_thumbnail_dir,
            self.video_dir,
            self.face_video_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.video_fps = max(1.0, float(video_fps))
        self.video_idle_seconds = max(1.0, float(video_idle_seconds))
        self.video_pre_roll_frames = max(0, min(int(video_pre_roll_frames), 30))
        self.video_post_roll_frames = max(0, min(int(video_post_roll_frames), 30))
        self.video_pre_roll_max_bytes = max(0, int(video_pre_roll_max_bytes))
        self.snapshot_min_improvement = max(0.0, float(snapshot_min_improvement))
        self.face_candidate_limit = max(1, int(face_candidate_limit))
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
        self._last_pre_roll_frame_index: dict[tuple[str, str], int] = {}
        self._pre_roll_frames: dict[
            tuple[str, str], deque[np.ndarray]
        ] = {}
        self._pre_roll_buffered_bytes = 0
        self._pending_finalizations: dict[
            tuple[str, str, int], PendingFinalization
        ] = {}
        self._best_face_scores: dict[tuple[str, str, int], float] = {}
        # Video recording is admitted by the first valid face for a track, but
        # remains independent from polygon-gated human-log persistence.
        self._video_enabled_tracks: set[tuple[str, str, int]] = set()
        self._track_room_ids: dict[tuple[str, str, int], int] = {}
        self._track_visited_room_ids: dict[
            tuple[str, str, int], set[int]
        ] = {}
        self._media: dict[tuple[str, str, int], TrackMediaState] = {}
        self._still_evidence: dict[
            tuple[str, str, int], StillEvidenceBundle
        ] = {}
        self._face_candidates: dict[
            tuple[str, str, int], list[RankedFaceCandidate]
        ] = {}
        self._lock = threading.RLock()
        self._dropped_events = 0
        self._saved_snapshots = 0
        self._full_frame_video_frames = 0
        self._accepted_face_video_frames = 0
        self._created_human_videos = 0
        self._created_face_videos = 0
        self._saved_ranked_face_candidates = 0
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
        path = self.media_storage.resolve(storage_key, require_file=True)
        if path is None:
            return None
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return None
        self._reference_image_cache[ref_text] = image.copy()
        return image

    def _detection_person(self, personnel_id: int | None) -> str:
        """Return the canonical identity persisted in detection_logs.person."""
        if personnel_id is None:
            return "Unknown"
        with self._connect() as connection:
            row = connection.execute(
                "SELECT national_code FROM personnel WHERE id = ?",
                (int(personnel_id),),
            ).fetchone()
        if row is None:
            return "Unknown"
        national_code = normalize_national_code(str(row["national_code"] or ""))
        return national_code or "Unknown"

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

    def _buffer_pre_roll_frame(
        self,
        session_id: str,
        source_id: str,
        frame_index: int,
        source_frame: np.ndarray,
    ) -> None:
        if self.video_pre_roll_frames <= 0 or self.video_pre_roll_max_bytes <= 0:
            return
        buffer_key = (session_id, source_id)
        with self._lock:
            if self._last_pre_roll_frame_index.get(buffer_key) == frame_index:
                return
            for stale_key in [
                key
                for key in self._pre_roll_frames
                if key[1] == source_id and key != buffer_key
            ]:
                stale = self._pre_roll_frames.pop(stale_key)
                self._pre_roll_buffered_bytes -= sum(frame.nbytes for frame in stale)
                self._last_pre_roll_frame_index.pop(stale_key, None)
            frames = self._pre_roll_frames.setdefault(
                buffer_key, deque(maxlen=self.video_pre_roll_frames + 1)
            )
            if frames.maxlen is not None and len(frames) >= frames.maxlen:
                self._pre_roll_buffered_bytes -= frames[0].nbytes
            buffered = source_frame.copy()
            frames.append(buffered)
            self._pre_roll_buffered_bytes += buffered.nbytes
            self._last_pre_roll_frame_index[buffer_key] = frame_index
            while self._pre_roll_buffered_bytes > self.video_pre_roll_max_bytes:
                evicted = False
                for candidate_key, candidate_frames in list(
                    self._pre_roll_frames.items()
                ):
                    if not candidate_frames:
                        continue
                    oldest = candidate_frames.popleft()
                    self._pre_roll_buffered_bytes -= oldest.nbytes
                    evicted = True
                    if not candidate_frames:
                        self._pre_roll_frames.pop(candidate_key, None)
                        self._last_pre_roll_frame_index.pop(candidate_key, None)
                    break
                if not evicted:
                    break

    def _advance_pending_finalizations(
        self, packet: FramePacket, session_id: str
    ) -> None:
        pending_events: list[tuple[tuple[str, str, int], PendingFinalization]] = []
        with self._lock:
            for key, pending in self._pending_finalizations.items():
                if key[0] != session_id or key[1] != packet.source_id:
                    continue
                if pending.last_frame_index == packet.frame_index:
                    continue
                pending_events.append((key, pending))

        for key, pending in pending_events:
            is_final = pending.remaining_frames <= 1
            event = replace(
                pending.event,
                full_frame_video_frame=packet.source_frame.copy(),
                face_video_frame=None,
                pre_roll_frames=(),
                snapshot_frame=(
                    pending.event.snapshot_frame if is_final else None
                ),
                whole_snapshot_frame=(
                    pending.event.whole_snapshot_frame if is_final else None
                ),
                face_image_frame=(
                    pending.event.face_image_frame if is_final else None
                ),
                ranked_face_candidates=(
                    pending.event.ranked_face_candidates if is_final else ()
                ),
                finalize_detection_log=is_final,
                persist_human_log=is_final,
            )
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                with self._lock:
                    self._dropped_events += 1
                LOGGER.warning(
                    "Human media queue is full; post-roll frame was dropped"
                )
                continue
            with self._lock:
                current = self._pending_finalizations.get(key)
                if current is not pending:
                    continue
                if is_final:
                    self._pending_finalizations.pop(key, None)
                else:
                    pending.remaining_frames -= 1
                    pending.last_frame_index = packet.frame_index

    def observe_result(
        self,
        packet: FramePacket,
        result: TaskResult,
        *,
        persist_human_log: bool = True,
        room_ids_by_track: dict[int, int] | None = None,
        observed_room_ids_by_track: dict[int, set[int]] | None = None,
        exited_track_ids: set[int] | None = None,
        counts_for_attendance: bool = True,
    ) -> None:
        if result.error:
            return
        source_frame = packet.source_frame
        session_id = str(result.data.get("tracking_session_id") or "unknown-session")
        self._buffer_pre_roll_frame(
            session_id,
            packet.source_id,
            packet.frame_index,
            source_frame,
        )
        self._advance_pending_finalizations(packet, session_id)
        faces_by_track: dict[int, list[dict[str, Any]]] = {}
        for face in result.data.get("faces", []):
            # Invalid-quality faces remain excluded from recognition evidence.
            # Every valid face candidate is ranked below, including multiple
            # candidates associated with the same track in one result.
            if face.get("quality_valid") is not True:
                continue
            track_id = face.get("track_id")
            if track_id is None:
                continue
            faces_by_track.setdefault(int(track_id), []).append(face)
        for track_faces in faces_by_track.values():
            track_faces.sort(
                key=lambda item: float(item.get("quality_score", 0.0)),
                reverse=True,
            )

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
            require_valid_room = room_ids_by_track is not None
            admitted_room_id = (
                room_ids_by_track.get(track_id)
                if room_ids_by_track is not None
                else None
            )
            with self._lock:
                observed_room_ids = (
                    observed_room_ids_by_track.get(track_id, set())
                    if observed_room_ids_by_track is not None
                    else set()
                )
                if observed_room_ids:
                    self._track_visited_room_ids.setdefault(key, set()).update(
                        int(room_id) for room_id in observed_room_ids
                    )
                if admitted_room_id is not None:
                    self._track_room_ids[key] = int(admitted_room_id)
                    self._track_visited_room_ids.setdefault(key, set()).add(
                        int(admitted_room_id)
                    )
                stored_room_id = self._track_room_ids.get(key)
                visited_room_ids = tuple(
                    sorted(self._track_visited_room_ids.get(key, ()))
                )
            if persist_human_log and require_valid_room and stored_room_id is None:
                continue
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
            track_faces = faces_by_track.get(track_id, [])
            valid_room_frame = (
                room_ids_by_track is None or admitted_room_id is not None
            )
            if not valid_room_frame and not disappeared:
                # Do not let an outside-polygon face become the immutable still
                # evidence or activate video recording for a persisted log.
                track_faces = []
            face = track_faces[0] if track_faces else None
            face_quality = float(face.get("quality_score", 0.0)) if face else 0.0
            frame_area = max(1.0, float(source_frame.shape[0] * source_frame.shape[1]))
            human_area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            snapshot_quality = min(
                0.60,
                0.35 * float(human.get("confidence", 0.0) or 0.0)
                + 0.25 * min(1.0, human_area / frame_area * 5.0),
            )
            now = float(packet.captured_monotonic)
            face_frame = None
            frame_face_candidates: list[RankedFaceCandidate] = []
            if face is not None:
                for observed_face in track_faces:
                    observed_face_frame = self._aligned_face(
                        source_frame,
                        [
                            float(value)
                            for value in observed_face.get(
                                "source_bbox",
                                observed_face.get("bbox", [0, 0, 0, 0]),
                            )[:4]
                        ],
                        list(
                            observed_face.get(
                                "source_landmarks",
                                observed_face.get("landmarks", []),
                            )
                        ),
                    )
                    if observed_face_frame is None:
                        observed_face_frame = self._human_crop(
                            source_frame,
                            [
                                float(value)
                                for value in observed_face.get(
                                    "source_bbox",
                                    observed_face.get("bbox", [0, 0, 0, 0]),
                                )[:4]
                            ],
                        )
                    if observed_face_frame is None:
                        continue
                    frame_face_candidates.append(
                        RankedFaceCandidate(
                            captured_at_utc=packet.captured_at_utc,
                            frame_index=packet.frame_index,
                            face_frame=observed_face_frame.copy(),
                            face_quality=float(
                                observed_face.get("quality_score", 0.0)
                            ),
                        )
                    )
                if frame_face_candidates:
                    face_frame = frame_face_candidates[0].face_frame
                    face_quality = frame_face_candidates[0].face_quality
                    snapshot_quality = 0.70 + 0.30 * face_quality
            if human_crop is None and disappeared:
                human_crop = source_frame.copy()
            if human_crop is None:
                continue
            if not persist_human_log and disappeared:
                continue
            # body_image is a real crop of the tracked person. Face evidence is
            # stored separately and only its small thumbnail is embedded in JSON.
            snapshot_image = human_crop
            with self._lock:
                video_was_enabled = key in self._video_enabled_tracks
                if admitted_room_id is not None:
                    self._video_enabled_tracks.add(key)
                video_enabled = key in self._video_enabled_tracks
                starting_video = video_enabled and not video_was_enabled
                source_pre_roll = tuple(
                    self._pre_roll_frames.get((session_id, packet.source_id), ())
                )
                pre_roll_frames = (
                    source_pre_roll[-(self.video_pre_roll_frames + 1) : -1]
                    if starting_video and len(source_pre_roll) > 1
                    else ()
                )
                previous_name = self._observed_names.get(key)
                previous_score = self._candidate_scores.get(key, -1.0)
                previous_full_frame_at = self._last_full_frame_at.get(
                    key, float("-inf")
                )
                previous_face_video_at = self._last_face_video_at.get(
                    key, float("-inf")
                )
                previous_best_face = self._best_face_scores.get(key, -1.0)
                previous_evidence = self._still_evidence.get(key)
                previous_ranked_candidates = list(
                    self._face_candidates.get(key, ())
                )
                identity_changed = previous_name is None or (
                    previous_name == "Unknown" and name != "Unknown"
                )
                candidate_evidence = (
                    StillEvidenceBundle(
                        captured_at_utc=packet.captured_at_utc,
                        frame_index=packet.frame_index,
                        body_frame=snapshot_image.copy(),
                        full_frame=source_frame.copy(),
                        face_frame=face_frame.copy(),
                        quality=snapshot_quality,
                        face_quality=face_quality,
                    )
                    if face_frame is not None
                    else None
                )
                ranked_candidates = list(self._face_candidates.get(key, ()))
                ranked_candidates.extend(frame_face_candidates)
                ranked_candidates.sort(
                    key=lambda item: (-item.face_quality, item.frame_index)
                )
                ranked_candidates = ranked_candidates[: self.face_candidate_limit]
                if ranked_candidates:
                    self._face_candidates[key] = ranked_candidates
                better_snapshot = candidate_evidence is not None and (
                    previous_evidence is None
                    or candidate_evidence.face_quality > previous_evidence.face_quality
                    or (
                        candidate_evidence.face_quality
                        == previous_evidence.face_quality
                        and candidate_evidence.frame_index < previous_evidence.frame_index
                    )
                )
                full_frame_due = (
                    video_enabled
                    and not disappeared
                    and now - previous_full_frame_at >= 1.0 / self.video_fps
                )
                better_face = (
                    face_frame is not None
                    and face_quality > previous_best_face
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
                    self._still_evidence[key] = candidate_evidence
                if full_frame_due:
                    self._last_full_frame_at[key] = now
                if face_due:
                    self._last_face_video_at[key] = now
                if better_face:
                    self._best_face_scores[key] = face_quality
                selected_evidence = (
                    candidate_evidence
                    if better_snapshot and candidate_evidence is not None
                    else previous_evidence
                )
                selected_candidates = tuple(ranked_candidates) if disappeared else ()

            # Evidence-only observations do not create/update human-log rows,
            # but after a valid face admits the track they still feed sampled
            # person/face crops to the video writer.
            if not persist_human_log and not (full_frame_due or face_due):
                continue

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
                evidence_captured_at_utc=(
                    selected_evidence.captured_at_utc
                    if selected_evidence is not None
                    else packet.captured_at_utc
                ),
                recognition_score=float(human.get("recognition_score", 0.0) or 0.0),
                ref_img_id=raw_ref,
                personnel_id=raw_personnel_id,
                room_id=stored_room_id,
                room_ids=visited_room_ids,
                counts_for_attendance=bool(counts_for_attendance),
                snapshot_frame=(
                    selected_evidence.body_frame.copy()
                    if disappeared and selected_evidence is not None
                    else None
                ),
                whole_snapshot_frame=(
                    selected_evidence.full_frame.copy()
                    if disappeared and selected_evidence is not None
                    else None
                ),
                face_image_frame=(
                    selected_evidence.face_frame.copy()
                    if disappeared
                    and selected_evidence is not None
                    and selected_evidence.face_frame is not None
                    else None
                ),
                snapshot_quality=(
                    selected_evidence.quality
                    if selected_evidence is not None
                    else snapshot_quality
                ),
                full_frame_video_frame=(
                    source_frame.copy() if full_frame_due else None
                ),
                face_video_frame=face_frame if face_due else None,
                face_quality=(
                    selected_evidence.face_quality
                    if selected_evidence is not None
                    else face_quality
                ),
                pre_roll_frames=pre_roll_frames,
                evidence_frame_index=(
                    selected_evidence.frame_index
                    if selected_evidence is not None
                    else None
                ),
                ranked_face_candidates=selected_candidates,
                finalize_detection_log=(
                    persist_human_log
                    and disappeared
                    and stored_room_id is not None
                ),
                persist_human_log=persist_human_log,
            )
            try:
                if (
                    disappeared
                    and event.finalize_detection_log
                    and self.video_post_roll_frames > 0
                ):
                    with self._lock:
                        self._pending_finalizations.setdefault(
                            key,
                            PendingFinalization(
                                event=event,
                                remaining_frames=self.video_post_roll_frames,
                            ),
                        )
                elif disappeared:
                    # Finalization is lossless within the bounded queue: apply
                    # backpressure rather than dropping the only expiry event.
                    self._queue.put(event)
                else:
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
                        if previous_evidence is None:
                            self._still_evidence.pop(key, None)
                        else:
                            self._still_evidence[key] = previous_evidence
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
                    if previous_ranked_candidates:
                        self._face_candidates[key] = previous_ranked_candidates
                    else:
                        self._face_candidates.pop(key, None)
                    if starting_video:
                        self._video_enabled_tracks.discard(key)
                LOGGER.warning("Human media queue is full; newest frame was dropped")
            if disappeared and (
                selected_evidence is None or selected_evidence.face_frame is None
            ):
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
        return writer, self.media_storage.key_for_path(path)

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
            state.full_frame_writer, state.video_key = self._new_writer(
                self.video_dir,
                "human/videos",
                stem,
                state.full_frame_size,
            )
            self._created_human_videos += 1
        if event.face_video_frame is not None and state.face_writer is None:
            face_height, face_width = event.face_video_frame.shape[:2]
            state.face_size = (face_width, face_height)
            state.face_writer, state.face_video_key = self._new_writer(
                self.face_video_dir,
                "human/face_videos",
                f"{stem}_faces",
                state.face_size,
            )
            self._created_face_videos += 1
        return state

    def _save_snapshot(
        self, event: HumanMediaEvent, stem: str
    ) -> tuple[str, Path]:
        assert event.snapshot_frame is not None
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{stem}_body.jpg"
        path = self.snapshot_dir / filename
        if not cv2.imwrite(
            str(path),
            event.snapshot_frame,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        ):
            raise RuntimeError(f"Could not save human snapshot: {path}")
        return self.media_storage.key_for_path(path), path

    def _save_whole_snapshot(
        self, event: HumanMediaEvent, stem: str
    ) -> tuple[str, Path]:
        assert event.whole_snapshot_frame is not None
        self.whole_snapshot_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{stem}_full.jpg"
        path = self.whole_snapshot_dir / filename
        if not cv2.imwrite(
            str(path),
            event.whole_snapshot_frame,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        ):
            raise RuntimeError(f"Could not save whole snapshot: {path}")
        return self.media_storage.key_for_path(path), path

    def _save_face_image(
        self, event: HumanMediaEvent, stem: str
    ) -> tuple[str, str, Path]:
        assert event.face_image_frame is not None
        face_key, path = self.media_storage.save_jpeg(
            event.face_image_frame,
            directory="human/detected_faces",
            filename=f"{stem}_face.jpg",
            quality=92,
        )
        thumbnail_key, _ = self.media_storage.save_jpeg(
            event.face_image_frame,
            directory="human/face_thumbnails",
            filename=f"{stem}_thumbnail.jpg",
            quality=72,
            max_size=224,
        )
        LOGGER.info(
            "HUMAN_FACE_IMAGE_SAVED camera=%s track_id=%s key=%s thumbnail_key=%s",
            event.camera,
            event.track_id,
            face_key,
            thumbnail_key,
        )
        return face_key, thumbnail_key, path

    def _save_ranked_face_candidates(
        self,
        event: HumanMediaEvent,
        stem: str,
    ) -> None:
        if not event.ranked_face_candidates:
            return
        best = event.ranked_face_candidates[0]
        if event.evidence_frame_index != best.frame_index:
            raise RuntimeError(
                "Best face and still snapshots must originate from the same frame"
            )
        pending: list[tuple[Path, Path]] = []
        committed: list[Path] = []
        try:
            for rank, candidate in enumerate(event.ranked_face_candidates, start=1):
                filename = (
                    f"{stem}_rank_{rank:03d}_q_{candidate.face_quality:.6f}"
                    f"_frame_{candidate.frame_index:010d}.jpg"
                )
                path = self.face_video_dir / filename
                temporary = path.with_name(f".{path.stem}.pending.jpg")
                if not cv2.imwrite(
                    str(temporary),
                    candidate.face_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 92],
                ):
                    raise RuntimeError(
                        f"Could not save ranked face candidate: {path}"
                    )
                pending.append((temporary, path))
            for temporary, path in pending:
                temporary.replace(path)
                committed.append(path)
        except Exception:
            for temporary, _ in pending:
                temporary.unlink(missing_ok=True)
            for path in committed:
                path.unlink(missing_ok=True)
            raise
        self._saved_ranked_face_candidates += len(event.ranked_face_candidates)

    def _write(self, event: HumanMediaEvent) -> None:
        state = self._media_state(event)
        wrote_full_frame = 0
        wrote_face = False
        base_source_event_key = (
            f"human-track:{event.session_id}:{event.camera}:{event.track_id}"
        )
        logged_room_ids = event.room_ids or (
            (event.room_id,) if event.room_id is not None else ()
        )
        room_event_keys = [
            (
                int(room_id),
                base_source_event_key
                if index == 0
                else f"{base_source_event_key}:room:{int(room_id)}",
            )
            for index, room_id in enumerate(logged_room_ids)
        ]
        existing_detections = {
            room_id: self.detection_log_store.get_by_source_event_key(event_key)
            for room_id, event_key in room_event_keys
        } if event.finalize_detection_log and self.detection_log_store is not None else {}
        existing_detection = next(
            (
                detection
                for detection in existing_detections.values()
                if detection is not None
            ),
            None,
        )
        face_image_key = (
            str(existing_detection.face_image or "") if existing_detection else ""
        )
        face_thumbnail_key = (
            str(existing_detection.face_thumbnail or "") if existing_detection else ""
        )
        whole_snapshot_key = (
            str(existing_detection.snapshot_image or "") if existing_detection else ""
        )
        evidence_stem = self._safe_stem(event.camera, event.track_id)
        # Detection evidence is persisted once, when the track is finalized.
        # This avoids creating an orphan face/whole-frame file on every frame.
        if (
            event.finalize_detection_log
            and not face_image_key
            and event.face_image_frame is not None
        ):
            face_image_key, face_thumbnail_key, _ = self._save_face_image(
                event, evidence_stem
            )
        if (
            event.finalize_detection_log
            and not whole_snapshot_key
            and event.face_image_frame is not None
            and event.whole_snapshot_frame is not None
        ):
            whole_snapshot_key, _ = self._save_whole_snapshot(
                event, evidence_stem
            )
        if event.finalize_detection_log and event.ranked_face_candidates:
            self._save_ranked_face_candidates(event, evidence_stem)
        if event.pre_roll_frames and state.full_frame_writer is not None:
            for buffered_frame in event.pre_roll_frames:
                frame = buffered_frame
                if (
                    state.full_frame_size is not None
                    and (frame.shape[1], frame.shape[0]) != state.full_frame_size
                ):
                    frame = cv2.resize(frame, state.full_frame_size)
                state.full_frame_writer.write(frame)
                wrote_full_frame += 1
        if (
            event.full_frame_video_frame is not None
            and state.full_frame_writer is not None
            and state.full_frame_size is not None
        ):
            frame = event.full_frame_video_frame
            if (frame.shape[1], frame.shape[0]) != state.full_frame_size:
                frame = cv2.resize(frame, state.full_frame_size)
            state.full_frame_writer.write(frame)
            wrote_full_frame += 1
        if event.face_video_frame is not None and state.face_writer is not None:
            face_frame = event.face_video_frame
            if (
                state.face_size is not None
                and (face_frame.shape[1], face_frame.shape[0]) != state.face_size
            ):
                face_frame = cv2.resize(face_frame, state.face_size)
            state.face_writer.write(face_frame)
            wrote_face = True

        state.full_frame_frames += wrote_full_frame
        state.face_frames += int(wrote_face)

        if not event.persist_human_log:
            with self._lock:
                self._full_frame_video_frames += wrote_full_frame
                self._accepted_face_video_frames += int(wrote_face)
                self._last_error = None
            return

        new_snapshot_url = (
            str(existing_detection.body_image or "") if existing_detection else ""
        )
        new_snapshot_path: Path | None = None
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT snapshot_url, video_url, face_video_url, snapshot_quality,
                       best_face_quality, name, recognition_score, ref_img_id
                FROM human_logs
                WHERE session_id = ? AND camera = ? AND track_id = ?
                """,
                (event.session_id, event.camera, event.track_id),
            ).fetchone()
            current_snapshot_quality = (
                float(existing["snapshot_quality"] or 0.0) if existing else -1.0
            )
            should_save_snapshot = (
                event.finalize_detection_log
                and event.face_image_frame is not None
                and event.snapshot_frame is not None
                and not new_snapshot_url
                and not (str(existing["snapshot_url"] or "") if existing else "")
            )
            if should_save_snapshot:
                new_snapshot_url, new_snapshot_path = self._save_snapshot(
                    event, evidence_stem
                )
            snapshot_url = (
                new_snapshot_url
                if new_snapshot_url
                else str(existing["snapshot_url"] or "") if existing else ""
            )
            # The writer may have been closed by idle cleanup before the final
            # disappearance event. Preserve the keys already persisted in
            # human_logs instead of overwriting them with empty strings.
            video_key = state.video_key or (
                str(existing["video_url"] or "") if existing else ""
            )
            face_video_key = state.face_video_key or (
                str(existing["face_video_url"] or "") if existing else ""
            )
            snapshot_quality = (
                event.snapshot_quality
                if should_save_snapshot
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
                        video_key,
                        face_video_key,
                        snapshot_quality,
                        event.face_quality,
                        state.full_frame_frames,
                        state.face_frames,
                        event.personnel_id,
                        int(event.counts_for_attendance),
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
                        full_frame_video_frames = CASE
                            WHEN full_frame_video_frames < ? THEN ?
                            ELSE full_frame_video_frames END,
                        accepted_face_frames = CASE
                            WHEN accepted_face_frames < ? THEN ?
                            ELSE accepted_face_frames END,
                        personnel_id = CASE WHEN ? THEN ? ELSE personnel_id END
                    WHERE session_id = ? AND camera = ? AND track_id = ?
                    """,
                    (
                        stored_name,
                        event.captured_at_utc,
                        max(stored_recognition_score, 0.0),
                        None if stored_ref_img_id is None else str(stored_ref_img_id),
                        snapshot_url,
                        video_key,
                        face_video_key,
                        snapshot_quality,
                        bool(better_face),
                        event.face_quality,
                        state.full_frame_frames,
                        state.full_frame_frames,
                        state.face_frames,
                        state.face_frames,
                        bool(event.personnel_id is not None),
                        event.personnel_id,
                        event.session_id,
                        event.camera,
                        event.track_id,
                    ),
                )
            human_row = connection.execute(
                "SELECT id, snapshot_url, video_url, face_video_url "
                "FROM human_logs WHERE session_id = ? AND camera = ? AND track_id = ?",
                (event.session_id, event.camera, event.track_id),
            ).fetchone()
        if event.finalize_detection_log:
            # Release writers before exposing video URLs. MP4 metadata is not
            # guaranteed to be readable until VideoWriter.release() completes.
            self._release_state(state)
            state.full_frame_writer = None
            state.face_writer = None
            self._media.pop((event.session_id, event.camera, event.track_id), None)
        if (
            event.finalize_detection_log
            and self.detection_log_store is not None
            and face_image_key
        ):
            detection_person = self._detection_person(event.personnel_id)
            finalized_video_key = (
                str(human_row["video_url"] or "") if human_row is not None else video_key
            )
            finalized_face_video_key = (
                str(human_row["face_video_url"] or "")
                if human_row is not None
                else face_video_key
            )
            video_status = self._finalized_video_status(finalized_video_key)
            face_video_status = self._finalized_video_status(finalized_face_video_key)
            media_finalized_at = event.evidence_captured_at_utc
            for room_id, source_event_key in room_event_keys:
                room_detection = existing_detections.get(room_id)
                if room_detection is None:
                    self.detection_log_store.create(
                        source_system="face_recognition",
                        source_event_key=source_event_key,
                        source_human_log_id=(
                            int(human_row["id"]) if human_row else None
                        ),
                        personnel_id=event.personnel_id,
                        person=detection_person,
                        confidence=event.recognition_score,
                        detection_time=event.evidence_captured_at_utc,
                        ref_img_id=(
                            None
                            if event.ref_img_id is None
                            else str(event.ref_img_id)
                        ),
                        room_id=room_id,
                        camera_id=event.camera,
                        access_granted=event.personnel_id is not None,
                        counts_for_attendance=event.counts_for_attendance,
                        log_type="camera_rtsp",
                        face_image=face_image_key or None,
                        face_thumbnail=face_thumbnail_key or None,
                        body_image=snapshot_url or None,
                        snapshot_image=whole_snapshot_key or None,
                        video=finalized_video_key or None,
                        face_video_or_unknown_faces=(
                            finalized_face_video_key or None
                        ),
                        video_status=video_status,
                        face_video_status=face_video_status,
                        media_finalized_at=media_finalized_at,
                    )
                    continue
                updates: dict[str, Any] = {
                    "person": detection_person,
                    "video_status": video_status,
                    "face_video_status": face_video_status,
                    "media_finalized_at": media_finalized_at,
                }
                # Finalized media is shared by all room logs and remains immutable.
                candidates = {
                    "face_image": face_image_key,
                    "face_thumbnail": face_thumbnail_key,
                    "body_image": snapshot_url,
                    "snapshot_image": whole_snapshot_key,
                    "video": finalized_video_key,
                    "face_video_or_unknown_faces": finalized_face_video_key,
                }
                for field, value in candidates.items():
                    if value and not getattr(room_detection, field):
                        updates[field] = value
                if room_detection.room_id is None:
                    updates["room_id"] = room_id
                if event.camera and not room_detection.camera_id:
                    updates["camera_id"] = event.camera
                self.detection_log_store.update(room_detection.id, **updates)
        with self._lock:
            self._saved_snapshots += int(bool(new_snapshot_path))
            self._full_frame_video_frames += wrote_full_frame
            self._accepted_face_video_frames += int(wrote_face)
            if event.finalize_detection_log:
                key = (event.session_id, event.camera, event.track_id)
                self._still_evidence.pop(key, None)
                self._face_candidates.pop(key, None)
                self._video_enabled_tracks.discard(key)
                self._track_room_ids.pop(key, None)
                self._track_visited_room_ids.pop(key, None)
                self._observed_names.pop(key, None)
                self._candidate_scores.pop(key, None)
                self._last_full_frame_at.pop(key, None)
                self._last_face_video_at.pop(key, None)
                self._best_face_scores.pop(key, None)
            self._last_error = None

    def _finalized_video_status(self, key: str) -> str:
        return self.media_storage.finalized_video_status(key)

    @staticmethod
    def _release_state(state: TrackMediaState) -> None:
        if state.full_frame_writer is not None:
            state.full_frame_writer.release()
        if state.face_writer is not None:
            state.face_writer.release()

    def _close_idle_media(self) -> None:
        # A tracker can remain alive longer than ``video_idle_seconds`` while
        # detections are temporarily missed. Closing here splits one track into
        # multiple MP4 files and leaves the earlier segment orphaned. Writers
        # hold no frame buffer and are bounded by active tracker state; they are
        # released by the disappeared event or during shutdown.
        return None

    def _close_all_media(self) -> None:
        for state in self._media.values():
            self._release_state(state)
        self._media.clear()
        with self._lock:
            self._pre_roll_frames.clear()
            self._last_pre_roll_frame_index.clear()
            self._pre_roll_buffered_bytes = 0
            self._pending_finalizations.clear()

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
                conditions.append(f"h.{column} = ?")
                values.append(value)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        values.append(max(1, min(int(limit), 1000)))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT h.id, h.session_id, h.camera, h.track_id, h.name,
                       h.first_seen, h.last_seen, h.recognition_score,
                       h.ref_img_id, h.snapshot_url, h.video_url,
                       h.face_video_url, h.snapshot_quality,
                       h.best_face_quality, h.full_frame_video_frames,
                       h.accepted_face_frames, h.personnel_id,
                       h.counts_for_attendance, d.id AS detection_log_id,
                       d.video_status, d.face_video_status
                FROM human_logs h
                LEFT JOIN detection_logs d ON d.source_human_log_id = h.id
                """
                + where
                + " ORDER BY h.id DESC LIMIT ?",
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
            detection_log_id = item.get("detection_log_id")
            item["snapshot_url"] = (
                f"/api/v1/logs/{detection_log_id}/body"
                if detection_log_id and item.get("snapshot_url")
                else None
            )
            item["image_url"] = item["snapshot_url"]
            item["video_url"] = (
                f"/api/v1/logs/{detection_log_id}/video"
                if detection_log_id
                and item.get("video_url")
                and item.get("video_status") == MEDIA_STATUS_READY
                else None
            )
            item["face_video_url"] = (
                f"/api/v1/logs/{detection_log_id}/face-video"
                if detection_log_id
                and item.get("face_video_url")
                and item.get("face_video_status") == MEDIA_STATUS_READY
                else None
            )
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
                "saved_ranked_face_candidates": self._saved_ranked_face_candidates,
                "face_candidate_limit": self.face_candidate_limit,
                "ranked_candidate_tracks": len(self._face_candidates),
                "open_track_recorders": len(self._media),
                "video_fps": self.video_fps,
                "video_pre_roll_frames": self.video_pre_roll_frames,
                "video_post_roll_frames": self.video_post_roll_frames,
                "pre_roll_buffered_sources": len(self._pre_roll_frames),
                "pre_roll_buffered_frames": sum(
                    len(frames) for frames in self._pre_roll_frames.values()
                ),
                "pre_roll_buffered_bytes": self._pre_roll_buffered_bytes,
                "pre_roll_max_bytes": self.video_pre_roll_max_bytes,
                "pending_post_roll_tracks": len(self._pending_finalizations),
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
            pending_events = [
                pending.event for pending in self._pending_finalizations.values()
            ]
            self._pending_finalizations.clear()
        for event in pending_events:
            self._queue.put(event)
        self._queue.put(_STOP)
        self._thread.join(timeout=10.0)
