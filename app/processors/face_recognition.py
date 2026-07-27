from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Protocol, Sequence

import cv2
import numpy as np

from app.core.types import FramePacket, TaskName, TaskResult
from app.database import Database
from app.processors.base import BatchProcessor
from app.processors.ultralytics_loader import load_yolo_class, serialized_model_load

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FaceRecognitionSettings:
    human_model_path: Path
    face_model_path: Path
    embedding_model_path: Path
    device: str = "0"
    batch_size: int = 8
    human_imgsz: int = 640
    face_imgsz: int = 640
    human_engine_fixed_batch: int | None = 8
    face_engine_fixed_batch: int | None = 8
    human_confidence: float = 0.40
    face_confidence: float = 0.50
    recognition_threshold: float = 0.45
    min_face_width: int = 24
    min_face_height: int = 24
    blur_threshold: float = 20.0
    min_eye_distance: float = 8.0
    quality_threshold: float = 0.55
    max_abs_yaw: float = 45.0
    max_abs_pitch: float = 55.0
    max_abs_roll: float = 35.0
    require_landmarks: bool = True
    human_pose_enabled: bool = True
    human_pose_min_keypoints: int = 4
    human_pose_keypoint_confidence: float = 0.25
    recognition_quality_weight: float = 0.5
    tracker_high_threshold: float = 0.40
    tracker_low_threshold: float = 0.10
    tracker_new_threshold: float = 0.40
    tracker_match_threshold: float = 0.80
    tracker_max_missed: int = 30
    history_size: int = 30
    stable_min_hits: int = 3
    embedding_batch_size: int = 64
    vector_size: int = 512
    qdrant_collection: str = "faces"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    human_crop_padding_ratio: float = 0.08
    face_roi_mosaic_padding: int = 8
    face_roi_mosaic_max_width: int = 1920
    face_roi_mosaic_max_height: int = 1920
    use_gpu_quality: bool = True


@dataclass(frozen=True, slots=True)
class FaceMatch:
    person: str = "Unknown"
    score: float = 0.0
    ref_img_id: str | int | None = None


class FaceEmbedder(Protocol):
    def embed(self, faces: Sequence[np.ndarray]) -> np.ndarray: ...

    def close(self) -> None: ...


class FaceVectorStore(Protocol):
    def search_batch(self, embeddings: np.ndarray, threshold: float) -> list[FaceMatch]: ...

    def enroll(
        self,
        person: str,
        embedding: np.ndarray,
        ref_img_id: str | int | None,
    ) -> str: ...

    def identities(self, limit: int = 1000) -> list[dict[str, Any]]: ...

    def delete_person(self, person: str) -> int: ...

    def delete_points(self, point_ids: list[str]) -> int: ...

    def status(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _as_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(slots=True)
class TrackState:
    track_id: int
    bbox: list[float]
    missed: int = 0
    names: deque[str] = field(default_factory=deque)
    scores: deque[float] = field(default_factory=deque)
    ref_img_ids: deque[str | int | None] = field(default_factory=deque)
    stable_person: str = "Unknown"
    stable_score: float = 0.0
    stable_ref_img_id: str | int | None = None
    best_face_quality: float = 0.0


@dataclass(slots=True)
class HumanRoi:
    frame_index: int
    human_index: int
    track_id: int | None
    crop: np.ndarray
    crop_bbox: list[int]


@dataclass(slots=True)
class HumanRoiMosaicEntry:
    roi_index: int
    frame_index: int
    human_index: int
    track_id: int | None
    crop_bbox: list[int]
    x: int
    y: int
    width: int
    height: int


@dataclass(slots=True)
class HumanRoiMosaic:
    image: np.ndarray
    entries: list[HumanRoiMosaicEntry]


class _ByteDetections:
    """Minimal Results-like object consumed by Ultralytics BYTETracker."""

    def __init__(self, boxes: np.ndarray, confidences: np.ndarray) -> None:
        self.xyxy = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(confidences, dtype=np.float32).reshape(-1)
        self.cls = np.zeros((len(self.xyxy),), dtype=np.float32)

    @property
    def xywh(self) -> np.ndarray:
        values = self.xyxy.copy()
        values[:, 0] = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0
        values[:, 1] = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2.0
        values[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        values[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        return values

    def __len__(self) -> int:
        return len(self.xyxy)

    def __getitem__(self, index: Any) -> "_ByteDetections":
        return _ByteDetections(self.xyxy[index], self.conf[index])


class SourceFaceTracker:
    """One ByteTrack instance per source plus persistent face identity history."""

    def __init__(
        self,
        *,
        high_threshold: float,
        low_threshold: float,
        new_threshold: float,
        match_threshold: float,
        max_missed: int,
        history_size: int,
        stable_min_hits: int,
        backend: Any | None = None,
    ) -> None:
        self.max_missed = max(1, int(max_missed))
        self.history_size = max(1, int(history_size))
        self.stable_min_hits = max(1, int(stable_min_hits))
        if backend is None:
            try:
                from ultralytics.trackers.byte_tracker import BYTETracker
            except ImportError as exc:
                raise RuntimeError(
                    "Ultralytics BYTETracker and its 'lap' dependency are required"
                ) from exc
            backend = BYTETracker(
                SimpleNamespace(
                    track_high_thresh=float(high_threshold),
                    track_low_thresh=float(low_threshold),
                    new_track_thresh=float(new_threshold),
                    track_buffer=self.max_missed,
                    match_thresh=float(match_threshold),
                    fuse_score=True,
                )
            )
        self._backend = backend
        self._next_id = 1
        self._object_track_ids: dict[int, int] = {}
        self._visible_boxes: dict[int, list[float]] = {}
        self.tracks: dict[int, TrackState] = {}
        self._disappeared: list[TrackState] = []

    def update(
        self,
        boxes: Sequence[Sequence[float]],
        confidences: Sequence[float] | None = None,
    ) -> list[int | None]:
        box_values = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        confidence_values = np.asarray(
            confidences if confidences is not None else [1.0] * len(box_values),
            dtype=np.float32,
        )
        detections = _ByteDetections(box_values, confidence_values)
        self._backend.update(detections)

        assigned: list[int | None] = [None] * len(box_values)
        self._visible_boxes = {}
        self._disappeared = []
        active_objects = list(getattr(self._backend, "tracked_stracks", []))
        lost_objects = list(getattr(self._backend, "lost_stracks", []))
        alive_object_ids = {id(track) for track in active_objects + lost_objects}
        for object_id, local_id in list(self._object_track_ids.items()):
            if object_id not in alive_object_ids:
                self._object_track_ids.pop(object_id, None)
                state = self.tracks.pop(local_id, None)
                if state is not None:
                    self._disappeared.append(
                        TrackState(
                            track_id=state.track_id,
                            bbox=list(state.bbox),
                            missed=state.missed,
                            names=deque(state.names, maxlen=self.history_size),
                            scores=deque(state.scores, maxlen=self.history_size),
                            ref_img_ids=deque(
                                state.ref_img_ids, maxlen=self.history_size
                            ),
                            stable_person=state.stable_person,
                            stable_score=state.stable_score,
                            stable_ref_img_id=state.stable_ref_img_id,
                            best_face_quality=state.best_face_quality,
                        )
                    )

        for track in active_objects:
            if not bool(getattr(track, "is_activated", False)):
                continue
            detection_index = int(getattr(track, "idx", -1))
            if not 0 <= detection_index < len(assigned):
                continue
            object_id = id(track)
            track_id = self._object_track_ids.get(object_id)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._object_track_ids[object_id] = track_id
                self.tracks[track_id] = TrackState(
                    track_id=track_id,
                    bbox=[float(value) for value in box_values[detection_index]],
                    names=deque(maxlen=self.history_size),
                    scores=deque(maxlen=self.history_size),
                    ref_img_ids=deque(maxlen=self.history_size),
                )
            state = self.tracks[track_id]
            state.bbox = [float(value) for value in box_values[detection_index]]
            state.missed = 0
            assigned[detection_index] = track_id
            self._visible_boxes[track_id] = state.bbox

        visible_ids = set(self._visible_boxes)
        for track_id, state in self.tracks.items():
            if track_id not in visible_ids:
                state.missed += 1
        return assigned

    def consume_disappeared(self) -> list[TrackState]:
        disappeared = self._disappeared
        self._disappeared = []
        return disappeared

    def track_for_face(self, face_box: Sequence[float]) -> int | None:
        center_x = (float(face_box[0]) + float(face_box[2])) / 2.0
        center_y = (float(face_box[1]) + float(face_box[3])) / 2.0
        candidates: list[tuple[float, int]] = []
        for track_id, box in self._visible_boxes.items():
            x1, y1, x2, y2 = box
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                candidates.append(((x2 - x1) * (y2 - y1), track_id))
        if candidates:
            return min(candidates)[1]
        return None

    def observe(self, track_id: int, match: FaceMatch) -> FaceMatch:
        track = self.tracks.get(track_id)
        if track is None:
            return match
        track.names.append(match.person)
        track.scores.append(float(match.score))
        track.ref_img_ids.append(match.ref_img_id)
        if track.stable_person != "Unknown":
            if (
                match.person == track.stable_person
                and float(match.score) > track.stable_score
            ):
                track.stable_score = float(match.score)
                track.stable_ref_img_id = match.ref_img_id
            return FaceMatch(
                person=track.stable_person,
                score=track.stable_score,
                ref_img_id=track.stable_ref_img_id,
            )
        known_counts = Counter(name for name in track.names if name != "Unknown")
        if not known_counts:
            return FaceMatch()
        person, count = known_counts.most_common(1)[0]
        if count < self.stable_min_hits:
            return FaceMatch()
        indexes = [index for index, name in enumerate(track.names) if name == person]
        best_index = max(indexes, key=lambda index: track.scores[index])
        track.stable_person = person
        track.stable_score = float(track.scores[best_index])
        track.stable_ref_img_id = track.ref_img_ids[best_index]
        return self.identity(track_id)

    def identity(self, track_id: int | None) -> FaceMatch:
        if track_id is None:
            return FaceMatch()
        track = self.tracks.get(track_id)
        if track is None or track.stable_person == "Unknown":
            return FaceMatch()
        return FaceMatch(
            person=track.stable_person,
            score=track.stable_score,
            ref_img_id=track.stable_ref_img_id,
        )

    def record_face_quality(self, track_id: int | None, quality: float) -> None:
        if track_id is None:
            return
        track = self.tracks.get(int(track_id))
        if track is not None:
            track.best_face_quality = max(track.best_face_quality, float(quality))

    def best_quality(self, track_id: int | None) -> float:
        if track_id is None:
            return 0.0
        track = self.tracks.get(int(track_id))
        return round(float(track.best_face_quality), 6) if track is not None else 0.0

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "track_id": track.track_id,
                "bbox": list(track.bbox),
                "person": track.stable_person,
                "recognition_score": round(track.stable_score, 6),
                "best_face_quality": round(track.best_face_quality, 6),
                "ref_img_id": track.stable_ref_img_id,
                "visible": track.track_id in self._visible_boxes,
                "missed_frames": track.missed,
                "identity_history": list(track.names),
            }
            for track in self.tracks.values()
        ]


class OnnxFaceEmbedder:
    def __init__(self, model_path: Path, *, device: str, max_batch: int) -> None:
        self.lock = threading.RLock()
        self.session = None
        self.net = None
        self.backend = "onnxruntime"
        try:
            import onnxruntime as ort
        except ImportError:
            # The temporary portable deployment intentionally avoids installing
            # ONNX Runtime. OpenCV DNN can execute the old ArcFace ONNX model.
            self.backend = "opencv-dnn"
            self.net = cv2.dnn.readNetFromONNX(str(model_path))
            self.max_batch = 1
            self.static_batch = 1
            self.input = None
            self.output = None
        else:
            providers = ort.get_available_providers()
            requested = []
            if device.lower() != "cpu" and "CUDAExecutionProvider" in providers:
                requested.append("CUDAExecutionProvider")
            requested.append("CPUExecutionProvider")
            self.session = ort.InferenceSession(str(model_path), providers=requested)
            self.input = self.session.get_inputs()[0]
            self.output = self.session.get_outputs()[0]
            self.max_batch = max(1, int(max_batch))
            if isinstance(self.input.shape[0], int) and self.input.shape[0] > 0:
                self.max_batch = min(self.max_batch, int(self.input.shape[0]))
            self.static_batch = (
                int(self.input.shape[0])
                if isinstance(self.input.shape[0], int) and self.input.shape[0] > 0
                else None
            )

    @staticmethod
    def _blob(faces: Sequence[np.ndarray]) -> np.ndarray:
        values = [cv2.resize(face, (112, 112)) for face in faces]
        blob = cv2.dnn.blobFromImages(
            values,
            scalefactor=1.0 / 128.0,
            size=(112, 112),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        return np.ascontiguousarray(blob, dtype=np.float32)

    def embed(self, faces: Sequence[np.ndarray]) -> np.ndarray:
        outputs: list[np.ndarray] = []
        with self.lock:
            for start in range(0, len(faces), self.max_batch):
                chunk = list(faces[start : start + self.max_batch])
                real_count = len(chunk)
                if self.static_batch and real_count < self.static_batch:
                    chunk.extend(
                        [np.zeros_like(chunk[0])] * (self.static_batch - real_count)
                    )
                blob = self._blob(chunk)
                if self.session is not None:
                    assert self.input is not None and self.output is not None
                    embedding = self.session.run(
                        [self.output.name],
                        {self.input.name: blob},
                    )[0][:real_count]
                else:
                    assert self.net is not None
                    self.net.setInput(blob)
                    embedding = self.net.forward()[:real_count]
                outputs.append(
                    np.asarray(embedding, dtype=np.float32).reshape(real_count, -1)
                )
        return _normalize_embeddings(np.concatenate(outputs, axis=0))

    def close(self) -> None:
        return None


class TensorRTFaceEmbedder:
    """ArcFace TensorRT runner using Torch CUDA memory and one locked context."""

    def __init__(self, model_path: Path, *, device: str, max_batch: int) -> None:
        try:
            import tensorrt as trt
            import torch
        except ImportError as exc:
            raise RuntimeError("TensorRT and Torch are required for ArcFace engines") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("ArcFace TensorRT embeddings require a CUDA device")
        self.trt = trt
        self.torch = torch
        self.backend = "tensorrt"
        self.device = torch.device(f"cuda:{int(device)}" if device.isdigit() else "cuda:0")
        self.lock = threading.RLock()
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        with open(model_path, "rb") as f:
            engine_data = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_data)

        if self.engine is None:
            raise RuntimeError(f"Could not deserialize ArcFace engine: {model_path}")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_name = next(
            name
            for name in names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        )
        self.output_name = next(
            name
            for name in names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        )
        input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.input_torch_dtype = torch.from_numpy(
            np.empty((1,), dtype=input_dtype)
        ).dtype
        input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.static_batch = int(input_shape[0]) if input_shape[0] > 0 else None
        self.max_batch = max(1, int(max_batch))
        if self.static_batch:
            self.max_batch = min(self.max_batch, self.static_batch)
        else:
            _, _, profile_max = self.engine.get_tensor_profile_shape(self.input_name, 0)
            self.max_batch = min(self.max_batch, int(profile_max[0]))

    @staticmethod
    def _blob(faces: Sequence[np.ndarray]) -> np.ndarray:
        return OnnxFaceEmbedder._blob(faces)

    def embed(self, faces: Sequence[np.ndarray]) -> np.ndarray:
        outputs: list[np.ndarray] = []
        with self.lock, self.torch.cuda.device(self.device):
            for start in range(0, len(faces), self.max_batch):
                chunk = list(faces[start : start + self.max_batch])
                real_count = len(chunk)
                if self.static_batch and real_count < self.static_batch:
                    chunk.extend([np.zeros_like(chunk[0])] * (self.static_batch - real_count))
                batch = self.torch.from_numpy(self._blob(chunk)).to(
                    self.device, dtype=self.input_torch_dtype
                )
                if not self.static_batch:
                    if not self.context.set_input_shape(self.input_name, tuple(batch.shape)):
                        raise RuntimeError(
                            f"ArcFace engine rejected input shape {tuple(batch.shape)}"
                        )
                output_shape = tuple(self.context.get_tensor_shape(self.output_name))
                output_dtype = self.trt.nptype(
                    self.engine.get_tensor_dtype(self.output_name)
                )
                torch_dtype = self.torch.from_numpy(
                    np.empty((1,), dtype=output_dtype)
                ).dtype
                output = self.torch.empty(
                    output_shape,
                    dtype=torch_dtype,
                    device=self.device,
                )
                self.context.set_tensor_address(self.input_name, int(batch.data_ptr()))
                self.context.set_tensor_address(self.output_name, int(output.data_ptr()))
                stream = self.torch.cuda.current_stream(self.device)
                if not self.context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError("ArcFace TensorRT execute_async_v3 failed")
                stream.synchronize()
                values = output[:real_count].float().cpu().numpy().reshape(real_count, -1)
                outputs.append(values)
        return _normalize_embeddings(np.concatenate(outputs, axis=0))

    def close(self) -> None:
        with self.lock:
            self.context = None
            self.engine = None
            self.runtime = None


def _normalize_embeddings(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


class QdrantFaceStore:
    """Optional remote Qdrant backend, enabled only when FACE_QDRANT_URL is set."""

    def __init__(self, settings: FaceRecognitionSettings) -> None:
        if not settings.qdrant_url:
            raise ValueError("FACE_QDRANT_URL is required for the Qdrant backend")
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise RuntimeError("qdrant-client is required when FACE_QDRANT_URL is set") from exc
        self.models = models
        self.collection = settings.qdrant_collection
        self.vector_size = settings.vector_size
        self.lock = threading.RLock()
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self.mode = "qdrant-remote"
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        with self.lock:
            if self.client.collection_exists(self.collection):
                return
            try:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=self.models.VectorParams(
                        size=self.vector_size,
                        distance=self.models.Distance.COSINE,
                    ),
                )
            except Exception:
                if self.client.collection_exists(self.collection):
                    return
                raise
            LOGGER.info(
                "Created Qdrant face collection '%s' with vector size %d",
                self.collection,
                self.vector_size,
            )

    def search_batch(self, embeddings: np.ndarray, threshold: float) -> list[FaceMatch]:
        if len(embeddings) == 0:
            return []
        requests = [
            self.models.QueryRequest(
                query=embedding.tolist(),
                limit=1,
                score_threshold=float(threshold),
                with_payload=True,
            )
            for embedding in embeddings
        ]
        with self.lock:
            try:
                responses = self.client.query_batch_points(
                    collection_name=self.collection,
                    requests=requests,
                )
            except Exception:
                if self.client.collection_exists(self.collection):
                    raise
                self._ensure_collection()
                responses = self.client.query_batch_points(
                    collection_name=self.collection,
                    requests=requests,
                )
        matches: list[FaceMatch] = []
        for response in responses:
            points = list(getattr(response, "points", []) or [])
            if not points:
                matches.append(FaceMatch())
                continue
            point = points[0]
            payload = dict(getattr(point, "payload", {}) or {})
            matches.append(
                FaceMatch(
                    person=str(payload.get("person") or "Unknown"),
                    score=float(getattr(point, "score", 0.0) or 0.0),
                    ref_img_id=payload.get("ref_img_id"),
                )
            )
        return matches

    def enroll(
        self,
        person: str,
        embedding: np.ndarray,
        ref_img_id: str | int | None,
    ) -> str:
        point_id = str(uuid.uuid4())
        with self.lock:
            self._ensure_collection()
            self.client.upsert(
                collection_name=self.collection,
                wait=True,
                points=[
                    self.models.PointStruct(
                        id=point_id,
                        vector=np.asarray(embedding, dtype=np.float32).tolist(),
                        payload={"person": person, "ref_img_id": ref_img_id},
                    )
                ],
            )
        return point_id

    def identities(self, limit: int = 1000) -> list[dict[str, Any]]:
        counts: Counter[tuple[str, str | int | None]] = Counter()
        offset = None
        remaining = max(1, min(int(limit), 10000))
        with self.lock:
            while remaining > 0:
                records, offset = self.client.scroll(
                    collection_name=self.collection,
                    limit=min(256, remaining),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for record in records:
                    payload = dict(record.payload or {})
                    counts[(str(payload.get("person") or "Unknown"), payload.get("ref_img_id"))] += 1
                remaining -= len(records)
                if offset is None or not records:
                    break
        return [
            {"person": person, "ref_img_id": ref_img_id, "embeddings": count}
            for (person, ref_img_id), count in sorted(
                counts.items(), key=lambda item: (item[0][0], str(item[0][1]))
            )
        ]

    def delete_person(self, person: str) -> int:
        condition = self.models.FieldCondition(
            key="person",
            match=self.models.MatchValue(value=person),
        )
        query_filter = self.models.Filter(must=[condition])
        with self.lock:
            count = int(
                self.client.count(
                    collection_name=self.collection,
                    count_filter=query_filter,
                    exact=True,
                ).count
            )
            self.client.delete(
                collection_name=self.collection,
                points_selector=self.models.FilterSelector(filter=query_filter),
                wait=True,
            )
        return count

    def delete_points(self, point_ids: list[str]) -> int:
        if not point_ids:
            return 0
        with self.lock:
            self.client.delete(
                collection_name=self.collection,
                points_selector=point_ids,
                wait=True,
            )
        return len(point_ids)

    def status(self) -> dict[str, Any]:
        with self.lock:
            count = int(self.client.count(collection_name=self.collection).count)
        return {"mode": self.mode, "collection": self.collection, "points": count}

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


class PostgresFaceStore:
    """PostgreSQL-backed cosine store used as the default face-vector backend."""

    def __init__(self, database: Database, vector_size: int) -> None:
        self.database = database
        self.vector_size = int(vector_size)
        self.lock = threading.RLock()

    def search_batch(self, embeddings: np.ndarray, threshold: float) -> list[FaceMatch]:
        with self.lock, self.database.connection() as connection:
            rows = connection.execute(
                "SELECT person, ref_img_id, embedding, dimension FROM face_embeddings"
            ).fetchall()
        if not rows:
            return [FaceMatch() for _ in embeddings]
        vectors: list[np.ndarray] = []
        metadata: list[tuple[str, str | int | None]] = []
        for row in rows:
            dimension = int(row["dimension"])
            raw = bytes(row["embedding"])
            vector = np.frombuffer(raw, dtype=np.float32, count=dimension)
            if len(vector) != self.vector_size:
                continue
            vectors.append(vector.copy())
            metadata.append((str(row["person"]), row["ref_img_id"]))
        if not vectors:
            return [FaceMatch() for _ in embeddings]
        matrix = _normalize_embeddings(np.stack(vectors))
        queries = _normalize_embeddings(embeddings)
        scores = queries @ matrix.T
        matches: list[FaceMatch] = []
        for values in scores:
            index = int(np.argmax(values))
            score = float(values[index])
            if score < float(threshold):
                matches.append(FaceMatch())
            else:
                person, ref_img_id = metadata[index]
                matches.append(FaceMatch(person, score, ref_img_id))
        return matches

    def enroll(
        self,
        person: str,
        embedding: np.ndarray,
        ref_img_id: str | int | None,
    ) -> str:
        point_id = str(uuid.uuid4())
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if len(vector) != self.vector_size:
            raise ValueError(
                f"Embedding dimension is {len(vector)}; expected {self.vector_size}"
            )
        with self.lock, self.database.connection() as connection:
            connection.execute(
                "INSERT INTO face_embeddings "
                "(id, person, ref_img_id, embedding, dimension) VALUES (?, ?, ?, ?, ?)",
                (
                    point_id,
                    person,
                    None if ref_img_id is None else str(ref_img_id),
                    vector.tobytes(),
                    len(vector),
                ),
            )
        return point_id

    def identities(self, limit: int = 1000) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 10000))
        with self.lock, self.database.connection() as connection:
            rows = connection.execute(
                "SELECT person, ref_img_id, COUNT(*) AS embeddings "
                "FROM face_embeddings GROUP BY person, ref_img_id "
                "ORDER BY person, ref_img_id LIMIT ?",
                (bounded_limit,),
            ).fetchall()
        return [
            {
                "person": str(row["person"]),
                "ref_img_id": row["ref_img_id"],
                "embeddings": int(row["embeddings"]),
            }
            for row in rows
        ]

    def delete_person(self, person: str) -> int:
        with self.lock, self.database.connection() as connection:
            cursor = connection.execute(
                "DELETE FROM face_embeddings WHERE person = ?",
                (person,),
            )
            return int(cursor.rowcount)

    def delete_points(self, point_ids: list[str]) -> int:
        if not point_ids:
            return 0
        placeholders = ",".join("?" for _ in point_ids)
        with self.lock, self.database.connection() as connection:
            cursor = connection.execute(
                f"DELETE FROM face_embeddings WHERE id IN ({placeholders})",
                point_ids,
            )
            return int(cursor.rowcount)

    def status(self) -> dict[str, Any]:
        with self.lock, self.database.connection() as connection:
            count = int(
                connection.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0]
            )
        return {"mode": "postgresql", "points": count}

    def close(self) -> None:
        return None


class FaceRecognitionProcessor(BatchProcessor):
    task = TaskName.FACE_RECOGNITION

    def __init__(
        self,
        settings: FaceRecognitionSettings,
        *,
        human_detector: Any | None = None,
        face_detector: Any | None = None,
        embedder: FaceEmbedder | None = None,
        vector_store: FaceVectorStore | None = None,
        database: Database | None = None,
        tracker_backend_factory: Callable[[], Any] | None = None,
        settings_provider: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self.settings = settings
        self._human_detector = human_detector
        self._face_detector = face_detector
        self._embedder = embedder
        self._vector_store = vector_store
        self._database = database
        self._tracker_backend_factory = tracker_backend_factory
        self._settings_provider = settings_provider
        self._load_lock = threading.RLock()
        self._trackers: dict[str, SourceFaceTracker] = {}
        self._tracking_session_id = uuid.uuid4().hex
        self._processed_batches = 0
        self._processed_frames = 0
        self._detected_faces = 0
        self._recognized_faces = 0
        self._human_candidates = 0
        self._filtered_humans = 0
        self._last_batch_ms = 0.0
        self._last_detection_ms = 0.0
        self._last_human_detection_ms = 0.0
        self._last_face_roi_detection_ms = 0.0
        self._last_quality_ms = 0.0
        self._last_quality_backend = "cpu"
        self._last_embedding_ms = 0.0
        self._last_search_ms = 0.0
        self._last_error: str | None = None
        self._vector_store_warning: str | None = None

    def _source_thresholds(self, source_id: str) -> tuple[float, float, float]:
        if self._settings_provider is None:
            return (
                float(self.settings.human_confidence),
                float(self.settings.face_confidence),
                float(self.settings.recognition_threshold),
            )
        try:
            values = self._settings_provider(source_id)
            return (
                float(values.get("face_human_confidence", self.settings.human_confidence)),
                float(values.get("face_detection_confidence", self.settings.face_confidence)),
                float(values.get("face_recognition_threshold", self.settings.recognition_threshold)),
            )
        except Exception as exc:
            LOGGER.warning("Face source settings unavailable for %s: %s", source_id, exc)
            return (
                float(self.settings.human_confidence),
                float(self.settings.face_confidence),
                float(self.settings.recognition_threshold),
            )

    def _ensure_dependencies(self) -> None:
        with self._load_lock:
            required_paths = []
            if self._human_detector is None:
                required_paths.append(self.settings.human_model_path)
            if self._face_detector is None:
                required_paths.append(self.settings.face_model_path)
            if self._embedder is None:
                required_paths.append(self.settings.embedding_model_path)
            missing = [path for path in required_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Face-recognition model file(s) missing: "
                    + ", ".join(str(path) for path in missing)
                )
            if self._human_detector is None or self._face_detector is None:
                with serialized_model_load():
                    YOLO = load_yolo_class()
                    if self._human_detector is None:
                        self._human_detector = YOLO(
                            str(self.settings.human_model_path), task="pose"
                        )
                    if self._face_detector is None:
                        self._face_detector = YOLO(
                            str(self.settings.face_model_path), task="pose"
                        )
            if self._embedder is None:
                suffix = self.settings.embedding_model_path.suffix.lower()
                if suffix == ".engine":
                    self._embedder = TensorRTFaceEmbedder(
                        self.settings.embedding_model_path,
                        device=self.settings.device,
                        max_batch=self.settings.embedding_batch_size,
                    )
                elif suffix == ".onnx":
                    self._embedder = OnnxFaceEmbedder(
                        self.settings.embedding_model_path,
                        device=self.settings.device,
                        max_batch=self.settings.embedding_batch_size,
                    )
                else:
                    raise ValueError("FACE_EMBEDDING_MODEL must be .engine or .onnx")
            if self._vector_store is None:
                if self.settings.qdrant_url:
                    try:
                        self._vector_store = QdrantFaceStore(self.settings)
                        self._vector_store_warning = None
                    except Exception as exc:
                        if self._database is None:
                            raise
                        self._vector_store = PostgresFaceStore(
                            self._database,
                            self.settings.vector_size,
                        )
                        self._vector_store_warning = (
                            "Qdrant unavailable; using PostgreSQL fallback: "
                            f"{type(exc).__name__}: {exc}"
                        )
                else:
                    if self._database is None:
                        raise RuntimeError(
                            "PostgreSQL database is required for the default face-vector store"
                        )
                    self._vector_store = PostgresFaceStore(
                        self._database,
                        self.settings.vector_size,
                    )
                    self._vector_store_warning = None

    def preload(self) -> None:
        try:
            self._ensure_dependencies()
            # Import the tracker stack on the main startup thread. This repo
            # intentionally avoids concurrent lazy Ultralytics imports because
            # native ML dependencies have previously crashed during startup.
            from ultralytics.trackers.byte_tracker import BYTETracker  # noqa: F401

            self._last_error = None
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise

    def _predict_yolo(
        self,
        model: Any,
        frames: Sequence[np.ndarray],
        *,
        model_path: Path,
        imgsz: int,
        confidence: float,
        fixed_batch: int | None,
    ) -> list[Any]:
        """Run a YOLO model in chunks.

        TensorRT engines in this project are often exported with a fixed batch
        size. The old implementation raised an error when the real batch was
        larger than that fixed size. For real-time multi-camera processing we
        instead split the input into multiple chunks and still pad only the last
        chunk when the engine requires a static batch.
        """
        real_count = len(frames)
        if real_count == 0:
            return []

        engine_fixed_batch = (
            int(fixed_batch)
            if model_path.suffix.lower() == ".engine" and fixed_batch
            else None
        )
        max_real_batch = engine_fixed_batch or max(1, int(self.settings.batch_size))
        output: list[Any] = []

        for start in range(0, real_count, max_real_batch):
            chunk = list(frames[start : start + max_real_batch])
            chunk_real_count = len(chunk)
            source = list(chunk)
            batch = chunk_real_count

            if engine_fixed_batch:
                batch = engine_fixed_batch
                while len(source) < engine_fixed_batch:
                    source.append(np.zeros_like(source[0]))
     
            results = list(
                model.predict(
                    source=source,
                    batch=batch,
                    imgsz=imgsz,
                    conf=confidence,
                    device=self.settings.device,
                    rect=False,
                    verbose=False,
                )
            )
            if len(results) < chunk_real_count:
                raise RuntimeError(
                    f"{model_path.name} returned fewer results than input frames"
                )
            output.extend(results[:chunk_real_count])

        return output[:real_count]


    @staticmethod
    def _boxes(result: Any, confidence: float) -> list[dict[str, Any]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        xyxy = _as_numpy(getattr(boxes, "xyxy", None))
        confs = _as_numpy(getattr(boxes, "conf", None)).reshape(-1)
        classes = _as_numpy(getattr(boxes, "cls", None)).reshape(-1)
        output: list[dict[str, Any]] = []
        for index, box in enumerate(xyxy):
            score = float(confs[index]) if index < len(confs) else 0.0
            class_id = int(classes[index]) if index < len(classes) else 0
            if class_id == 0 and score >= confidence:
                output.append(
                    {
                        "bbox": [float(value) for value in box[:4]],
                        "confidence": score,
                        "_result_index": index,
                    }
                )
        return output

    def _human_boxes(
        self,
        result: Any,
        confidence: float | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        candidates = self._boxes(
            result,
            self.settings.human_confidence if confidence is None else confidence,
        )
        if not self.settings.human_pose_enabled:
            for candidate in candidates:
                candidate.update(
                    {
                        "human_pose_valid": False,
                        "human_pose_score": None,
                        "human_pose_keypoint_count": 0,
                        "human_pose_reason": "disabled",
                    }
                )
            return candidates, []

        keypoints = getattr(result, "keypoints", None)
        xy = _as_numpy(getattr(keypoints, "xy", None)) if keypoints is not None else np.empty((0,))
        confidence = (
            _as_numpy(getattr(keypoints, "conf", None))
            if keypoints is not None
            else np.empty((0,))
        )
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for candidate in candidates:
            result_index = int(candidate.get("_result_index", -1))
            points = (
                np.asarray(xy[result_index], dtype=np.float32)
                if xy.ndim == 3 and 0 <= result_index < len(xy)
                else np.empty((0, 2), dtype=np.float32)
            )
            point_confidence = (
                np.asarray(confidence[result_index], dtype=np.float32).reshape(-1)
                if confidence.ndim >= 2 and 0 <= result_index < len(confidence)
                else np.empty((0,), dtype=np.float32)
            )
            finite = np.isfinite(points).all(axis=1) if len(points) else np.empty((0,), dtype=bool)
            if len(point_confidence) != len(points) or not len(points):
                visible = np.zeros(len(points), dtype=bool)
                pose_score = 0.0
            else:
                visible = finite & (point_confidence >= self.settings.human_pose_keypoint_confidence)
                pose_score = float(np.mean(point_confidence[visible])) if np.any(visible) else 0.0
            keypoint_count = int(np.count_nonzero(visible))
            valid = keypoint_count >= self.settings.human_pose_min_keypoints
            candidate.update(
                {
                    "human_pose_valid": valid,
                    "human_pose_score": round(pose_score, 6),
                    "human_pose_keypoint_count": keypoint_count,
                    "human_pose_reason": "ok" if valid else "insufficient_keypoints",
                }
            )
            if valid:
                accepted.append(candidate)
            else:
                rejected.append(
                    {
                        "bbox": list(candidate["bbox"]),
                        "confidence": float(candidate["confidence"]),
                        "human_pose_score": round(pose_score, 6),
                        "human_pose_keypoint_count": keypoint_count,
                        "reason": "insufficient_keypoints",
                    }
                )
        return accepted, rejected

    def _faces(self, result: Any, confidence: float | None = None) -> list[dict[str, Any]]:
        faces = self._boxes(
            result,
            self.settings.face_confidence if confidence is None else confidence,
        )
        keypoints = getattr(result, "keypoints", None)
        values = _as_numpy(getattr(keypoints, "xy", None)) if keypoints is not None else np.empty((0,))
        for face in faces:
            result_index = int(face.pop("_result_index", 0))
            face["landmarks"] = (
                np.asarray(values[result_index], dtype=np.float32)
                if values.ndim == 3 and result_index < len(values)
                else np.empty((0, 2), dtype=np.float32)
            )
        return faces

    @staticmethod
    def _scale_bbox(
        bbox: Sequence[float],
        inference_shape: Sequence[int],
        source_shape: Sequence[int],
    ) -> list[float]:
        inference_height, inference_width = (
            int(inference_shape[0]),
            int(inference_shape[1]),
        )
        source_height, source_width = int(source_shape[0]), int(source_shape[1])
        scale_x = source_width / max(float(inference_width), 1.0)
        scale_y = source_height / max(float(inference_height), 1.0)
        return [
            float(bbox[0]) * scale_x,
            float(bbox[1]) * scale_y,
            float(bbox[2]) * scale_x,
            float(bbox[3]) * scale_y,
        ]

    @classmethod
    def _source_face(
        cls,
        face: dict[str, Any],
        inference_frame: np.ndarray,
        source_frame: np.ndarray,
    ) -> dict[str, Any]:
        landmarks = np.asarray(face.get("landmarks", []), dtype=np.float32)
        scale_x = source_frame.shape[1] / max(float(inference_frame.shape[1]), 1.0)
        scale_y = source_frame.shape[0] / max(float(inference_frame.shape[0]), 1.0)
        source_landmarks = landmarks.copy()
        if source_landmarks.ndim == 2 and source_landmarks.shape[1] >= 2:
            source_landmarks[:, 0] *= scale_x
            source_landmarks[:, 1] *= scale_y
        return {
            **face,
            "bbox": cls._scale_bbox(
                face["bbox"], inference_frame.shape, source_frame.shape
            ),
            "landmarks": source_landmarks,
        }

    def _quality(
        self,
        frame: np.ndarray,
        face: dict[str, Any],
    ) -> tuple[bool, float, str, np.ndarray | None, dict[str, Any]]:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in face["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        crop = frame[y1:y2, x1:x2]
        metrics: dict[str, Any] = {
            "blur": 0.0,
            "eye_distance": 0.0,
            "yaw": None,
            "pitch": None,
            "roll": None,
            "landmark_count": 0,
            "size_score": 0.0,
            "blur_score": 0.0,
            "pose_score": 0.0,
            "face_width": 0,
            "face_height": 0,
            "quality_frame_width": int(width),
            "quality_frame_height": int(height),
        }
        if crop.size == 0:
            return False, 0.0, "empty_crop", None, metrics
        face_height, face_width = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        landmarks = np.asarray(face["landmarks"], dtype=np.float32)
        metrics["blur"] = round(blur, 4)
        metrics["landmark_count"] = int(len(landmarks))
        metrics["face_width"] = int(face_width)
        metrics["face_height"] = int(face_height)
        width_score = min(
            1.0,
            face_width / max(float(self.settings.min_face_width * 2), 1.0),
        )
        height_score = min(
            1.0,
            face_height / max(float(self.settings.min_face_height * 2), 1.0),
        )
        metrics["size_score"] = round(
            min(width_score, height_score),
            6,
        )
        metrics["blur_score"] = round(
            min(1.0, blur / max(self.settings.blur_threshold * 4.0, 1.0)),
            6,
        )
        eye_distance = 0.0
        if len(landmarks) >= 2:
            eye_distance = float(np.linalg.norm(landmarks[0] - landmarks[1]))
        metrics["eye_distance"] = round(eye_distance, 4)
        eye_score = min(
            1.0,
            eye_distance / max(float(self.settings.min_eye_distance * 2), 1.0),
        )

        yaw = pitch = roll = None
        if len(landmarks) >= 5:
            yaw, pitch, roll = self._head_pose(landmarks[:5], frame.shape)
        metrics.update(
            {
                "yaw": None if yaw is None else round(yaw, 4),
                "pitch": None if pitch is None else round(pitch, 4),
                "roll": None if roll is None else round(roll, 4),
            }
        )
        pose_parts = []
        for value, maximum in (
            (yaw, self.settings.max_abs_yaw),
            (pitch, self.settings.max_abs_pitch),
            (roll, self.settings.max_abs_roll),
        ):
            if value is not None:
                pose_parts.append(max(0.0, 1.0 - abs(value) / max(maximum, 1e-6)))
        pose_score = sum(pose_parts) / len(pose_parts) if pose_parts else 0.0
        metrics["pose_score"] = round(pose_score, 6)
        confidence = float(face.get("confidence", 0.0) or 0.0)
        quality = (
            0.30 * float(metrics["blur_score"])
            + 0.20 * float(metrics["size_score"])
            + 0.15 * eye_score
            + 0.25 * pose_score
            + 0.10 * confidence
        )
        quality = max(0.0, min(1.0, quality))

        reason = "ok"
        if (
            face_width < self.settings.min_face_width
            or face_height < self.settings.min_face_height
        ):
            reason = "face_too_small"
        elif self.settings.require_landmarks and len(landmarks) < 5:
            reason = "missing_landmarks"
        elif blur < self.settings.blur_threshold:
            reason = "blurry"
        elif eye_distance < self.settings.min_eye_distance:
            reason = "eyes_too_close"
        elif self.settings.require_landmarks and None in (yaw, pitch, roll):
            reason = "pose_unavailable"
        elif yaw is not None and abs(yaw) > self.settings.max_abs_yaw:
            reason = "yaw_out_of_range"
        elif pitch is not None and abs(pitch) > self.settings.max_abs_pitch:
            reason = "pitch_out_of_range"
        elif roll is not None and abs(roll) > self.settings.max_abs_roll:
            reason = "roll_out_of_range"
        elif quality < self.settings.quality_threshold:
            reason = "quality_below_threshold"
        valid = reason == "ok"
        aligned = crop
        if valid:
            corrected = self._align_face(frame, face["bbox"], landmarks)
            if corrected is None:
                valid = False
                reason = "alignment_failed"
            else:
                aligned = corrected
        return valid, quality, reason, aligned, metrics

    @staticmethod
    def _head_pose(
        landmarks: np.ndarray,
        frame_shape: Sequence[int],
    ) -> tuple[float | None, float | None, float | None]:
        """Estimate yaw, pitch, and roll in degrees from five face landmarks."""
        if landmarks.shape[0] < 5:
            return None, None, None
        height, width = int(frame_shape[0]), int(frame_shape[1])
        model_points = np.asarray(
            [
                [-30.0, 35.0, 30.0],
                [30.0, 35.0, 30.0],
                [0.0, 0.0, 0.0],
                [-25.0, -30.0, 30.0],
                [25.0, -30.0, 30.0],
            ],
            dtype=np.float64,
        )
        focal_length = float(max(width, height))
        camera_matrix = np.asarray(
            [
                [focal_length, 0.0, width / 2.0],
                [0.0, focal_length, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        try:
            success, rotation_vector, _ = cv2.solvePnP(
                model_points,
                landmarks[:5].astype(np.float64),
                camera_matrix,
                np.zeros((4, 1), dtype=np.float64),
                flags=cv2.SOLVEPNP_SQPNP,
            )
            if not success:
                return None, None, None
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            pitch, yaw, roll = cv2.RQDecomp3x3(rotation_matrix)[0]

            def normalized(value: float) -> float:
                value = ((float(value) + 180.0) % 360.0) - 180.0
                if value > 90.0:
                    return 180.0 - value
                if value < -90.0:
                    return -180.0 - value
                return value

            return normalized(yaw), normalized(pitch), normalized(roll)
        except cv2.error:
            return None, None, None

    @staticmethod
    def _align_face(
        frame: np.ndarray,
        bbox: Sequence[float],
        landmarks: np.ndarray,
    ) -> np.ndarray | None:
        if landmarks.shape[0] >= 5:
            reference = np.array(
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
                landmarks[:5].astype(np.float32),
                reference,
                method=cv2.LMEDS,
            )
            if transform is not None:
                return cv2.warpAffine(frame, transform, (112, 112))
            return None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in bbox[:4]]
        crop = frame[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]
        return cv2.resize(crop, (112, 112)) if crop.size else None

    @staticmethod
    def _clip_bbox_with_padding(
        bbox: Sequence[float],
        frame_shape: Sequence[int],
        padding_ratio: float,
    ) -> list[int]:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        pad_x = box_width * max(0.0, float(padding_ratio))
        pad_y = box_height * max(0.0, float(padding_ratio))
        return [
            max(0, int(np.floor(x1 - pad_x))),
            max(0, int(np.floor(y1 - pad_y))),
            min(width, int(np.ceil(x2 + pad_x))),
            min(height, int(np.ceil(y2 + pad_y))),
        ]

    @staticmethod
    def _landmarks_to_list(values: Any) -> list[list[float]]:
        landmarks = np.asarray(values, dtype=np.float32)
        if landmarks.ndim != 2 or landmarks.shape[1] < 2:
            return []
        return landmarks[:, :2].round(3).tolist()

    def _build_human_rois(
        self,
        packets: Sequence[FramePacket],
        frame_payloads: Sequence[dict[str, Any]],
    ) -> list[HumanRoi]:
        rois: list[HumanRoi] = []
        padding_ratio = float(self.settings.human_crop_padding_ratio)
        for frame_index, (packet, payload) in enumerate(zip(packets, frame_payloads)):
            frame = packet.frame
            for human_index, human in enumerate(payload["humans"]):
                track_id = human.get("track_id")
                if track_id is None:
                    continue
                x1, y1, x2, y2 = self._clip_bbox_with_padding(
                    human["bbox"], frame.shape, padding_ratio
                )
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                rois.append(
                    HumanRoi(
                        frame_index=frame_index,
                        human_index=human_index,
                        track_id=int(track_id),
                        crop=crop,
                        crop_bbox=[x1, y1, x2, y2],
                    )
                )
        return rois

    def _build_human_roi_mosaics(self, rois: Sequence[HumanRoi]) -> list[HumanRoiMosaic]:
        """Concatenate human crops into one or more mosaic images.

        Every detected face is later mapped back from mosaic coordinates to the
        original inference-frame coordinates using the entry metadata.
        """
        if not rois:
            return []

        padding = max(0, int(self.settings.face_roi_mosaic_padding))
        max_width = max(1, int(self.settings.face_roi_mosaic_max_width))
        max_height = max(1, int(self.settings.face_roi_mosaic_max_height))
        mosaics: list[HumanRoiMosaic] = []
        current_entries: list[HumanRoiMosaicEntry] = []
        current_rows: list[tuple[HumanRoi, int, int, int]] = []
        current_width = 1
        current_height = padding

        def flush() -> None:
            nonlocal current_entries, current_rows, current_width, current_height
            if not current_rows:
                return
            channels = int(current_rows[0][0].crop.shape[2]) if current_rows[0][0].crop.ndim == 3 else 1
            image = np.zeros((max(current_height, 1), max(current_width, 1), channels), dtype=current_rows[0][0].crop.dtype)
            if channels == 1:
                image = image.reshape(max(current_height, 1), max(current_width, 1))
            for roi, x, y, roi_index in current_rows:
                crop_height, crop_width = roi.crop.shape[:2]
                image[y : y + crop_height, x : x + crop_width] = roi.crop
                current_entries.append(
                    HumanRoiMosaicEntry(
                        roi_index=roi_index,
                        frame_index=roi.frame_index,
                        human_index=roi.human_index,
                        track_id=roi.track_id,
                        crop_bbox=list(roi.crop_bbox),
                        x=x,
                        y=y,
                        width=crop_width,
                        height=crop_height,
                    )
                )
            mosaics.append(HumanRoiMosaic(image=image, entries=current_entries))
            current_entries = []
            current_rows = []
            current_width = 1
            current_height = padding

        for roi_index, roi in enumerate(rois):
            crop_height, crop_width = roi.crop.shape[:2]
            if crop_width <= 0 or crop_height <= 0:
                continue

            # If one crop is wider/taller than the configured mosaic limits,
            # resize it while preserving aspect ratio. This keeps YOLO input
            # bounded and still allows exact coordinate mapping through the
            # stored resized crop bbox.
            if crop_width > max_width - 2 * padding or crop_height > max_height - 2 * padding:
                scale = min(
                    (max_width - 2 * padding) / max(float(crop_width), 1.0),
                    (max_height - 2 * padding) / max(float(crop_height), 1.0),
                )
                scale = max(scale, 1e-3)
                new_width = max(1, int(round(crop_width * scale)))
                new_height = max(1, int(round(crop_height * scale)))
                resized = cv2.resize(roi.crop, (new_width, new_height))
                roi = HumanRoi(
                    frame_index=roi.frame_index,
                    human_index=roi.human_index,
                    track_id=roi.track_id,
                    crop=resized,
                    crop_bbox=roi.crop_bbox,
                )
                crop_height, crop_width = roi.crop.shape[:2]

            next_height = current_height + crop_height + padding
            next_width = max(current_width, crop_width + 2 * padding)
            if current_rows and (next_height > max_height or next_width > max_width):
                flush()
                next_height = padding + crop_height + padding
                next_width = crop_width + 2 * padding

            x = padding
            y = current_height
            current_rows.append((roi, x, y, roi_index))
            current_height = next_height
            current_width = max(current_width, next_width)

        flush()
        return mosaics

    def _faces_from_human_roi_mosaics(
        self,
        mosaics: Sequence[HumanRoiMosaic],
        face_results: Sequence[Any],
        face_confidences: Sequence[float] | None = None,
    ) -> list[dict[str, Any]]:
        mapped_faces: list[dict[str, Any]] = []
        for mosaic, result in zip(mosaics, face_results):
            minimum_confidence = min(
                face_confidences or [self.settings.face_confidence]
            )
            faces = self._faces(result, minimum_confidence)
            for face in faces:
                bbox = [float(value) for value in face["bbox"]]
                center_x = (bbox[0] + bbox[2]) / 2.0
                center_y = (bbox[1] + bbox[3]) / 2.0
                entry = next(
                    (
                        item
                        for item in mosaic.entries
                        if item.x <= center_x <= item.x + item.width
                        and item.y <= center_y <= item.y + item.height
                    ),
                    None,
                )
                if entry is None:
                    continue
                if (
                    face_confidences is not None
                    and float(face.get("confidence", 0.0))
                    < float(face_confidences[entry.frame_index])
                ):
                    continue

                local_bbox = [
                    max(0.0, min(float(entry.width), bbox[0] - entry.x)),
                    max(0.0, min(float(entry.height), bbox[1] - entry.y)),
                    max(0.0, min(float(entry.width), bbox[2] - entry.x)),
                    max(0.0, min(float(entry.height), bbox[3] - entry.y)),
                ]
                if local_bbox[2] <= local_bbox[0] or local_bbox[3] <= local_bbox[1]:
                    continue

                crop_x1, crop_y1, crop_x2, crop_y2 = entry.crop_bbox
                crop_width = max(1.0, float(entry.width))
                crop_height = max(1.0, float(entry.height))
                original_crop_width = max(1.0, float(crop_x2 - crop_x1))
                original_crop_height = max(1.0, float(crop_y2 - crop_y1))
                scale_x = original_crop_width / crop_width
                scale_y = original_crop_height / crop_height

                frame_bbox = [
                    crop_x1 + local_bbox[0] * scale_x,
                    crop_y1 + local_bbox[1] * scale_y,
                    crop_x1 + local_bbox[2] * scale_x,
                    crop_y1 + local_bbox[3] * scale_y,
                ]

                landmarks = np.asarray(face.get("landmarks", []), dtype=np.float32)
                frame_landmarks = np.empty((0, 2), dtype=np.float32)
                if landmarks.ndim == 2 and landmarks.shape[1] >= 2:
                    frame_landmarks = landmarks[:, :2].copy()
                    frame_landmarks[:, 0] = crop_x1 + (frame_landmarks[:, 0] - entry.x) * scale_x
                    frame_landmarks[:, 1] = crop_y1 + (frame_landmarks[:, 1] - entry.y) * scale_y

                mapped_faces.append(
                    {
                        "frame_index": entry.frame_index,
                        "human_index": entry.human_index,
                        "track_id": entry.track_id,
                        "bbox": [float(value) for value in frame_bbox],
                        "landmarks": frame_landmarks,
                        "confidence": float(face.get("confidence", 0.0) or 0.0),
                    }
                )
        return mapped_faces

    def _quality_batch(
        self,
        items: Sequence[tuple[np.ndarray, dict[str, Any]]],
    ) -> list[tuple[bool, float, str, np.ndarray | None, dict[str, Any]]]:
        if not items:
            return []
        if self.settings.use_gpu_quality:
            gpu_values = self._quality_batch_gpu(items)
            if gpu_values is not None:
                self._last_quality_backend = "torch-cuda"
                return gpu_values
        self._last_quality_backend = "cpu"
        return [self._quality(frame, face) for frame, face in items]

    def _quality_batch_gpu(
        self,
        items: Sequence[tuple[np.ndarray, dict[str, Any]]],
    ) -> list[tuple[bool, float, str, np.ndarray | None, dict[str, Any]]] | None:
        """GPU-assisted batched face-quality gate.

        The expensive per-face blur and scoring math is batched on CUDA when
        torch is available. Geometric values are approximated from landmarks to
        avoid cv2.solvePnP in the real-time path. Face alignment still uses the
        existing OpenCV affine transform for compatibility with the current
        ArcFace preprocessing.
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            return None
        if self.settings.device.lower() == "cpu" or not torch.cuda.is_available():
            return None

        device = torch.device(
            f"cuda:{int(self.settings.device)}"
            if str(self.settings.device).isdigit()
            else "cuda:0"
        )
        prepared: list[dict[str, Any]] = []
        tensors: list[np.ndarray] = []
        for frame, face in items:
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = [int(round(value)) for value in face["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            crop = frame[y1:y2, x1:x2]
            landmarks = np.asarray(face.get("landmarks", []), dtype=np.float32)
            base_metrics: dict[str, Any] = {
                "blur": 0.0,
                "eye_distance": 0.0,
                "yaw": None,
                "pitch": None,
                "roll": None,
                "landmark_count": int(len(landmarks)) if landmarks.ndim == 2 else 0,
                "size_score": 0.0,
                "blur_score": 0.0,
                "pose_score": 0.0,
                "face_width": int(max(0, x2 - x1)),
                "face_height": int(max(0, y2 - y1)),
                "quality_frame_width": int(width),
                "quality_frame_height": int(height),
            }
            if crop.size == 0:
                prepared.append(
                    {
                        "empty": True,
                        "frame": frame,
                        "face": face,
                        "crop": None,
                        "landmarks": landmarks,
                        "metrics": base_metrics,
                    }
                )
                continue
            tensors.append(cv2.resize(crop, (112, 112)))
            prepared.append(
                {
                    "empty": False,
                    "frame": frame,
                    "face": face,
                    "crop": crop,
                    "landmarks": landmarks,
                    "metrics": base_metrics,
                }
            )

        if not tensors:
            return [(False, 0.0, "empty_crop", None, item["metrics"]) for item in prepared]

        batch = torch.from_numpy(np.stack(tensors)).to(device=device, dtype=torch.float32)
        # Input is BGR because frames come from OpenCV.
        gray = (
            0.114 * batch[..., 0]
            + 0.587 * batch[..., 1]
            + 0.299 * batch[..., 2]
        ).unsqueeze(1)
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            device=device,
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        laplacian = F.conv2d(gray, kernel, padding=1)
        blur_values = laplacian.flatten(1).var(dim=1, unbiased=False).detach().cpu().numpy()

        output: list[tuple[bool, float, str, np.ndarray | None, dict[str, Any]]] = []
        blur_index = 0
        for item in prepared:
            metrics = dict(item["metrics"])
            if item["empty"]:
                output.append((False, 0.0, "empty_crop", None, metrics))
                continue

            frame = item["frame"]
            face = item["face"]
            crop = item["crop"]
            assert crop is not None
            landmarks = np.asarray(item["landmarks"], dtype=np.float32)
            face_height, face_width = crop.shape[:2]
            blur = float(blur_values[blur_index])
            blur_index += 1

            width_score = min(
                1.0,
                face_width / max(float(self.settings.min_face_width * 2), 1.0),
            )
            height_score = min(
                1.0,
                face_height / max(float(self.settings.min_face_height * 2), 1.0),
            )
            size_score = min(width_score, height_score)
            blur_score = min(1.0, blur / max(self.settings.blur_threshold * 4.0, 1.0))

            eye_distance = 0.0
            yaw = pitch = roll = None
            pose_score = 0.0
            if landmarks.ndim == 2 and len(landmarks) >= 2:
                left_eye, right_eye = landmarks[0], landmarks[1]
                eye_delta = right_eye - left_eye
                eye_distance = float(np.linalg.norm(eye_delta))
                roll = float(np.degrees(np.arctan2(float(eye_delta[1]), float(eye_delta[0]))))
                if len(landmarks) >= 5:
                    nose = landmarks[2]
                    mouth_center = (landmarks[3] + landmarks[4]) / 2.0
                    eye_center = (left_eye + right_eye) / 2.0
                    vertical = max(float(np.linalg.norm(mouth_center - eye_center)), 1e-6)
                    yaw = float(((nose[0] - eye_center[0]) / max(eye_distance, 1e-6)) * 60.0)
                    pitch = float((((nose[1] - eye_center[1]) / vertical) - 0.45) * 80.0)
                    pose_parts = []
                    for value, maximum in (
                        (yaw, self.settings.max_abs_yaw),
                        (pitch, self.settings.max_abs_pitch),
                        (roll, self.settings.max_abs_roll),
                    ):
                        pose_parts.append(max(0.0, 1.0 - abs(value) / max(float(maximum), 1e-6)))
                    pose_score = sum(pose_parts) / len(pose_parts)

            eye_score = min(
                1.0,
                eye_distance / max(float(self.settings.min_eye_distance * 2), 1.0),
            )
            confidence = float(face.get("confidence", 0.0) or 0.0)
            quality = (
                0.30 * blur_score
                + 0.20 * size_score
                + 0.15 * eye_score
                + 0.25 * pose_score
                + 0.10 * confidence
            )
            quality = max(0.0, min(1.0, quality))
            metrics.update(
                {
                    "blur": round(blur, 4),
                    "eye_distance": round(eye_distance, 4),
                    "yaw": None if yaw is None else round(yaw, 4),
                    "pitch": None if pitch is None else round(pitch, 4),
                    "roll": None if roll is None else round(roll, 4),
                    "landmark_count": int(len(landmarks)) if landmarks.ndim == 2 else 0,
                    "size_score": round(size_score, 6),
                    "blur_score": round(blur_score, 6),
                    "pose_score": round(pose_score, 6),
                    "face_width": int(face_width),
                    "face_height": int(face_height),
                }
            )

            reason = "ok"
            if (
                face_width < self.settings.min_face_width
                or face_height < self.settings.min_face_height
            ):
                reason = "face_too_small"
            elif self.settings.require_landmarks and len(landmarks) < 5:
                reason = "missing_landmarks"
            elif blur < self.settings.blur_threshold:
                reason = "blurry"
            elif eye_distance < self.settings.min_eye_distance:
                reason = "eyes_too_close"
            elif self.settings.require_landmarks and None in (yaw, pitch, roll):
                reason = "pose_unavailable"
            elif yaw is not None and abs(yaw) > self.settings.max_abs_yaw:
                reason = "yaw_out_of_range"
            elif pitch is not None and abs(pitch) > self.settings.max_abs_pitch:
                reason = "pitch_out_of_range"
            elif roll is not None and abs(roll) > self.settings.max_abs_roll:
                reason = "roll_out_of_range"
            elif quality < self.settings.quality_threshold:
                reason = "quality_below_threshold"

            valid = reason == "ok"
            aligned: np.ndarray | None = crop
            if valid:
                corrected = self._align_face(frame, face["bbox"], landmarks)
                if corrected is None:
                    valid = False
                    reason = "alignment_failed"
                    aligned = None
                else:
                    aligned = corrected
            output.append((valid, quality, reason, aligned, metrics))

        return output

    def _tracker(self, source_id: str) -> SourceFaceTracker:
        tracker = self._trackers.get(source_id)
        if tracker is None:
            tracker = SourceFaceTracker(
                high_threshold=self.settings.tracker_high_threshold,
                low_threshold=self.settings.tracker_low_threshold,
                new_threshold=self.settings.tracker_new_threshold,
                match_threshold=self.settings.tracker_match_threshold,
                max_missed=self.settings.tracker_max_missed,
                history_size=self.settings.history_size,
                stable_min_hits=self.settings.stable_min_hits,
                backend=(
                    self._tracker_backend_factory()
                    if self._tracker_backend_factory is not None
                    else None
                ),
            )
            self._trackers[source_id] = tracker
        return tracker

    def process_batch(self, packets: Sequence[FramePacket]) -> list[TaskResult]:
        with self._load_lock:
            return self._process_batch(packets)

    def _process_batch(self, packets: Sequence[FramePacket]) -> list[TaskResult]:
        if not packets:
            return []
        started = time.perf_counter()
        try:
            self._ensure_dependencies()
            assert self._human_detector is not None
            assert self._face_detector is not None
            assert self._embedder is not None
            assert self._vector_store is not None

            frames = [packet.frame for packet in packets]
            source_frames = [packet.source_frame for packet in packets]
            thresholds = [self._source_thresholds(packet.source_id) for packet in packets]

            # 1) Human detection still runs on every input frame.
            detection_started = time.perf_counter()
            human_results = self._predict_yolo(
                self._human_detector,
                frames,
                model_path=self.settings.human_model_path,
                imgsz=self.settings.human_imgsz,
                confidence=min(item[0] for item in thresholds),
                fixed_batch=self.settings.human_engine_fixed_batch,
            )
            self._last_human_detection_ms = (time.perf_counter() - detection_started) * 1000.0

            # 2) Track humans before face recognition. Faces detected inside a
            # human ROI inherit that human track_id directly.
            frame_payloads: list[dict[str, Any]] = []
            for frame_index, (packet, source_frame, human_result) in enumerate(
                zip(packets, source_frames, human_results)
            ):
                humans, human_rejections = self._human_boxes(
                    human_result,
                    thresholds[frame_index][0],
                )
                candidate_count = len(humans) + len(human_rejections)
                self._human_candidates += candidate_count
                self._filtered_humans += len(human_rejections)
                for human in humans:
                    human.pop("_result_index", None)
                    human["source_bbox"] = self._scale_bbox(
                        human["bbox"], packet.frame.shape, source_frame.shape
                    )

                tracker = self._tracker(packet.source_id)
                track_ids = tracker.update(
                    [human["bbox"] for human in humans],
                    [human["confidence"] for human in humans],
                )
                tracked_humans = [
                    {
                        **human,
                        "track_id": track_ids[index],
                        "person": "Unknown",
                        "recognition_score": 0.0,
                        "ref_img_id": None,
                        "identity_stable": False,
                        "face_visible": False,
                    }
                    for index, human in enumerate(humans)
                    if index < len(track_ids)
                ]
                disappeared_humans = [
                    {
                        "track_id": state.track_id,
                        "bbox": list(state.bbox),
                        "source_bbox": self._scale_bbox(
                            state.bbox, packet.frame.shape, source_frame.shape
                        ),
                        "person": state.stable_person,
                        "recognition_score": round(state.stable_score, 6),
                        "ref_img_id": state.stable_ref_img_id,
                        "identity_stable": state.stable_person != "Unknown",
                        "best_face_quality": round(state.best_face_quality, 6),
                        "confidence": 0.0,
                    }
                    for state in tracker.consume_disappeared()
                ]
                frame_payloads.append(
                    {
                        "humans": tracked_humans,
                        "faces": [],
                        "disappeared_humans": disappeared_humans,
                        "human_filter": {
                            "enabled": self.settings.human_pose_enabled,
                            "candidate_count": candidate_count,
                            "accepted_count": len(humans),
                            "rejected_count": len(human_rejections),
                            "rejections": human_rejections,
                        },
                    }
                )

            # 3) Build human crops, concatenate them into batched mosaic images,
            # and run face detection only on those human ROI mosaics.
            face_roi_started = time.perf_counter()
            human_rois = self._build_human_rois(packets, frame_payloads)
            mosaics = self._build_human_roi_mosaics(human_rois)
            face_results = self._predict_yolo(
                self._face_detector,
                [mosaic.image for mosaic in mosaics],
                model_path=self.settings.face_model_path,
                imgsz=self.settings.face_imgsz,
                confidence=min(item[1] for item in thresholds),
                fixed_batch=self.settings.face_engine_fixed_batch,
            )
            detected_faces = self._faces_from_human_roi_mosaics(
                mosaics,
                face_results,
                [item[1] for item in thresholds],
            )
            self._last_face_roi_detection_ms = (time.perf_counter() - face_roi_started) * 1000.0
            self._last_detection_ms = self._last_human_detection_ms + self._last_face_roi_detection_ms

            # 4) Prepare detected faces and run batched quality checks. Faces are
            # already guaranteed to come from inside a detected human ROI.
            quality_items: list[tuple[np.ndarray, dict[str, Any]]] = []
            quality_locations: list[tuple[int, int]] = []
            for detected in detected_faces:
                frame_index = int(detected["frame_index"])
                human_index = int(detected["human_index"])
                packet = packets[frame_index]
                source_frame = packet.source_frame
                source_face = self._source_face(detected, packet.frame, source_frame)
                prepared = {
                    "bbox": [float(value) for value in detected["bbox"]],
                    "source_bbox": source_face["bbox"],
                    "landmarks": self._landmarks_to_list(detected.get("landmarks")),
                    "source_landmarks": self._landmarks_to_list(source_face.get("landmarks")),
                    "detection_confidence": float(detected.get("confidence", 0.0) or 0.0),
                    "track_id": detected.get("track_id"),
                    "human_index": human_index,
                    "quality": 0.0,
                    "quality_score": 0.0,
                    "quality_metrics": {},
                    "quality_valid": False,
                    "quality_reason": "not_checked",
                    "person": "Unknown",
                    "recognition_score": 0.0,
                    "ref_img_id": None,
                    "stable": False,
                }
                face_index = len(frame_payloads[frame_index]["faces"])
                frame_payloads[frame_index]["faces"].append(prepared)
                quality_locations.append((frame_index, face_index))
                quality_items.append((source_frame, source_face))

            quality_started = time.perf_counter()
            quality_results = self._quality_batch(quality_items)
            self._last_quality_ms = (time.perf_counter() - quality_started) * 1000.0

            face_crops: list[np.ndarray] = []
            face_locations: list[tuple[int, int]] = []
            for (frame_index, face_index), quality_result in zip(
                quality_locations, quality_results
            ):
                valid, quality, reason, crop, quality_metrics = quality_result
                face = frame_payloads[frame_index]["faces"][face_index]
                face.update(
                    {
                        "quality": round(float(quality), 6),
                        "quality_score": round(float(quality), 6),
                        "quality_metrics": quality_metrics,
                        "quality_valid": valid,
                        "quality_reason": reason,
                    }
                )
                if valid:
                    self._tracker(packets[frame_index].source_id).record_face_quality(
                        face.get("track_id"), quality
                    )
                if valid and crop is not None and face.get("track_id") is not None:
                    face_locations.append((frame_index, face_index))
                    face_crops.append(crop)

            # 5) Recognition is batched: ArcFace embeddings are generated for all
            # valid faces, then vector search is also performed as a batch.
            if face_crops:
                embedding_started = time.perf_counter()
                embeddings = self._embedder.embed(face_crops)
                self._last_embedding_ms = (time.perf_counter() - embedding_started) * 1000.0
                if len(embeddings) != len(face_crops):
                    raise RuntimeError("ArcFace embedding count does not match valid faces")

                search_started = time.perf_counter()
                matches = self._vector_store.search_batch(
                    embeddings,
                    min(item[2] for item in thresholds),
                )
                self._last_search_ms = (time.perf_counter() - search_started) * 1000.0
                if len(matches) != len(face_crops):
                    raise RuntimeError("Qdrant match count does not match embeddings")

                for (frame_index, face_index), match in zip(face_locations, matches):
                    face = frame_payloads[frame_index]["faces"][face_index]
                    track_id = face.get("track_id")
                    quality_value = float(face.get("quality_score", 0.0) or 0.0)
                    quality_weight = float(self.settings.recognition_quality_weight)
                    quality_factor = (1.0 - quality_weight) + quality_weight * max(
                        0.0, min(1.0, quality_value)
                    )
                    weighted_match = FaceMatch(
                        person=match.person,
                        score=float(match.score) * quality_factor,
                        ref_img_id=match.ref_img_id,
                    )
                    adjusted_score = weighted_match.score
                    admitted = adjusted_score >= thresholds[frame_index][2]
                    admitted_match = (
                        weighted_match
                        if admitted
                        else FaceMatch()
                    )
                    stable = self._tracker(packets[frame_index].source_id).observe(
                        int(track_id), admitted_match
                    )
                    face.update(
                        {
                            "person": stable.person,
                            "recognition_score": round(stable.score, 6),
                            "ref_img_id": stable.ref_img_id,
                            "stable": stable.person != "Unknown",
                            "raw_person": match.person if admitted else "Unknown",
                            "raw_recognition_score": round(match.score, 6),
                            "recognition_admitted": admitted,
                            "quality_weight": round(quality_weight, 6),
                            "quality_adjusted_score": round(adjusted_score, 6),
                        }
                    )
            else:
                self._last_embedding_ms = 0.0
                self._last_search_ms = 0.0

            # 6) Attach the stable identity to the tracked human. The human keeps
            # the identity even when the face is not visible in a later frame.
            for frame_index, payload in enumerate(frame_payloads):
                tracker = self._tracker(packets[frame_index].source_id)
                faces_by_track = {
                    face["track_id"]
                    for face in payload["faces"]
                    if face.get("track_id") is not None
                    and face.get("quality_valid") is True
                }
                for human in payload["humans"]:
                    identity = tracker.identity(human.get("track_id"))
                    human.update(
                        {
                            "person": identity.person,
                            "recognition_score": round(identity.score, 6),
                            "ref_img_id": identity.ref_img_id,
                            "identity_stable": identity.person != "Unknown",
                            "face_visible": human.get("track_id") in faces_by_track,
                            "best_face_quality": tracker.best_quality(
                                human.get("track_id")
                            ),
                        }
                    )

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            output: list[TaskResult] = []
            for packet, payload in zip(packets, frame_payloads):
                faces = payload["faces"]
                recognized = sum(1 for face in faces if face["person"] != "Unknown")
                recognized_humans = sum(
                    1
                    for human in payload["humans"]
                    if human["person"] != "Unknown"
                )
                data = {
                    **payload,
                    "inference_frame_size": {
                        "width": int(packet.frame.shape[1]),
                        "height": int(packet.frame.shape[0]),
                    },
                    "source_frame_size": {
                        "width": int(packet.source_frame.shape[1]),
                        "height": int(packet.source_frame.shape[0]),
                    },
                    "tracking_session_id": self._tracking_session_id,
                    "human_count": len(payload["humans"]),
                    "disappeared_human_count": len(payload["disappeared_humans"]),
                    "face_count": len(faces),
                    "recognized_count": recognized,
                    "recognized_human_count": recognized_humans,
                    "timings_ms": {
                        "human_detection_batch": round(self._last_human_detection_ms, 3),
                        "face_roi_detection_batch": round(self._last_face_roi_detection_ms, 3),
                        "detection_batch": round(self._last_detection_ms, 3),
                        "quality_batch": round(self._last_quality_ms, 3),
                        "quality_backend": self._last_quality_backend,
                        "embedding_batch": round(self._last_embedding_ms, 3),
                        "qdrant_batch": round(self._last_search_ms, 3),
                    },
                }
                output.append(
                    TaskResult.success(
                        task=self.task,
                        packet=packet,
                        processing_ms=elapsed_ms,
                        data=data,
                    )
                )
                self._detected_faces += len(faces)
                self._recognized_faces += recognized
            self._processed_batches += 1
            self._processed_frames += len(packets)
            self._last_batch_ms = elapsed_ms
            self._last_error = None
            return output
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise

    def enroll(
        self,
        image: np.ndarray,
        *,
        person: str,
        ref_img_id: str | int | None = None,
    ) -> dict[str, Any]:
        person = person.strip()
        if not person:
            raise ValueError("person cannot be blank")
        self._ensure_dependencies()
        assert self._face_detector is not None
        assert self._embedder is not None
        assert self._vector_store is not None
        results = self._predict_yolo(
            self._face_detector,
            [image],
            model_path=self.settings.face_model_path,
            imgsz=self.settings.face_imgsz,
            confidence=self.settings.face_confidence,
            fixed_batch=self.settings.face_engine_fixed_batch,
        )
        faces = self._faces(results[0])
        valid: list[tuple[dict[str, Any], np.ndarray, float, dict[str, Any]]] = []
        for face in faces:
            accepted, quality, _, crop, metrics = self._quality(image, face)
            if accepted and crop is not None:
                valid.append((face, crop, quality, metrics))
        if len(valid) != 1:
            raise ValueError(
                f"Enrollment requires exactly one valid face; found {len(valid)}"
            )
        face, crop, quality, metrics = valid[0]
        embedding = self._embedder.embed([crop])[0]
        point_id = self._vector_store.enroll(person, embedding, ref_img_id)
        return {
            "point_id": point_id,
            "person": person,
            "ref_img_id": ref_img_id,
            "bbox": face["bbox"],
            "quality": round(float(quality), 6),
            "quality_metrics": metrics,
        }

    def count_faces(self, image: np.ndarray) -> int:
        """Count raw detected faces in an image (no quality gating)."""
        self._ensure_dependencies()
        assert self._face_detector is not None
        results = self._predict_yolo(
            self._face_detector,
            [image],
            model_path=self.settings.face_model_path,
            imgsz=self.settings.face_imgsz,
            confidence=self.settings.face_confidence,
            fixed_batch=self.settings.face_engine_fixed_batch,
        )
        faces = self._faces(results[0])
        return len(faces)

    def get_aligned_face(self, image: np.ndarray) -> tuple[bool, np.ndarray | None]:
        """Return aligned face crop (112x112) for a single-face image.

        Returns (success, aligned_face).
        """
        self._ensure_dependencies()
        assert self._face_detector is not None
        results = self._predict_yolo(
            self._face_detector,
            [image],
            model_path=self.settings.face_model_path,
            imgsz=self.settings.face_imgsz,
            confidence=self.settings.face_confidence,
            fixed_batch=self.settings.face_engine_fixed_batch,
        )
        faces = self._faces(results[0])
        if not faces:
            return False, None
        face = faces[0]
        valid, quality, _, crop, _ = self._quality(image, face)
        if valid and crop is not None:
            return True, crop
        return False, None

    def identities(self, limit: int = 1000) -> list[dict[str, Any]]:
        self._ensure_dependencies()
        assert self._vector_store is not None
        return self._vector_store.identities(limit=limit)

    def delete_person(self, person: str) -> int:
        self._ensure_dependencies()
        assert self._vector_store is not None
        return self._vector_store.delete_person(person.strip())

    def delete_points(self, point_ids: list[str]) -> int:
        self._ensure_dependencies()
        assert self._vector_store is not None
        return self._vector_store.delete_points(point_ids)

    def quality_settings(self) -> dict[str, Any]:
        return {
            "quality_threshold": self.settings.quality_threshold,
            "blur_threshold": self.settings.blur_threshold,
            "min_face_width": self.settings.min_face_width,
            "min_face_height": self.settings.min_face_height,
            "min_eye_distance": self.settings.min_eye_distance,
            "max_abs_yaw": self.settings.max_abs_yaw,
            "max_abs_pitch": self.settings.max_abs_pitch,
            "max_abs_roll": self.settings.max_abs_roll,
            "require_landmarks": self.settings.require_landmarks,
            "human_pose_enabled": self.settings.human_pose_enabled,
            "human_pose_min_keypoints": self.settings.human_pose_min_keypoints,
            "human_pose_keypoint_confidence": self.settings.human_pose_keypoint_confidence,
            "recognition_quality_weight": self.settings.recognition_quality_weight,
        }

    def update_quality_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        allowed = set(self.quality_settings())
        unexpected = set(values) - allowed
        if unexpected:
            raise ValueError(f"Unsupported face quality settings: {sorted(unexpected)}")
        with self._load_lock:
            self.settings = replace(self.settings, **values)
        return self.quality_settings()

    def update_runtime_thresholds(
        self,
        *,
        human_confidence: float,
        face_confidence: float,
        recognition_threshold: float,
    ) -> None:
        with self._load_lock:
            self.settings = replace(
                self.settings,
                human_confidence=float(human_confidence),
                face_confidence=float(face_confidence),
                recognition_threshold=float(recognition_threshold),
            )

    def status(self) -> dict[str, Any]:
        vector_status: dict[str, Any] | None = None
        if self._vector_store is not None:
            try:
                vector_status = self._vector_store.status()
            except Exception as exc:
                vector_status = {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "task": self.task.value,
            "ready": self._last_error is None
            and all(
                value is not None
                for value in (
                    self._human_detector,
                    self._face_detector,
                    self._embedder,
                    self._vector_store,
                )
            ),
            "human_model": str(self.settings.human_model_path),
            "human_model_exists": self.settings.human_model_path.is_file(),
            "face_model": str(self.settings.face_model_path),
            "face_model_exists": self.settings.face_model_path.is_file(),
            "embedding_model": str(self.settings.embedding_model_path),
            "embedding_model_exists": self.settings.embedding_model_path.is_file(),
            "models_loaded": all(
                value is not None
                for value in (
                    self._human_detector,
                    self._face_detector,
                    self._embedder,
                )
            ),
            "embedding_backend": getattr(self._embedder, "backend", None),
            "qdrant": vector_status,
            "vector_store_warning": self._vector_store_warning,
            "tracker": "ByteTrack",
            "quality_gate": self.quality_settings(),
            "tracking_session_id": self._tracking_session_id,
            "active_sources": len(self._trackers),
            "active_humans": sum(
                len(tracker.tracks) for tracker in self._trackers.values()
            ),
            "processed_batches": self._processed_batches,
            "processed_frames": self._processed_frames,
            "detected_faces": self._detected_faces,
            "recognized_faces": self._recognized_faces,
            "human_filter": {
                "enabled": self.settings.human_pose_enabled,
                "candidates": self._human_candidates,
                "rejected": self._filtered_humans,
            },
            "last_batch_ms": round(self._last_batch_ms, 3),
            "last_detection_ms": round(self._last_detection_ms, 3),
            "last_human_detection_ms": round(self._last_human_detection_ms, 3),
            "last_face_roi_detection_ms": round(self._last_face_roi_detection_ms, 3),
            "last_quality_ms": round(self._last_quality_ms, 3),
            "last_quality_backend": self._last_quality_backend,
            "last_embedding_ms": round(self._last_embedding_ms, 3),
            "last_search_ms": round(self._last_search_ms, 3),
            "last_error": self._last_error,
        }

    def active_tracks(self) -> list[dict[str, Any]]:
        return [
            {"camera": source_id, **track}
            for source_id, tracker in self._trackers.items()
            for track in tracker.snapshot()
        ]

    def close(self) -> None:
        if self._embedder is not None:
            self._embedder.close()
        if self._vector_store is not None:
            self._vector_store.close()
