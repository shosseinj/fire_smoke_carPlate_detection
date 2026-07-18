from __future__ import annotations

import threading
import time
import uuid
import sqlite3
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import cv2
import numpy as np

from app.core.types import FramePacket, TaskName, TaskResult
from app.processors.base import BatchProcessor
from app.processors.ultralytics_loader import load_yolo_class, serialized_model_load


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
    min_face_size: int = 24
    blur_threshold: float = 20.0
    min_eye_distance: float = 8.0
    tracker_iou_threshold: float = 0.25
    tracker_max_missed: int = 30
    history_size: int = 30
    stable_min_hits: int = 3
    embedding_batch_size: int = 64
    vector_size: int = 512
    qdrant_collection: str = "faces"
    qdrant_url: str | None = None
    qdrant_path: Path | None = None
    qdrant_api_key: str | None = None


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


class SourceFaceTracker:
    """Small per-source IoU tracker with identity vote history."""

    def __init__(
        self,
        *,
        iou_threshold: float,
        max_missed: int,
        history_size: int,
        stable_min_hits: int,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_missed = max(1, int(max_missed))
        self.history_size = max(1, int(history_size))
        self.stable_min_hits = max(1, int(stable_min_hits))
        self._next_id = 1
        self.tracks: dict[int, TrackState] = {}

    def update(self, boxes: Sequence[Sequence[float]]) -> list[int]:
        for track in self.tracks.values():
            track.missed += 1
        assigned: list[int | None] = [None] * len(boxes)
        candidates: list[tuple[float, int, int]] = []
        for box_index, box in enumerate(boxes):
            for track_id, track in self.tracks.items():
                score = _iou(box, track.bbox)
                if score >= self.iou_threshold:
                    candidates.append((score, box_index, track_id))
        used_boxes: set[int] = set()
        used_tracks: set[int] = set()
        for _, box_index, track_id in sorted(candidates, reverse=True):
            if box_index in used_boxes or track_id in used_tracks:
                continue
            track = self.tracks[track_id]
            track.bbox = [float(value) for value in boxes[box_index][:4]]
            track.missed = 0
            assigned[box_index] = track_id
            used_boxes.add(box_index)
            used_tracks.add(track_id)
        for box_index, box in enumerate(boxes):
            if assigned[box_index] is not None:
                continue
            track_id = self._next_id
            self._next_id += 1
            self.tracks[track_id] = TrackState(
                track_id=track_id,
                bbox=[float(value) for value in box[:4]],
                names=deque(maxlen=self.history_size),
                scores=deque(maxlen=self.history_size),
                ref_img_ids=deque(maxlen=self.history_size),
            )
            assigned[box_index] = track_id
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missed > self.max_missed
        ]
        for track_id in expired:
            self.tracks.pop(track_id, None)
        return [int(track_id) for track_id in assigned if track_id is not None]

    def track_for_face(self, face_box: Sequence[float]) -> int | None:
        center_x = (float(face_box[0]) + float(face_box[2])) / 2.0
        center_y = (float(face_box[1]) + float(face_box[3])) / 2.0
        candidates: list[tuple[float, int]] = []
        for track_id, track in self.tracks.items():
            x1, y1, x2, y2 = track.bbox
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
        known_counts = Counter(name for name in track.names if name != "Unknown")
        if not known_counts:
            return FaceMatch()
        person, count = known_counts.most_common(1)[0]
        if count < self.stable_min_hits:
            return FaceMatch()
        indexes = [index for index, name in enumerate(track.names) if name == person]
        best_index = max(indexes, key=lambda index: track.scores[index])
        return FaceMatch(
            person=person,
            score=float(track.scores[best_index]),
            ref_img_id=track.ref_img_ids[best_index],
        )


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
        self.device = torch.device(f"cuda:{int(device)}" if device.isdigit() else "cuda:0")
        self.lock = threading.RLock()
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(model_path.read_bytes())
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
    def __init__(self, settings: FaceRecognitionSettings) -> None:
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise RuntimeError("qdrant-client is required for face recognition") from exc
        self.models = models
        self.collection = settings.qdrant_collection
        self.vector_size = settings.vector_size
        self.lock = threading.RLock()
        if settings.qdrant_url:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
            self.mode = "remote"
        else:
            path = (settings.qdrant_path or Path("data/qdrant")).resolve()
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
            self.mode = "local"
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
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

    def status(self) -> dict[str, Any]:
        with self.lock:
            count = int(self.client.count(collection_name=self.collection).count)
        return {"mode": self.mode, "collection": self.collection, "points": count}

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


class SqliteFaceStore:
    """Package-free persistent cosine store used until Qdrant is available."""

    def __init__(self, path: Path, vector_size: int) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_size = int(vector_size)
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, check_same_thread=False, timeout=10.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id TEXT PRIMARY KEY,
                person TEXT NOT NULL,
                ref_img_id TEXT,
                embedding BLOB NOT NULL,
                dimension INTEGER NOT NULL
            )
            """
        )
        self.connection.commit()

    def search_batch(self, embeddings: np.ndarray, threshold: float) -> list[FaceMatch]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT person, ref_img_id, embedding, dimension FROM face_embeddings"
            ).fetchall()
        if not rows:
            return [FaceMatch() for _ in embeddings]
        vectors = []
        metadata = []
        for person, ref_img_id, raw, dimension in rows:
            vector = np.frombuffer(raw, dtype=np.float32, count=int(dimension))
            if len(vector) != self.vector_size:
                continue
            vectors.append(vector)
            metadata.append((str(person), ref_img_id))
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
        with self.lock:
            self.connection.execute(
                "INSERT INTO face_embeddings VALUES (?, ?, ?, ?, ?)",
                (point_id, person, None if ref_img_id is None else str(ref_img_id), vector.tobytes(), len(vector)),
            )
            self.connection.commit()
        return point_id

    def identities(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.connection.execute(
                """
                SELECT person, ref_img_id, COUNT(*)
                FROM face_embeddings
                GROUP BY person, ref_img_id
                ORDER BY person, ref_img_id
                LIMIT ?
                """,
                (max(1, min(int(limit), 10000)),),
            ).fetchall()
        return [
            {"person": person, "ref_img_id": ref_img_id, "embeddings": int(count)}
            for person, ref_img_id, count in rows
        ]

    def delete_person(self, person: str) -> int:
        with self.lock:
            cursor = self.connection.execute(
                "DELETE FROM face_embeddings WHERE person = ?", (person,)
            )
            self.connection.commit()
            return int(cursor.rowcount)

    def status(self) -> dict[str, Any]:
        with self.lock:
            count = int(
                self.connection.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0]
            )
        return {
            "mode": "sqlite-fallback",
            "path": str(self.path),
            "points": count,
            "reason": "qdrant-client is not installed",
        }

    def close(self) -> None:
        with self.lock:
            self.connection.close()


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
    ) -> None:
        self.settings = settings
        self._human_detector = human_detector
        self._face_detector = face_detector
        self._embedder = embedder
        self._vector_store = vector_store
        self._load_lock = threading.RLock()
        self._trackers: dict[str, SourceFaceTracker] = {}
        self._processed_batches = 0
        self._processed_frames = 0
        self._detected_faces = 0
        self._recognized_faces = 0
        self._last_batch_ms = 0.0
        self._last_detection_ms = 0.0
        self._last_embedding_ms = 0.0
        self._last_search_ms = 0.0
        self._last_error: str | None = None

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
                try:
                    self._vector_store = QdrantFaceStore(self.settings)
                except RuntimeError as exc:
                    if "qdrant-client is required" not in str(exc):
                        raise
                    qdrant_path = self.settings.qdrant_path or Path("data/qdrant")
                    self._vector_store = SqliteFaceStore(
                        qdrant_path.parent / "face_embeddings.sqlite3",
                        self.settings.vector_size,
                    )

    def preload(self) -> None:
        try:
            self._ensure_dependencies()
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
        real_count = len(frames)
        source = list(frames)
        batch = real_count
        if model_path.suffix.lower() == ".engine" and fixed_batch:
            if real_count > fixed_batch:
                raise ValueError(
                    f"{model_path.name} accepts at most {fixed_batch} frames"
                )
            batch = fixed_batch
            while len(source) < fixed_batch:
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
        if len(results) < real_count:
            raise RuntimeError("Face model returned fewer results than input frames")
        return results[:real_count]

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

    def _faces(self, result: Any) -> list[dict[str, Any]]:
        faces = self._boxes(result, self.settings.face_confidence)
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

    def _quality(
        self,
        frame: np.ndarray,
        face: dict[str, Any],
    ) -> tuple[bool, float, str, np.ndarray | None]:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in face["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False, 0.0, "empty_crop", None
        if min(crop.shape[:2]) < self.settings.min_face_size:
            return False, 0.0, "face_too_small", crop
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur < self.settings.blur_threshold:
            return False, blur, "blurry", crop
        landmarks = np.asarray(face["landmarks"], dtype=np.float32)
        if len(landmarks) >= 2:
            eye_distance = float(np.linalg.norm(landmarks[0] - landmarks[1]))
            if eye_distance < self.settings.min_eye_distance:
                return False, eye_distance, "eyes_too_close", crop
        quality = min(1.0, blur / max(self.settings.blur_threshold * 4.0, 1.0))
        return True, quality, "ok", self._align_face(frame, face["bbox"], landmarks)

    @staticmethod
    def _align_face(
        frame: np.ndarray,
        bbox: Sequence[float],
        landmarks: np.ndarray,
    ) -> np.ndarray:
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
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in bbox[:4]]
        crop = frame[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]
        return cv2.resize(crop, (112, 112))

    def _tracker(self, source_id: str) -> SourceFaceTracker:
        tracker = self._trackers.get(source_id)
        if tracker is None:
            tracker = SourceFaceTracker(
                iou_threshold=self.settings.tracker_iou_threshold,
                max_missed=self.settings.tracker_max_missed,
                history_size=self.settings.history_size,
                stable_min_hits=self.settings.stable_min_hits,
            )
            self._trackers[source_id] = tracker
        return tracker

    def process_batch(self, packets: Sequence[FramePacket]) -> list[TaskResult]:
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
            detection_started = time.perf_counter()
            human_results = self._predict_yolo(
                self._human_detector,
                frames,
                model_path=self.settings.human_model_path,
                imgsz=self.settings.human_imgsz,
                confidence=self.settings.human_confidence,
                fixed_batch=self.settings.human_engine_fixed_batch,
            )
            face_results = self._predict_yolo(
                self._face_detector,
                frames,
                model_path=self.settings.face_model_path,
                imgsz=self.settings.face_imgsz,
                confidence=self.settings.face_confidence,
                fixed_batch=self.settings.face_engine_fixed_batch,
            )
            self._last_detection_ms = (time.perf_counter() - detection_started) * 1000.0

            frame_payloads: list[dict[str, Any]] = []
            face_crops: list[np.ndarray] = []
            face_locations: list[tuple[int, int]] = []
            for frame_index, (packet, human_result, face_result) in enumerate(
                zip(packets, human_results, face_results)
            ):
                humans = self._boxes(human_result, self.settings.human_confidence)
                for human in humans:
                    human.pop("_result_index", None)
                faces = self._faces(face_result)
                tracking_boxes = [human["bbox"] for human in humans]
                for face in faces:
                    center_x = (face["bbox"][0] + face["bbox"][2]) / 2.0
                    center_y = (face["bbox"][1] + face["bbox"][3]) / 2.0
                    if not any(
                        box[0] <= center_x <= box[2] and box[1] <= center_y <= box[3]
                        for box in tracking_boxes
                    ):
                        tracking_boxes.append(face["bbox"])
                tracker = self._tracker(packet.source_id)
                track_ids = tracker.update(tracking_boxes)
                tracked_humans = [
                    {
                        **human,
                        "track_id": track_ids[index],
                    }
                    for index, human in enumerate(humans)
                    if index < len(track_ids)
                ]
                prepared_faces: list[dict[str, Any]] = []
                for face_index, face in enumerate(faces):
                    track_id = tracker.track_for_face(face["bbox"])
                    valid, quality, reason, crop = self._quality(packet.frame, face)
                    prepared = {
                        "bbox": face["bbox"],
                        "detection_confidence": face["confidence"],
                        "track_id": track_id,
                        "quality": round(float(quality), 4),
                        "quality_valid": valid,
                        "quality_reason": reason,
                        "person": "Unknown",
                        "recognition_score": 0.0,
                        "ref_img_id": None,
                        "stable": False,
                    }
                    prepared_faces.append(prepared)
                    if valid and crop is not None:
                        face_locations.append((frame_index, face_index))
                        face_crops.append(crop)
                frame_payloads.append({"humans": tracked_humans, "faces": prepared_faces})

            if face_crops:
                embedding_started = time.perf_counter()
                embeddings = self._embedder.embed(face_crops)
                self._last_embedding_ms = (time.perf_counter() - embedding_started) * 1000.0
                if len(embeddings) != len(face_crops):
                    raise RuntimeError("ArcFace embedding count does not match valid faces")
                search_started = time.perf_counter()
                matches = self._vector_store.search_batch(
                    embeddings,
                    self.settings.recognition_threshold,
                )
                self._last_search_ms = (time.perf_counter() - search_started) * 1000.0
                if len(matches) != len(face_crops):
                    raise RuntimeError("Qdrant match count does not match embeddings")
                for (frame_index, face_index), match in zip(face_locations, matches):
                    face = frame_payloads[frame_index]["faces"][face_index]
                    track_id = face["track_id"]
                    stable = (
                        self._tracker(packets[frame_index].source_id).observe(track_id, match)
                        if track_id is not None
                        else match
                    )
                    face.update(
                        {
                            "person": stable.person,
                            "recognition_score": round(stable.score, 6),
                            "ref_img_id": stable.ref_img_id,
                            "stable": stable.person != "Unknown",
                            "raw_person": match.person,
                            "raw_recognition_score": round(match.score, 6),
                        }
                    )
            else:
                self._last_embedding_ms = 0.0
                self._last_search_ms = 0.0

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            output: list[TaskResult] = []
            for packet, payload in zip(packets, frame_payloads):
                faces = payload["faces"]
                recognized = sum(1 for face in faces if face["person"] != "Unknown")
                data = {
                    **payload,
                    "face_count": len(faces),
                    "recognized_count": recognized,
                    "timings_ms": {
                        "detection_batch": round(self._last_detection_ms, 3),
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
        valid: list[tuple[dict[str, Any], np.ndarray, float]] = []
        for face in faces:
            accepted, quality, _, crop = self._quality(image, face)
            if accepted and crop is not None:
                valid.append((face, crop, quality))
        if len(valid) != 1:
            raise ValueError(
                f"Enrollment requires exactly one valid face; found {len(valid)}"
            )
        face, crop, quality = valid[0]
        embedding = self._embedder.embed([crop])[0]
        point_id = self._vector_store.enroll(person, embedding, ref_img_id)
        return {
            "point_id": point_id,
            "person": person,
            "ref_img_id": ref_img_id,
            "bbox": face["bbox"],
            "quality": round(float(quality), 4),
        }

    def identities(self, limit: int = 1000) -> list[dict[str, Any]]:
        self._ensure_dependencies()
        assert self._vector_store is not None
        return self._vector_store.identities(limit=limit)

    def delete_person(self, person: str) -> int:
        self._ensure_dependencies()
        assert self._vector_store is not None
        return self._vector_store.delete_person(person.strip())

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
            "active_sources": len(self._trackers),
            "processed_batches": self._processed_batches,
            "processed_frames": self._processed_frames,
            "detected_faces": self._detected_faces,
            "recognized_faces": self._recognized_faces,
            "last_batch_ms": round(self._last_batch_ms, 3),
            "last_detection_ms": round(self._last_detection_ms, 3),
            "last_embedding_ms": round(self._last_embedding_ms, 3),
            "last_search_ms": round(self._last_search_ms, 3),
            "last_error": self._last_error,
        }

    def close(self) -> None:
        if self._embedder is not None:
            self._embedder.close()
        if self._vector_store is not None:
            self._vector_store.close()
