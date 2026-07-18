from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.core.types import FramePacket
from app.processors.face_recognition import (
    FaceMatch,
    FaceRecognitionProcessor,
    FaceRecognitionSettings,
    QdrantFaceStore,
    SqliteFaceStore,
)


class FakeDetector:
    def __init__(self, *, face: bool) -> None:
        self.face = face
        self.enabled = True
        self.landmarks = np.asarray(
            [[38, 38], [62, 38], [50, 50], [41, 63], [59, 63]],
            dtype=np.float32,
        )
        self.calls: list[int] = []

    def predict(self, *, source, **_kwargs):
        self.calls.append(len(source))
        results = []
        for _ in source:
            if not self.enabled:
                results.append(
                    SimpleNamespace(
                        boxes=SimpleNamespace(
                            xyxy=np.empty((0, 4), dtype=np.float32),
                            conf=np.empty((0,), dtype=np.float32),
                            cls=np.empty((0,), dtype=np.float32),
                        ),
                        keypoints=None,
                    )
                )
                continue
            box = [25, 20, 75, 75] if self.face else [5, 5, 100, 115]
            boxes = SimpleNamespace(
                xyxy=np.asarray([box], dtype=np.float32),
                conf=np.asarray([0.95], dtype=np.float32),
                cls=np.asarray([0], dtype=np.float32),
            )
            landmarks = self.landmarks.reshape(1, 5, 2)
            results.append(
                SimpleNamespace(
                    boxes=boxes,
                    keypoints=SimpleNamespace(xy=landmarks) if self.face else None,
                )
            )
        return results


class FakeEmbedder:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def embed(self, faces):
        self.batch_sizes.append(len(faces))
        values = np.zeros((len(faces), 512), dtype=np.float32)
        values[:, 0] = 1.0
        return values

    def close(self) -> None:
        return None


class FakeStore:
    def __init__(self) -> None:
        self.search_batch_sizes: list[int] = []
        self.rows: list[dict] = []

    def search_batch(self, embeddings, threshold):
        self.search_batch_sizes.append(len(embeddings))
        assert threshold == 0.45
        return [FaceMatch("Alice", 0.91, "reference-1") for _ in embeddings]

    def enroll(self, person, embedding, ref_img_id):
        self.rows.append({"person": person, "ref_img_id": ref_img_id})
        return "point-1"

    def identities(self, limit=1000):
        return [{**row, "embeddings": 1} for row in self.rows[:limit]]

    def delete_person(self, person):
        before = len(self.rows)
        self.rows = [row for row in self.rows if row["person"] != person]
        return before - len(self.rows)

    def status(self):
        return {"mode": "fake", "collection": "faces", "points": len(self.rows)}

    def close(self) -> None:
        return None


class FakeByteTrack:
    def __init__(self) -> None:
        self.idx = 0
        self.is_activated = True


class FakeByteTracker:
    def __init__(self) -> None:
        self.track = FakeByteTrack()
        self.tracked_stracks = []
        self.lost_stracks = []

    def update(self, detections):
        if len(detections):
            self.track.idx = 0
            self.tracked_stracks = [self.track]
        else:
            self.tracked_stracks = []
            self.lost_stracks = [self.track]
        return np.empty((0, 8), dtype=np.float32)


def packet(source_id: str, frame_index: int) -> FramePacket:
    checker = np.indices((120, 120)).sum(axis=0) % 2
    frame = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    return FramePacket(
        source_id=source_id,
        frame=frame,
        round_sequence=frame_index,
        frame_index=frame_index,
        captured_monotonic=1.0,
        captured_at_utc="2026-07-18T00:00:00+00:00",
    )


def build_processor(tmp_path: Path):
    human = FakeDetector(face=False)
    face = FakeDetector(face=True)
    embedder = FakeEmbedder()
    store = FakeStore()
    processor = FaceRecognitionProcessor(
        FaceRecognitionSettings(
            human_model_path=tmp_path / "human.engine",
            face_model_path=tmp_path / "face.engine",
            embedding_model_path=tmp_path / "arcface.engine",
            human_engine_fixed_batch=8,
            face_engine_fixed_batch=8,
            stable_min_hits=2,
            blur_threshold=1.0,
        ),
        human_detector=human,
        face_detector=face,
        embedder=embedder,
        vector_store=store,
        tracker_backend_factory=FakeByteTracker,
    )
    return processor, human, face, embedder, store


def test_batches_across_sources_and_stabilizes_per_track(tmp_path: Path) -> None:
    processor, human, face, embedder, store = build_processor(tmp_path)

    first = processor.process_batch([packet("cam-a", 1), packet("cam-b", 1)])
    second = processor.process_batch([packet("cam-a", 2), packet("cam-b", 2)])

    assert human.calls == [8, 8]
    assert face.calls == [8, 8]
    assert embedder.batch_sizes == [2, 2]
    assert store.search_batch_sizes == [2, 2]
    assert [item.data["faces"][0]["raw_person"] for item in first] == ["Alice", "Alice"]
    assert [item.data["faces"][0]["person"] for item in first] == ["Unknown", "Unknown"]
    assert [item.data["faces"][0]["person"] for item in second] == ["Alice", "Alice"]
    assert all(item.data["faces"][0]["track_id"] == 1 for item in second)
    assert all(item.data["recognized_count"] == 1 for item in second)
    assert all(item.data["humans"][0]["person"] == "Alice" for item in second)


def test_human_keeps_track_identity_when_face_is_no_longer_visible(tmp_path: Path) -> None:
    processor, _human, face, _embedder, _store = build_processor(tmp_path)

    processor.process_batch([packet("cam-a", 1)])
    recognized = processor.process_batch([packet("cam-a", 2)])[0]
    face.enabled = False
    rotated = processor.process_batch([packet("cam-a", 3)])[0]

    assert recognized.data["humans"][0]["track_id"] == 1
    assert recognized.data["humans"][0]["person"] == "Alice"
    assert rotated.data["faces"] == []
    assert rotated.data["humans"][0]["track_id"] == 1
    assert rotated.data["humans"][0]["person"] == "Alice"
    assert rotated.data["humans"][0]["identity_stable"] is True
    assert rotated.data["humans"][0]["face_visible"] is False


def test_quality_gate_blocks_low_score_and_out_of_pose_faces(tmp_path: Path) -> None:
    processor, _human, face, embedder, store = build_processor(tmp_path)
    processor.update_quality_settings({"quality_threshold": 0.99})

    low_quality = processor.process_batch([packet("cam-a", 1)])[0]

    assert low_quality.data["faces"][0]["quality_valid"] is False
    assert low_quality.data["faces"][0]["quality_reason"] == (
        "quality_below_threshold"
    )
    assert low_quality.data["faces"][0]["quality_metrics"]["yaw"] is not None
    assert embedder.batch_sizes == []
    assert store.search_batch_sizes == []

    processor.update_quality_settings(
        {"quality_threshold": 0.0, "min_face_width": 100, "min_face_height": 100}
    )
    too_small = processor.process_batch([packet("cam-size", 1)])[0]

    assert too_small.data["faces"][0]["quality_valid"] is False
    assert too_small.data["faces"][0]["quality_reason"] == "face_too_small"
    assert too_small.data["faces"][0]["quality_metrics"]["face_width"] < 100
    assert too_small.data["faces"][0]["quality_metrics"]["face_height"] < 100

    processor.update_quality_settings(
        {
            "quality_threshold": 0.0,
            "min_face_width": 24,
            "min_face_height": 24,
            "max_abs_roll": 10.0,
        }
    )
    face.landmarks = np.asarray(
        [[38, 30], [62, 55], [50, 50], [41, 63], [59, 63]],
        dtype=np.float32,
    )
    bad_roll = processor.process_batch([packet("cam-a", 2)])[0]

    assert bad_roll.data["faces"][0]["quality_valid"] is False
    assert bad_roll.data["faces"][0]["quality_reason"] == "roll_out_of_range"
    assert abs(bad_roll.data["faces"][0]["quality_metrics"]["roll"]) > 10.0
    assert embedder.batch_sizes == []


def test_enrollment_uses_same_detector_embedder_and_store(tmp_path: Path) -> None:
    processor, _human, face, embedder, store = build_processor(tmp_path)
    image = packet("cam-a", 1).frame

    enrolled = processor.enroll(image, person="Alice", ref_img_id="reference-1")

    assert face.calls == [8]
    assert embedder.batch_sizes == [1]
    assert enrolled["point_id"] == "point-1"
    assert processor.identities() == [
        {"person": "Alice", "ref_img_id": "reference-1", "embeddings": 1}
    ]
    assert processor.delete_person("Alice") == 1


def test_local_qdrant_store_batches_search_and_manages_identity(tmp_path: Path) -> None:
    store = QdrantFaceStore(
        FaceRecognitionSettings(
            human_model_path=tmp_path / "human.engine",
            face_model_path=tmp_path / "face.engine",
            embedding_model_path=tmp_path / "arcface.engine",
            vector_size=3,
            qdrant_path=tmp_path / "qdrant",
        )
    )
    try:
        store.enroll("Alice", np.asarray([1.0, 0.0, 0.0]), "reference-1")
        matches = store.search_batch(
            np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            threshold=0.8,
        )
        assert [match.person for match in matches] == ["Alice", "Unknown"]
        assert store.identities() == [
            {"person": "Alice", "ref_img_id": "reference-1", "embeddings": 1}
        ]
        assert store.delete_person("Alice") == 1
        assert store.status()["points"] == 0
    finally:
        store.close()


def test_sqlite_vector_fallback_is_persistent_and_uses_cosine(tmp_path: Path) -> None:
    path = tmp_path / "face_embeddings.sqlite3"
    store = SqliteFaceStore(path, vector_size=3)
    try:
        store.enroll("Alice", np.asarray([1.0, 0.0, 0.0]), "reference-1")
        matches = store.search_batch(
            np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            threshold=0.8,
        )
        assert [match.person for match in matches] == ["Alice", "Unknown"]
        assert store.status()["mode"] == "sqlite-fallback"
    finally:
        store.close()

    reopened = SqliteFaceStore(path, vector_size=3)
    try:
        assert reopened.identities() == [
            {"person": "Alice", "ref_img_id": "reference-1", "embeddings": 1}
        ]
    finally:
        reopened.close()
