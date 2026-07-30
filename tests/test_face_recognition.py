from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

import numpy as np
import pytest

from app.core.types import FramePacket
from app.database import Database
from app.processors.face_recognition import (
    FaceEnrollmentValidationError,
    FaceMatch,
    FaceRecognitionProcessor,
    FaceRecognitionSettings,
    PostgresFaceStore,
    QdrantFaceStore,
    SourceFaceTracker,
)


class FakeDetector:
    def __init__(self, *, face: bool) -> None:
        self.face = face
        self.enabled = True
        self.pose_enabled = True
        self.landmarks = np.asarray(
            [[38, 38], [62, 38], [50, 50], [41, 63], [59, 63]],
            dtype=np.float32,
        )
        self.landmark_confidences: np.ndarray | None = None
        self.calls: list[int] = []
        self.input_shapes: list[list[tuple[int, ...]]] = []

    def predict(self, *, source, **_kwargs):
        self.calls.append(len(source))
        self.input_shapes.append([tuple(frame.shape) for frame in source])
        results = []
        for frame in source:
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
            face_count = max(1, (frame.shape[0] - 8) // 128) if self.face else 1
            box = [25, 20, 75, 75] if self.face else [5, 5, 100, 115]
            boxes_array = np.asarray(
                [
                    [box[0], box[1] + 128 * index, box[2], box[3] + 128 * index]
                    for index in range(face_count)
                ],
                dtype=np.float32,
            )
            boxes = SimpleNamespace(
                xyxy=boxes_array,
                conf=np.full((face_count,), 0.95, dtype=np.float32),
                cls=np.zeros((face_count,), dtype=np.float32),
            )
            landmarks = np.stack(
                [
                    self.landmarks + np.asarray([0, 128 * index], dtype=np.float32)
                    for index in range(face_count)
                ]
            )
            landmark_confidences = (
                np.stack([self.landmark_confidences] * face_count)
                if self.face and self.landmark_confidences is not None
                else None
            )
            pose_points = np.stack(
                [
                    np.asarray(
                        [
                            [35, 18], [65, 18], [30, 30], [70, 30],
                            [25, 48], [75, 48], [32, 65], [68, 65],
                            [38, 82], [62, 82], [35, 100], [65, 100],
                            [25, 112], [75, 112], [20, 115], [80, 115],
                            [50, 40],
                        ],
                        dtype=np.float32,
                    )
                    + np.asarray([0, 128 * index], dtype=np.float32)
                    for index in range(face_count)
                ]
            )
            results.append(
                SimpleNamespace(
                    boxes=boxes,
                    keypoints=(
                        SimpleNamespace(xy=landmarks, conf=landmark_confidences)
                        if self.face
                        else SimpleNamespace(
                            xy=pose_points,
                            conf=np.full(pose_points.shape[:2], 0.95, dtype=np.float32),
                        )
                        if self.pose_enabled
                        else None
                    ),
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


class ExpiringByteTracker(FakeByteTracker):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def update(self, detections):
        self.calls += 1
        if self.calls == 1:
            return super().update(detections)
        self.tracked_stracks = []
        self.lost_stracks = [self.track] if self.calls == 2 else []
        return np.empty((0, 8), dtype=np.float32)


def test_tracker_emits_a_track_once_when_byte_track_expires() -> None:
    tracker = SourceFaceTracker(
        high_threshold=0.5,
        low_threshold=0.1,
        new_threshold=0.5,
        match_threshold=0.8,
        max_missed=2,
        history_size=4,
        stable_min_hits=1,
        backend=ExpiringByteTracker(),
    )

    tracker.update([[5, 5, 100, 115]], [0.95])
    tracker.update([], [])
    assert tracker.consume_disappeared() == []
    tracker.update([], [])
    disappeared = tracker.consume_disappeared()

    assert [state.track_id for state in disappeared] == [1]
    assert tracker.consume_disappeared() == []


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


def test_pose_checker_rejects_human_detector_false_positive(tmp_path: Path) -> None:
    processor, human, face, embedder, store = build_processor(tmp_path)
    human.pose_enabled = False

    output = processor.process_batch([packet("cam-a", 1)])[0]

    assert output.data["humans"] == []
    assert output.data["faces"] == []
    assert output.data["human_filter"]["rejected_count"] == 1
    assert output.data["human_filter"]["rejections"][0]["reason"] == (
        "insufficient_keypoints"
    )
    assert face.calls == []
    assert embedder.batch_sizes == []
    assert store.search_batch_sizes == []
    assert processor.status()["human_filter"]["rejected"] == 1


def test_pose_checker_can_be_disabled_for_detection_only_models(tmp_path: Path) -> None:
    processor, human, _face, _embedder, _store = build_processor(tmp_path)
    human.pose_enabled = False
    processor.update_quality_settings({"human_pose_enabled": False})

    output = processor.process_batch([packet("cam-a", 1)])[0]

    assert len(output.data["humans"]) == 1
    assert output.data["human_filter"]["enabled"] is False


def test_pose_checker_rejects_missing_or_low_keypoint_confidence(tmp_path: Path) -> None:
    processor, _human, _face, _embedder, _store = build_processor(tmp_path)
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.asarray([[5, 5, 100, 115]], dtype=np.float32),
            conf=np.asarray([0.95], dtype=np.float32),
            cls=np.asarray([0], dtype=np.float32),
        ),
        keypoints=SimpleNamespace(
            xy=np.ones((1, 17, 2), dtype=np.float32),
            conf=np.zeros((1, 17), dtype=np.float32),
        ),
    )

    accepted, rejected = processor._human_boxes(result)

    assert accepted == []
    assert rejected[0]["human_pose_keypoint_count"] == 0

    result.keypoints = SimpleNamespace(xy=np.ones((1, 17, 2), dtype=np.float32))
    accepted, rejected = processor._human_boxes(result)
    assert accepted == []
    assert rejected[0]["reason"] == "insufficient_keypoints"


def test_face_quality_rejects_face_box_away_from_human_head_pose(
    tmp_path: Path,
) -> None:
    processor, _human, _face, _embedder, _store = build_processor(tmp_path)
    human = {
        "human_pose_keypoints": np.asarray(
            [[20, 20], [17, 18], [23, 18], [14, 20], [26, 20]],
            dtype=np.float32,
        ),
        "human_pose_keypoint_confidences": np.full(5, 0.95, dtype=np.float32),
    }
    assert processor._face_matches_human_head([70, 70, 110, 110], human) is False

    frame = packet("cam-a", 1).source_frame
    face = {
        "bbox": [70, 70, 110, 110],
        "landmarks": np.asarray(
            [[78, 80], [101, 80], [90, 91], [81, 102], [99, 102]],
            dtype=np.float32,
        ),
        "confidence": 0.99,
        "human_head_consistent": False,
    }
    valid, _quality, reason, crop, metrics = processor._quality(frame, face)

    assert valid is False
    assert reason == "face_pose_mismatch"
    assert crop is None
    assert metrics["human_head_consistent"] is False


def test_source_thresholds_are_applied_before_face_processing(tmp_path: Path) -> None:
    processor, _human, face, embedder, store = build_processor(tmp_path)
    processor._settings_provider = lambda _source: {
        "face_human_confidence": 0.99,
        "face_detection_confidence": 0.50,
        "face_recognition_threshold": 0.45,
    }

    output = processor.process_batch([packet("cam-a", 1)])[0]

    assert output.data["humans"] == []
    assert face.calls == []
    assert embedder.batch_sizes == []
    assert store.search_batch_sizes == []


def test_quality_weight_is_applied_to_recognition_score(tmp_path: Path) -> None:
    processor, _human, _face, _embedder, _store = build_processor(tmp_path)
    processor.update_quality_settings(
        {
            "quality_threshold": 0.0,
            "blur_threshold": 0.0,
            "recognition_quality_weight": 1.0,
        }
    )

    output = processor.process_batch([packet("cam-a", 1)])[0]
    face_result = output.data["faces"][0]

    assert face_result["quality_valid"] is True
    assert face_result["quality_adjusted_score"] == pytest.approx(
        0.91 * face_result["quality_score"], abs=1e-5
    )
    assert face_result["raw_recognition_score"] == pytest.approx(0.91)


def test_quality_adjusted_match_below_threshold_is_not_admitted(tmp_path: Path) -> None:
    processor, _human, _face, _embedder, store = build_processor(tmp_path)
    store.search_batch = lambda embeddings, threshold: [
        FaceMatch("Alice", 0.91, "reference-1") for _ in embeddings
    ]
    processor.settings = replace(processor.settings, recognition_threshold=0.99)

    face_result = processor.process_batch([packet("cam-a", 1)])[0].data["faces"][0]

    assert face_result["person"] == "Unknown"
    assert face_result["recognition_admitted"] is False
    assert face_result["raw_person"] == "Unknown"


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


def test_detection_stays_low_resolution_but_quality_uses_native_frame(
    tmp_path: Path,
) -> None:
    processor, human, face, embedder, store = build_processor(tmp_path)
    processor.update_quality_settings(
        {
            "quality_threshold": 0.0,
            "blur_threshold": 0.0,
            "max_abs_yaw": 90.0,
            "max_abs_pitch": 90.0,
            "max_abs_roll": 90.0,
        }
    )
    inference_packet = packet("cam-2k", 1)
    source_checker = np.indices((360, 240)).sum(axis=0) % 2
    source_frame = np.repeat(
        (source_checker * 255).astype(np.uint8)[:, :, None], 3, axis=2
    )
    inference_packet = FramePacket(
        source_id=inference_packet.source_id,
        frame=inference_packet.frame,
        round_sequence=inference_packet.round_sequence,
        frame_index=inference_packet.frame_index,
        captured_monotonic=inference_packet.captured_monotonic,
        captured_at_utc=inference_packet.captured_at_utc,
        metadata={"source_frame": source_frame},
    )

    output = processor.process_batch([inference_packet])[0]

    assert human.input_shapes[0][0] == (120, 120, 3)
    assert face.input_shapes[0][0] == (136, 124, 3)
    detected_face = output.data["faces"][0]
    assert detected_face["bbox"] == [17.0, 12.0, 67.0, 67.0]
    assert detected_face["quality_metrics"]["face_width"] == 100
    assert detected_face["quality_metrics"]["face_height"] == 165
    assert detected_face["quality_metrics"]["quality_frame_width"] == 240
    assert detected_face["quality_metrics"]["quality_frame_height"] == 360
    assert output.data["inference_frame_size"] == {"width": 120, "height": 120}
    assert output.data["source_frame_size"] == {"width": 240, "height": 360}
    assert embedder.batch_sizes == [1]
    assert store.search_batch_sizes == [1]


def test_quality_gate_blocks_low_score_and_out_of_pose_faces(tmp_path: Path) -> None:
    processor, _human, face, embedder, store = build_processor(tmp_path)
    processor.update_quality_settings({"quality_threshold": 1.0})

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
            "max_abs_pitch": 90.0,
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


def test_quality_gate_rejects_back_of_head_without_visible_eyes_and_nose(
    tmp_path: Path,
) -> None:
    processor, _human, face, embedder, store = build_processor(tmp_path)
    processor.update_quality_settings(
        {
            "quality_threshold": 0.0,
            "blur_threshold": 0.0,
            "require_landmarks": False,
        }
    )
    # A detector can emit a face box on a neck/back-of-head region. Five
    # collapsed or implausible landmarks must not make that box a real face.
    face.landmarks = np.asarray(
        [[50, 38], [50, 38], [50, 35], [48, 63], [52, 63]],
        dtype=np.float32,
    )

    output = processor.process_batch([packet("back-of-head", 1)])[0]
    detected = output.data["faces"][0]

    assert detected["quality_valid"] is False
    assert detected["quality_reason"] == "missing_eyes_or_nose"
    assert detected["quality_metrics"]["facial_features_visible"] is False
    assert embedder.batch_sizes == []
    assert store.search_batch_sizes == []


def test_facial_feature_gate_rejects_low_confidence_eye_or_nose() -> None:
    landmarks = np.asarray(
        [[38, 38], [62, 38], [50, 50], [41, 63], [59, 63]],
        dtype=np.float32,
    )
    assert FaceRecognitionProcessor._facial_features_visible(
        landmarks,
        [25, 20, 75, 75],
        [0.95, 0.10, 0.95, 0.95, 0.95],
    ) is False
    assert FaceRecognitionProcessor._facial_features_visible(
        landmarks,
        [25, 20, 75, 75],
        [0.95, 0.95, 0.95, 0.95, 0.95],
    ) is True


def test_runtime_quality_rejects_low_confidence_eye_landmark(tmp_path: Path) -> None:
    processor, _human, face, embedder, store = build_processor(tmp_path)
    processor.update_quality_settings(
        {"quality_threshold": 0.0, "blur_threshold": 0.0}
    )
    face.landmark_confidences = np.asarray(
        [0.95, 0.10, 0.95, 0.95, 0.95], dtype=np.float32
    )

    detected = processor.process_batch([packet("occluded-eye", 1)])[0].data["faces"][0]

    assert detected["quality_valid"] is False
    assert detected["quality_reason"] == "missing_eyes_or_nose"
    assert embedder.batch_sizes == []
    assert store.search_batch_sizes == []


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


def test_enrollment_reports_persian_quality_rejection_details(tmp_path: Path) -> None:
    processor, _human, _face, _embedder, _store = build_processor(tmp_path)
    processor.update_quality_settings({"quality_threshold": 0.99})

    with pytest.raises(FaceEnrollmentValidationError) as captured:
        processor.enroll(packet("cam-a", 1).frame, person="Alice")

    error = captured.value
    assert error.code == "face_quality_rejected"
    assert "چهره شناسایی شد" in str(error)
    assert error.details["detected_faces"] == 1
    assert error.details["valid_faces"] == 0
    assert error.details["rejections"][0]["reason"] == "quality_below_threshold"


def test_enrollment_rejects_back_of_head_as_missing_eyes_or_nose(
    tmp_path: Path,
) -> None:
    processor, _human, face, _embedder, _store = build_processor(tmp_path)
    processor.update_quality_settings(
        {
            "quality_threshold": 0.0,
            "blur_threshold": 0.0,
            "require_landmarks": False,
        }
    )
    face.landmarks = np.asarray(
        [[50, 38], [50, 38], [50, 35], [48, 63], [52, 63]],
        dtype=np.float32,
    )

    with pytest.raises(FaceEnrollmentValidationError) as captured:
        processor.enroll(packet("back-of-head", 1).frame, person="Alice")

    rejection = captured.value.details["rejections"][0]
    assert rejection["reason"] == "missing_eyes_or_nose"
    assert "چشم" in rejection["message"]
    assert "بینی" in rejection["message"]


def test_enrollment_reports_persian_no_face_error(tmp_path: Path) -> None:
    processor, _human, face, _embedder, _store = build_processor(tmp_path)
    face.enabled = False

    with pytest.raises(FaceEnrollmentValidationError) as captured:
        processor.enroll(packet("cam-a", 1).frame, person="Alice")

    assert captured.value.code == "no_face_detected"
    assert "هیچ چهره‌ای" in str(captured.value)


def test_remote_qdrant_store_batches_search_and_manages_identity(tmp_path: Path) -> None:
    pytest.importorskip("qdrant_client")
    collection = f"faces-test-{uuid.uuid4().hex}"
    try:
        store = QdrantFaceStore(
            FaceRecognitionSettings(
                human_model_path=tmp_path / "human.engine",
                face_model_path=tmp_path / "face.engine",
                embedding_model_path=tmp_path / "arcface.engine",
                vector_size=3,
                qdrant_url="http://127.0.0.1:6333",
                qdrant_collection=collection,
            )
        )
    except Exception as exc:
        pytest.skip(f"Qdrant service not reachable for integration test: {type(exc).__name__}: {exc}")
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
        store.client.delete_collection(collection)
        store.close()


def test_qdrant_store_recreates_collection_before_enrollment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeQdrantClient:
        def __init__(self, **_kwargs: object) -> None:
            self.collection_exists_value = False
            self.create_calls: list[dict[str, object]] = []
            self.upsert_calls: list[dict[str, object]] = []

        def collection_exists(self, _collection: str) -> bool:
            return self.collection_exists_value

        def create_collection(self, **kwargs: object) -> None:
            self.create_calls.append(kwargs)
            self.collection_exists_value = True

        def upsert(self, **kwargs: object) -> None:
            assert self.collection_exists_value is True
            self.upsert_calls.append(kwargs)

    fake_models = SimpleNamespace(
        Distance=SimpleNamespace(COSINE="cosine"),
        VectorParams=lambda **kwargs: kwargs,
        PointStruct=lambda **kwargs: kwargs,
    )
    fake_qdrant_module = SimpleNamespace(
        QdrantClient=FakeQdrantClient,
        models=fake_models,
    )
    monkeypatch.setitem(sys.modules, "qdrant_client", fake_qdrant_module)

    store = QdrantFaceStore(
        FaceRecognitionSettings(
            human_model_path=tmp_path / "human.engine",
            face_model_path=tmp_path / "face.engine",
            embedding_model_path=tmp_path / "arcface.engine",
            vector_size=3,
            qdrant_url="http://qdrant:6333",
            qdrant_collection="faces",
        )
    )
    assert len(store.client.create_calls) == 1

    store.client.collection_exists_value = False
    point_id = store.enroll(
        "1234567891",
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "42",
    )

    assert point_id
    assert len(store.client.create_calls) == 2
    assert len(store.client.upsert_calls) == 1
    assert store.client.upsert_calls[0]["collection_name"] == "faces"


def test_qdrant_store_tolerates_concurrent_collection_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RacingQdrantClient:
        def __init__(self, **_kwargs: object) -> None:
            self.collection_exists_value = False
            self.create_calls = 0
            self.upsert_calls = 0

        def collection_exists(self, _collection: str) -> bool:
            return self.collection_exists_value

        def create_collection(self, **_kwargs: object) -> None:
            self.create_calls += 1
            self.collection_exists_value = True
            raise RuntimeError("collection already exists")

        def upsert(self, **_kwargs: object) -> None:
            self.upsert_calls += 1

    fake_models = SimpleNamespace(
        Distance=SimpleNamespace(COSINE="cosine"),
        VectorParams=lambda **kwargs: kwargs,
        PointStruct=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(
        sys.modules,
        "qdrant_client",
        SimpleNamespace(QdrantClient=RacingQdrantClient, models=fake_models),
    )

    store = QdrantFaceStore(
        FaceRecognitionSettings(
            human_model_path=tmp_path / "human.engine",
            face_model_path=tmp_path / "face.engine",
            embedding_model_path=tmp_path / "arcface.engine",
            vector_size=3,
            qdrant_url="http://qdrant:6333",
            qdrant_collection="faces",
        )
    )
    store.enroll(
        "1234567891",
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "42",
    )

    assert store.client.create_calls == 1
    assert store.client.upsert_calls == 1


def test_postgresql_vector_store_is_persistent_and_uses_cosine(
    postgres_database: Database,
) -> None:
    store = PostgresFaceStore(postgres_database, vector_size=3)
    try:
        store.enroll("Alice", np.asarray([1.0, 0.0, 0.0]), "reference-1")
        matches = store.search_batch(
            np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            threshold=0.8,
        )
        assert [match.person for match in matches] == ["Alice", "Unknown"]
        assert store.status()["mode"] == "postgresql"
    finally:
        store.close()

    reopened = PostgresFaceStore(postgres_database, vector_size=3)
    try:
        assert reopened.identities() == [
            {"person": "Alice", "ref_img_id": "reference-1", "embeddings": 1}
        ]
    finally:
        reopened.close()


def test_qdrant_init_failure_falls_back_to_postgresql(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenQdrantFaceStore:
        def __init__(self, settings: FaceRecognitionSettings) -> None:
            raise RuntimeError("remote disconnected")

    class FallbackStore:
        def __init__(self, database: object, vector_size: int) -> None:
            self.database = database
            self.vector_size = vector_size

        def status(self) -> dict[str, object]:
            return {"mode": "postgresql", "points": 0}

    monkeypatch.setattr(
        "app.processors.face_recognition.QdrantFaceStore",
        BrokenQdrantFaceStore,
    )
    monkeypatch.setattr(
        "app.processors.face_recognition.PostgresFaceStore",
        FallbackStore,
    )
    fake_database = object()
    processor = FaceRecognitionProcessor(
        FaceRecognitionSettings(
            human_model_path=tmp_path / "human.engine",
            face_model_path=tmp_path / "face.engine",
            embedding_model_path=tmp_path / "arcface.engine",
            qdrant_url="http://qdrant.invalid:6333",
            vector_size=3,
        ),
        human_detector=FakeDetector(face=False),
        face_detector=FakeDetector(face=True),
        embedder=FakeEmbedder(),
        database=fake_database,
        tracker_backend_factory=FakeByteTracker,
    )

    processor._ensure_dependencies()

    status = processor.status()
    assert status["qdrant"]["mode"] == "postgresql"
    assert "Qdrant unavailable; using PostgreSQL fallback" in status["vector_store_warning"]
