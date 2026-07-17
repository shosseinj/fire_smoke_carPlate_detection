from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.types import FramePacket
from app.core.plate_settings_store import PlateDetectionPolicy
from app.processors.plate import (
    PlateRecognitionProcessor,
    PlateSettings,
    is_valid_iranian_plate,
)


class FakeBox:
    def __init__(
        self,
        *,
        class_id: int,
        bbox: tuple[int, int, int, int],
        confidence: float = 0.9,
    ) -> None:
        self.cls = np.array([class_id], dtype=np.float32)
        self.xyxy = np.array([bbox], dtype=np.float32)
        self.conf = np.array([confidence], dtype=np.float32)


class FakeResult:
    def __init__(self, boxes: list[FakeBox]) -> None:
        self.boxes = boxes


class FakePlateDetector:
    names = {0: "license_plate"}

    def __init__(self, confidence: float = 0.9) -> None:
        self.kwargs = None
        self.confidence = confidence

    def predict(self, source, **kwargs):
        self.kwargs = {"source": source, **kwargs}
        return [
            FakeResult(
                [
                    FakeBox(
                        class_id=0,
                        bbox=(2, 2, 20, 10),
                        confidence=self.confidence,
                    )
                ]
            )
            for _ in source
        ]


class FakeVehicleDetector:
    names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def __init__(
        self,
        *,
        has_vehicle: bool = True,
        bbox: tuple[int, int, int, int] = (1, 1, 30, 22),
        confidence: float = 0.9,
    ) -> None:
        self.has_vehicle = has_vehicle
        self.bbox = bbox
        self.confidence = confidence
        self.kwargs = None

    def predict(self, source, **kwargs):
        self.kwargs = {"source": source, **kwargs}
        return [
            FakeResult(
                [
                    FakeBox(
                        class_id=2,
                        bbox=self.bbox,
                        confidence=self.confidence,
                    )
                ]
                if self.has_vehicle
                else []
            )
            for _ in source
        ]


@dataclass
class FakeOCR:
    text: str
    score: float


class FakeRecognizer:
    def __init__(self, text: str = "12 ب 34567", score: float = 0.95) -> None:
        self.text = text
        self.score = score

    def predict(self, values, **kwargs):
        if isinstance(values, list):
            return [FakeOCR(self.text, self.score) for _ in values]
        return [FakeOCR(self.text, self.score)]


def make_packet(source_id: str) -> FramePacket:
    return FramePacket(
        source_id=source_id,
        frame=np.zeros((24, 32, 3), dtype=np.uint8),
        round_sequence=1,
        frame_index=1,
        captured_monotonic=time.monotonic(),
        captured_at_utc="2026-01-01T00:00:00+00:00",
    )


def settings(tmp_path: Path, **overrides) -> PlateSettings:
    values = {
        "detector_weights": tmp_path / "plate.pt",
        "vehicle_detector_weights": tmp_path / "vehicle.pt",
        "recognizer_model_dir": tmp_path / "recognizer",
        "device": "0",
        "use_fp16": True,
        "vehicle_crop_padding_ratio": 0.0,
        "min_vehicle_width_pixels": 1,
        "min_vehicle_height_pixels": 1,
        "min_vehicle_area_ratio": 0.0,
    }
    values.update(overrides)
    return PlateSettings(**values)


def test_vehicle_plate_and_ocr_stages_are_batched(tmp_path: Path) -> None:
    vehicle_detector = FakeVehicleDetector()
    plate_detector = FakePlateDetector()
    processor = PlateRecognitionProcessor(
        settings(tmp_path),
        vehicle_detector=vehicle_detector,
        detector=plate_detector,
        recognizer=FakeRecognizer(),
    )

    results = processor.process_batch(
        [make_packet("camera-01"), make_packet("camera-02")]
    )

    assert len(results) == 2
    assert len(vehicle_detector.kwargs["source"]) == 2
    assert vehicle_detector.kwargs["classes"] == [2, 3, 5, 7]
    assert len(plate_detector.kwargs["source"]) == 2
    assert plate_detector.kwargs["quantize"] == 16
    assert "half" not in plate_detector.kwargs
    assert results[0].data["plates"][0]["plate"] == "12ب34567"
    assert results[0].data["plates"][0]["bbox"] == [3, 3, 21, 11]
    assert results[0].data["plates"][0]["vehicle_bbox"] == [1, 1, 30, 22]
    assert results[0].data["vehicle_count"] == 1
    assert results[1].data["plate_count"] == 1


def test_plate_below_configured_score_is_not_recognized(tmp_path: Path) -> None:
    processor = PlateRecognitionProcessor(
        settings(tmp_path, device="cpu", detector_confidence=0.30),
        vehicle_detector=FakeVehicleDetector(),
        detector=FakePlateDetector(confidence=0.29),
        recognizer=FakeRecognizer(),
    )
    result = processor.process_batch([make_packet("camera-low-score")])[0]

    assert result.data["plates"] == []
    assert result.data["plate_count"] == 0


def test_plate_stage_is_skipped_when_no_vehicle_exists(tmp_path: Path) -> None:
    plate_detector = FakePlateDetector()
    processor = PlateRecognitionProcessor(
        settings(tmp_path, device="cpu"),
        vehicle_detector=FakeVehicleDetector(has_vehicle=False),
        detector=plate_detector,
        recognizer=FakeRecognizer(),
    )
    result = processor.process_batch([make_packet("camera-no-vehicle")])[0]

    assert plate_detector.kwargs is None
    assert result.data["vehicle_count"] == 0
    assert result.data["plates"] == []


def test_iranian_plate_requires_exactly_seven_digits_and_one_persian_letter() -> None:
    assert is_valid_iranian_plate("12 ب 34567") is True
    assert is_valid_iranian_plate("۱۲ ب ۳۴۵۶۷") is True
    assert is_valid_iranian_plate("12 ب 3456") is False
    assert is_valid_iranian_plate("12 B 34567") is False
    assert is_valid_iranian_plate("1234567") is False


def test_partial_ocr_is_rejected(tmp_path: Path) -> None:
    processor = PlateRecognitionProcessor(
        settings(tmp_path, device="cpu"),
        vehicle_detector=FakeVehicleDetector(),
        detector=FakePlateDetector(),
        recognizer=FakeRecognizer("12 ب 3456"),
    )

    result = processor.process_batch([make_packet("camera-partial")])[0]

    assert result.data["plates"] == []
    assert result.data["rejected_plate_formats"] == 1


def test_small_vehicle_is_rejected_before_plate_inference(tmp_path: Path) -> None:
    plate_detector = FakePlateDetector()
    processor = PlateRecognitionProcessor(
        settings(
            tmp_path,
            device="cpu",
            min_vehicle_width_pixels=30,
            min_vehicle_height_pixels=20,
            min_vehicle_area_ratio=0.2,
        ),
        vehicle_detector=FakeVehicleDetector(bbox=(1, 1, 12, 9)),
        detector=plate_detector,
        recognizer=FakeRecognizer(),
    )

    result = processor.process_batch([make_packet("camera-far")])[0]

    assert plate_detector.kwargs is None
    assert result.data["vehicle_count"] == 0
    assert result.data["rejected_small_vehicles"] == 1


def test_each_camera_uses_its_effective_policy_in_the_same_batch(
    tmp_path: Path,
) -> None:
    policies = {
        "camera-accept": PlateDetectionPolicy(
            min_vehicle_width_pixels=1,
            min_vehicle_height_pixels=1,
            min_vehicle_area_ratio=0.0,
        ),
        "camera-reject": PlateDetectionPolicy(
            min_vehicle_width_pixels=31,
            min_vehicle_height_pixels=23,
            min_vehicle_area_ratio=0.0,
        ),
    }
    processor = PlateRecognitionProcessor(
        settings(tmp_path, device="cpu"),
        vehicle_detector=FakeVehicleDetector(),
        detector=FakePlateDetector(),
        recognizer=FakeRecognizer(),
        settings_provider=policies.__getitem__,
    )

    accepted, rejected = processor.process_batch(
        [make_packet("camera-accept"), make_packet("camera-reject")]
    )

    assert accepted.data["plate_count"] == 1
    assert accepted.data["effective_settings"]["min_vehicle_width_pixels"] == 1
    assert rejected.data["plate_count"] == 0
    assert rejected.data["rejected_small_vehicles"] == 1


def test_each_camera_plate_threshold_is_applied_after_shared_gpu_batch(
    tmp_path: Path,
) -> None:
    common = {
        "min_vehicle_width_pixels": 1,
        "min_vehicle_height_pixels": 1,
        "min_vehicle_area_ratio": 0.0,
    }
    policies = {
        "camera-low-threshold": PlateDetectionPolicy(
            plate_confidence=0.30,
            **common,
        ),
        "camera-high-threshold": PlateDetectionPolicy(
            plate_confidence=0.95,
            **common,
        ),
    }
    plate_detector = FakePlateDetector(confidence=0.90)
    processor = PlateRecognitionProcessor(
        settings(tmp_path, device="cpu"),
        vehicle_detector=FakeVehicleDetector(),
        detector=plate_detector,
        recognizer=FakeRecognizer(),
        settings_provider=policies.__getitem__,
    )

    accepted, rejected = processor.process_batch(
        [
            make_packet("camera-low-threshold"),
            make_packet("camera-high-threshold"),
        ]
    )

    assert plate_detector.kwargs["conf"] == 0.30
    assert accepted.data["plate_count"] == 1
    assert rejected.data["plate_count"] == 0
    assert rejected.data["rejected_plate_scores"] == 1


def test_vehicle_engine_runtime_failure_uses_onnx_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    engine = tmp_path / "vehicle.engine"
    onnx = tmp_path / "vehicle.onnx"
    engine.write_bytes(b"engine")
    onnx.write_bytes(b"onnx")

    class FailingVehicleModel:
        names = FakeVehicleDetector.names

        def predict(self, *args, **kwargs):
            raise RuntimeError("incompatible TensorRT engine")

    def fake_yolo(path: str):
        if Path(path).suffix == ".engine":
            return FailingVehicleModel()
        return FakeVehicleDetector()

    monkeypatch.setattr("app.processors.plate.load_yolo_class", lambda: fake_yolo)
    processor = PlateRecognitionProcessor(
        settings(tmp_path, device="cpu"),
        detector=FakePlateDetector(),
        recognizer=FakeRecognizer(),
        vehicle_model_provider=lambda: (1, [engine, onnx]),
    )

    result = processor.process_batch([make_packet("camera-fallback")])[0]

    assert result.data["plate_count"] == 1
    assert processor.status()["vehicle_detector_weights"] == str(onnx)
    assert processor.status()["vehicle_model_fallbacks"] == 1
