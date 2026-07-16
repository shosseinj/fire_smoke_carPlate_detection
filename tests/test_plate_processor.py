from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.types import FramePacket
from app.processors.plate import PlateRecognitionProcessor, PlateSettings


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

    def __init__(self, *, has_vehicle: bool = True) -> None:
        self.has_vehicle = has_vehicle
        self.kwargs = None

    def predict(self, source, **kwargs):
        self.kwargs = {"source": source, **kwargs}
        return [
            FakeResult(
                [
                    FakeBox(
                        class_id=2,
                        bbox=(1, 1, 30, 22),
                        confidence=0.9,
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
    def predict(self, values, **kwargs):
        if isinstance(values, list):
            return [FakeOCR("12 B 34567", 0.95) for _ in values]
        return [FakeOCR("12 B 34567", 0.95)]


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
    assert results[0].data["plates"][0]["plate"] == "12B34567"
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
