from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.types import FramePacket
from app.processors.plate import PlateRecognitionProcessor, PlateSettings


class FakeBox:
    def __init__(self) -> None:
        self.cls = np.array([1], dtype=np.float32)
        self.xyxy = np.array([[2, 2, 20, 10]], dtype=np.float32)
        self.conf = np.array([0.9], dtype=np.float32)


class FakeResult:
    def __init__(self) -> None:
        self.boxes = [FakeBox()]


class FakeDetector:
    names = {0: "car", 1: "plate"}

    def __init__(self) -> None:
        self.kwargs = None

    def predict(self, source, **kwargs):
        self.kwargs = kwargs
        return [FakeResult() for _ in source]


@dataclass
class FakeOCR:
    text: str
    score: float


class FakeRecognizer:
    def predict(self, values, **kwargs):
        if isinstance(values, list):
            return [FakeOCR("۱۲ ب ۳۴۵۶۷", 0.95) for _ in values]
        return [FakeOCR("۱۲ ب ۳۴۵۶۷", 0.95)]


def make_packet(source_id: str) -> FramePacket:
    return FramePacket(
        source_id=source_id,
        frame=np.zeros((24, 32, 3), dtype=np.uint8),
        round_sequence=1,
        frame_index=1,
        captured_monotonic=time.monotonic(),
        captured_at_utc="2026-01-01T00:00:00+00:00",
    )


def test_plate_detector_and_ocr_are_batched(tmp_path: Path) -> None:
    detector = FakeDetector()
    processor = PlateRecognitionProcessor(
        PlateSettings(
            detector_weights=tmp_path / "detector.pt",
            recognizer_model_dir=tmp_path / "recognizer",
            device="0",
            use_fp16=True,
        ),
        detector=detector,
        recognizer=FakeRecognizer(),
    )
    results = processor.process_batch([make_packet("camera-01"), make_packet("camera-02")])
    assert len(results) == 2
    assert detector.kwargs["quantize"] == 16
    assert "half" not in detector.kwargs
    assert results[0].data["plates"][0]["plate"] == "12ب34567"
    assert results[1].data["plate_count"] == 1
