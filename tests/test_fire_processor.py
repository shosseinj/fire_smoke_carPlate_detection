from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from app.core.types import FramePacket
from app.fire_core.severity import HazardSeverity
from app.processors.fire_smoke import FireSmokeProcessor, FireSmokeSettings


class FakeBoxes:
    def __init__(self, rows):
        self.data = np.asarray(rows, dtype=np.float32)

    def __len__(self):
        return len(self.data)


class FakeResult:
    def __init__(self, confidence: float = 0.95):
        self.boxes = FakeBoxes([[1, 1, 12, 12, confidence, 0]])


class FakeModel:
    def __init__(self, confidence: float = 0.95):
        self.kwargs = None
        self.confidence = confidence

    def predict(self, source, **kwargs):
        self.kwargs = {"source": source, **kwargs}
        return [FakeResult(self.confidence) for _ in source]


def packet(
    source_id: str,
    frame_index: int,
    captured_monotonic: float | None = None,
) -> FramePacket:
    return FramePacket(
        source_id=source_id,
        frame=np.zeros((20, 20, 3), dtype=np.uint8),
        round_sequence=frame_index,
        frame_index=frame_index,
        captured_monotonic=(
            time.monotonic() if captured_monotonic is None else captured_monotonic
        ),
        captured_at_utc="2026-01-01T00:00:00+00:00",
        source_time_seconds=frame_index / 25.0,
    )


def test_fire_processor_keeps_independent_dynamic_source_state(tmp_path: Path) -> None:
    model = FakeModel()
    processor = FireSmokeProcessor(
        FireSmokeSettings(
            model_path=tmp_path / "fake.pt",
            device="cpu",
            engine_fixed_batch=None,
            required_positive_detections=1,
            required_positive_ratio=0.0,
            consecutive_detections=1,
            evidence_min_track_hits=1,
            low_severity_min_count=1,
            low_severity_min_ratio=0.0,
            medium_severity_min_count=1,
            medium_severity_min_ratio=0.0,
            high_severity_min_count=1,
            high_severity_min_ratio=0.0,
            fire_low_confidence=0.1,
            fire_medium_confidence=0.1,
            fire_high_confidence=0.1,
            incident_start_severity=HazardSeverity.MEDIUM,
            alert_start_severity=HazardSeverity.HIGH,
        ),
        model=model,
    )
    results = processor.process_batch([packet("camera-01", 1), packet("camera-50", 1)])
    assert len(results) == 2
    assert all(result.data["severity"] == "high" for result in results)
    assert all(result.data["events"][0]["event_type"] == "incident_started" for result in results)
    assert processor.status()["tracked_sources"] == 2
    assert "half" not in model.kwargs


def test_fire_pt_gpu_inference_uses_quantize_instead_of_half(tmp_path: Path) -> None:
    model = FakeModel()
    processor = FireSmokeProcessor(
        FireSmokeSettings(
            model_path=tmp_path / "fake.pt",
            device="0",
            engine_fixed_batch=None,
        ),
        model=model,
    )
    processor.process_batch([packet("camera-01", 1)])
    assert model.kwargs["quantize"] == 16
    assert "half" not in model.kwargs


def test_default_three_second_count_window_maps_5_10_20_to_severity(
    tmp_path: Path,
) -> None:
    processor = FireSmokeProcessor(
        FireSmokeSettings(
            model_path=tmp_path / "fake.pt",
            device="cpu",
            engine_fixed_batch=None,
            evidence_min_track_hits=1,
        ),
        model=FakeModel(),
    )
    severities = []
    for index in range(1, 21):
        result = processor.process_batch(
            [packet("camera-window", index, 100.0 + index * 0.1)]
        )[0]
        severities.append(result.data["severity"])

    assert severities[3] == "none"
    assert severities[4] == "low"
    assert severities[9] == "medium"
    assert severities[19] == "high"
    assert result.data["severity_window_seconds"] == 3.0


def test_fire_below_configured_score_is_not_a_detection(tmp_path: Path) -> None:
    processor = FireSmokeProcessor(
        FireSmokeSettings(
            model_path=tmp_path / "fake.pt",
            device="cpu",
            engine_fixed_batch=None,
            fire_candidate_confidence=0.30,
            evidence_min_track_hits=1,
        ),
        model=FakeModel(confidence=0.29),
    )
    result = processor.process_batch([packet("camera-low-score", 1)])[0]

    assert result.data["tracks"] == []
    assert result.data["fire"]["positive_count"] == 0
    assert result.data["severity"] == "none"
