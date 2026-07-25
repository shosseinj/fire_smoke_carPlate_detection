from __future__ import annotations

import logging
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


def test_repeated_low_confidence_fire_does_not_become_high(tmp_path: Path) -> None:
    processor = FireSmokeProcessor(
        FireSmokeSettings(
            model_path=tmp_path / "fake.pt",
            device="cpu",
            engine_fixed_batch=None,
            evidence_min_track_hits=1,
        ),
        model=FakeModel(confidence=0.40),
    )

    result = None
    for index in range(1, 21):
        result = processor.process_batch(
            [packet("camera-low-confidence", index, 100.0 + index * 0.1)]
        )[0]

    assert result is not None
    assert result.data["severity"] != "high"


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


def test_fire_uses_per_source_thresholds_and_respects_zero_confidence(
    tmp_path: Path,
) -> None:
    thresholds = {
        "camera-low": (1, 0.0, 0.0),
        "camera-high": (1, 0.9, 0.9),
    }
    processor = FireSmokeProcessor(
        FireSmokeSettings(
            model_path=tmp_path / "fake.pt",
            device="cpu",
            engine_fixed_batch=None,
            evidence_min_track_hits=1,
        ),
        model=FakeModel(confidence=0.05),
        settings_provider=thresholds.__getitem__,
    )

    low_result, high_result = processor.process_batch(
        [packet("camera-low", 1), packet("camera-high", 1)]
    )

    assert low_result.data["tracks"]
    assert low_result.data["detections"]
    assert high_result.data["tracks"] == []
    assert high_result.data["detections"] == []


def test_fire_processor_logs_threshold_filtered_candidates(
    tmp_path: Path,
    caplog,
) -> None:
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

    with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
        processor.process_batch([packet("camera-low-score", 1)])

    assert any("FIRE_DETECTION_FILTERED" in message for message in caplog.messages)


def test_fire_result_exposes_raw_detections_for_immediate_overlay(
    tmp_path: Path,
) -> None:
    processor = FireSmokeProcessor(
        FireSmokeSettings(
            model_path=tmp_path / "fake.pt",
            device="cpu",
            engine_fixed_batch=None,
            evidence_min_track_hits=99,
        ),
        model=FakeModel(confidence=0.95),
    )
    result = processor.process_batch([packet("camera-detection", 1)])[0]

    assert result.error is None
    assert result.data["detections"]
    assert result.data["detections"][0]["label"] == "fire"
    assert result.data["detections"][0]["bbox"] == [1.0, 1.0, 12.0, 12.0]


def test_fire_first_hit_is_credible_with_default_evidence_gate(tmp_path: Path) -> None:
    processor = FireSmokeProcessor(
        FireSmokeSettings(
            model_path=tmp_path / "fake.pt",
            device="cpu",
            engine_fixed_batch=None,
        ),
        model=FakeModel(confidence=0.95),
    )
    result = processor.process_batch([packet("camera-first-hit", 1)])[0]

    assert result.error is None
    assert result.data["tracks"]
    assert result.data["tracks"][0]["label"] == "fire"


def test_fire_engine_runtime_failure_uses_onnx_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    engine = tmp_path / "fire.engine"
    onnx = tmp_path / "fire.onnx"
    engine.write_bytes(b"engine")
    onnx.write_bytes(b"onnx")

    class FailingModel:
        def predict(self, *args, **kwargs):
            raise RuntimeError("incompatible TensorRT engine")

    def fake_yolo(path: str, task: str | None = None):
        return FailingModel() if Path(path).suffix == ".engine" else FakeModel()

    monkeypatch.setattr(
        "app.processors.fire_smoke.load_yolo_class",
        lambda: fake_yolo,
    )
    processor = FireSmokeProcessor(
        FireSmokeSettings(
            model_path=engine,
            device="cpu",
            engine_fixed_batch=None,
            evidence_min_track_hits=1,
        ),
        model_provider=lambda: (1, [engine, onnx]),
    )

    result = processor.process_batch([packet("camera-fallback", 1)])[0]

    assert result.error is None
    assert processor.status()["model_path"] == str(onnx)
    assert processor.status()["model_fallbacks"] == 1
