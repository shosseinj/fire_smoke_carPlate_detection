from datetime import datetime, timedelta, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from app.core.detection_event_schemas import (DetectionEvent, FireSmokeDetectionEvent,
    HumanDetectionEvent, PlateDetectionEvent, RecordingSegmentEvent)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def human(**changes):
    values = dict(event_id="e", camera_id="c", room_id=None, tracking_session_id="s", track_id=1,
        personnel_id=None, ref_img_id=None, name="Unknown", recognition_status="unknown", recognition_confidence=0.5,
        first_seen_at_utc=NOW, last_seen_at_utc=NOW + timedelta(seconds=2), best_frame_at_utc=NOW + timedelta(seconds=1),
        best_frame_index=1, bounding_box=(0.0, 0.0, 5.0, 5.0), frame_width=10, frame_height=10,
        snapshot_quality=0.8, clip_start_at_utc=NOW, clip_end_at_utc=NOW + timedelta(seconds=3),
        counts_for_attendance=False, created_at_utc=NOW + timedelta(seconds=2))
    values.update(changes)
    return HumanDetectionEvent(**values)


def test_human_round_trip_and_recognition_consistency() -> None:
    event = human()
    assert TypeAdapter(DetectionEvent).validate_json(event.model_dump_json()) == event
    assert human(personnel_id=1, ref_img_id="r", name="Ali", recognition_status="recognized")
    with pytest.raises(ValidationError): human(name="someone")
    with pytest.raises(ValidationError): human(bounding_box=(0.0, 0.0, 11.0, 5.0))
    with pytest.raises(ValidationError): human(first_seen_at_utc=datetime.now())


def test_recording_segment_round_trip_and_validation() -> None:
    event = RecordingSegmentEvent(segment_id="s", camera_id="c", bucket="b", object_key="k",
        started_at_utc=NOW, ended_at_utc=NOW + timedelta(seconds=1), frame_width=10, frame_height=10, fps=25.0)
    assert TypeAdapter(DetectionEvent).validate_json(event.model_dump_json()) == event
    assert set(event.model_dump()) == {"schema_version", "segment_id", "camera_id", "bucket", "object_key",
                                       "started_at_utc", "ended_at_utc", "frame_width", "frame_height", "fps", "status"}
    assert "event_name" not in event.model_dump() and "room_id" not in event.model_dump()
    with pytest.raises(ValidationError):
        event.model_copy(update={"ended_at_utc": NOW}).model_dump_json() and RecordingSegmentEvent(**{**event.model_dump(), "ended_at_utc": NOW})


def fire(**changes):
    values = dict(event_id="f", incident_id="i", camera_id="c", room_id=None, hazard_type="fire", severity="high",
        fire_count=1, smoke_count=0, confidence=0.9, first_seen_at_utc=NOW,
        last_seen_at_utc=NOW + timedelta(seconds=2), best_frame_at_utc=NOW + timedelta(seconds=1),
        bounding_boxes=((0.0, 0.0, 5.0, 5.0),), frame_width=10, frame_height=10,
        clip_start_at_utc=NOW, clip_end_at_utc=NOW + timedelta(seconds=3), created_at_utc=NOW + timedelta(seconds=2))
    values.update(changes); return FireSmokeDetectionEvent(**values)


def plate(**changes):
    values = dict(event_id="p", camera_id="c", room_id=None, plate_id=None, plate_number="12A", raw_plate_text="12A",
        recognition_confidence=0.8, detected_at_utc=NOW, best_frame_at_utc=NOW, frame_index=0,
        bounding_box=(0.0, 0.0, 5.0, 5.0), frame_width=10, frame_height=10,
        clip_start_at_utc=NOW, clip_end_at_utc=NOW + timedelta(seconds=1), created_at_utc=NOW)
    values.update(changes); return PlateDetectionEvent(**values)


@pytest.mark.parametrize("event,keys", [
    (human(), {"schema_version", "event_id", "event_name", "event_type", "camera_id", "room_id", "tracking_session_id",
      "track_id", "personnel_id", "ref_img_id", "name", "recognition_status", "recognition_confidence", "first_seen_at_utc",
      "last_seen_at_utc", "best_frame_at_utc", "best_frame_index", "bounding_box", "bounding_box_format", "frame_width",
      "frame_height", "snapshot_quality", "clip_start_at_utc", "clip_end_at_utc", "counts_for_attendance", "created_at_utc"}),
    (fire(), {"schema_version", "event_id", "event_name", "event_type", "incident_id", "camera_id", "room_id", "hazard_type",
      "severity", "fire_count", "smoke_count", "confidence", "first_seen_at_utc", "last_seen_at_utc", "best_frame_at_utc",
      "bounding_boxes", "bounding_box_format", "frame_width", "frame_height", "clip_start_at_utc", "clip_end_at_utc", "created_at_utc"}),
    (plate(), {"schema_version", "event_id", "event_name", "event_type", "camera_id", "room_id", "plate_id", "plate_number",
      "raw_plate_text", "recognition_confidence", "detected_at_utc", "best_frame_at_utc", "frame_index", "bounding_box",
      "bounding_box_format", "frame_width", "frame_height", "clip_start_at_utc", "clip_end_at_utc", "created_at_utc"}),
])
def test_detection_exact_keys_and_roundtrip(event, keys) -> None:
    assert set(event.model_dump()) == keys
    assert TypeAdapter(DetectionEvent).validate_json(event.model_dump_json()) == event


@pytest.mark.parametrize("factory", [fire, plate])
def test_nonhuman_rejects_ref_img_and_invalid_values(factory) -> None:
    with pytest.raises(ValidationError): factory(ref_img_id="x")
    with pytest.raises(ValidationError): factory(camera_id="  ")
    with pytest.raises(ValidationError): factory(frame_width=0)


def test_fire_and_plate_ranges_times_boxes_and_literals() -> None:
    for kwargs in ({"confidence": float("nan")}, {"confidence": 1.1}, {"bounding_boxes": ((0.0, 0.0, 11.0, 2.0),)},
                   {"severity": "critical"}, {"created_at_utc": NOW - timedelta(seconds=1)}):
        with pytest.raises(ValidationError): fire(**kwargs)
    for kwargs in ({"recognition_confidence": -0.1}, {"plate_id": 0}, {"frame_index": -1},
                   {"bounding_box": (0.0, 0.0, 11.0, 2.0)}, {"event_type": "other"},
                   {"created_at_utc": NOW - timedelta(seconds=1)}):
        with pytest.raises(ValidationError): plate(**kwargs)


def test_segment_rejects_nonhuman_ref_img_blank_nonfinite_and_extra() -> None:
    base = dict(segment_id="s", camera_id="c", bucket="b", object_key="k", started_at_utc=NOW,
                ended_at_utc=NOW + timedelta(seconds=1), frame_width=10, frame_height=10, fps=25.0)
    for changes in ({"ref_img_id": "x"}, {"bucket": "  "}, {"fps": float("inf")}, {"schema_version": 2}):
        with pytest.raises(ValidationError): RecordingSegmentEvent(**{**base, **changes})
