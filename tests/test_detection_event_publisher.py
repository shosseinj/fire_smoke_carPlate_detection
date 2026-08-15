import threading
import pytest

from app.core.detection_event_publisher import DetectionEventPublisher
from app.core.detection_event_schemas import (FireSmokeDetectionEvent, HumanDetectionEvent,
    PlateDetectionEvent, RecordingSegmentEvent)

STREAMS = {"human": "human", "fire_smoke": "fire", "plate": "plate", "recording_segment": "segments"}


def _event():
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    from datetime import timedelta
    return HumanDetectionEvent(event_id="e", camera_id="c", tracking_session_id="s", track_id=1,
        name="Unknown", recognition_status="unknown", recognition_confidence=0.5, first_seen_at_utc=now,
        last_seen_at_utc=now, best_frame_at_utc=now, best_frame_index=1, bounding_box=(0.0, 0.0, 5.0, 5.0),
        frame_width=10, frame_height=10, snapshot_quality=0.5, clip_start_at_utc=now,
        clip_end_at_utc=now + timedelta(seconds=1), counts_for_attendance=False, created_at_utc=now)


def test_publish_is_background_and_payload_is_exact() -> None:
    class Redis:
        def __init__(self): self.calls = []; self.called = threading.Event()
        def xadd(self, name, fields): self.calls.append((name, fields)); self.called.set()
    redis = Redis()
    publisher = DetectionEventPublisher(redis, STREAMS)
    event = _event()
    assert publisher.publish(event)
    assert redis.called.wait(1)
    publisher.close()
    assert redis.calls == [("human", {"event": event.model_dump_json()})]
    assert publisher.status()["published"] == 1


def test_failures_are_retried_without_killing_worker() -> None:
    class Redis:
        def __init__(self): self.count = 0
        def xadd(self, name, fields):
            self.count += 1
            if self.count < 3: raise RuntimeError("down")
    redis = Redis()
    publisher = DetectionEventPublisher(redis, STREAMS, retry_backoff_seconds=0)
    publisher.publish(_event())
    publisher._queue.join()
    status = publisher.status()
    assert status["published"] == 1
    assert status["retried"] == 2
    publisher.close(); publisher.close()


def test_constructor_and_unsupported_event_validation() -> None:
    with pytest.raises(ValueError):
        DetectionEventPublisher(object(), {"human": "h"})
    publisher = DetectionEventPublisher(object(), STREAMS)
    with pytest.raises(TypeError):
        publisher.publish(object())
    publisher.close()


def test_blocked_xadd_does_not_block_caller_and_close_is_bounded() -> None:
    entered, release = threading.Event(), threading.Event()
    class Redis:
        def xadd(self, name, fields): entered.set(); release.wait(2)
    publisher = DetectionEventPublisher(Redis(), STREAMS, close_timeout_seconds=0.01)
    caller_thread = threading.get_ident()
    assert publisher.publish(_event())
    assert entered.wait(1)
    assert publisher.worker_alive and publisher._worker.ident != caller_thread
    assert publisher.close() is False
    assert publisher.status()["error"] == "shutdown_timed_out"
    release.set()
    publisher._worker.join(1)
    assert publisher.close() is True
    assert publisher.status()["closed"] is True


def test_all_four_routes_use_exact_event_json() -> None:
    class Redis:
        def __init__(self): self.calls = []
        def xadd(self, name, fields): self.calls.append((name, fields))
    events = [
        HumanDetectionEvent.model_construct(event_name="human", event_type="track_ended", event_id="h"),
        FireSmokeDetectionEvent.model_construct(event_name="fire_smoke", event_type="incident_ended", event_id="f"),
        PlateDetectionEvent.model_construct(event_name="plate", event_type="plate_detected", event_id="p"),
        RecordingSegmentEvent.model_construct(segment_id="s"),
    ]
    redis = Redis(); publisher = DetectionEventPublisher(redis, STREAMS)
    for event in events: assert publisher.publish(event)
    publisher._queue.join(); publisher.close()
    assert redis.calls == [(stream, {"event": event.model_dump_json()}) for stream, event in zip(STREAMS.values(), events)]


def test_drop_newest_and_discard_pending_are_accounted() -> None:
    entered, release = threading.Event(), threading.Event()
    class Redis:
        def xadd(self, name, fields): entered.set(); release.wait(1)
    publisher = DetectionEventPublisher(Redis(), STREAMS, queue_capacity=1, close_timeout_seconds=0.01)
    assert publisher.publish(_event()); assert entered.wait(1)
    assert publisher.publish(_event())
    assert publisher.publish(_event()) is False
    assert publisher.close() is False
    status = publisher.status()
    assert status["queued"] == 2 and status["dropped"] == 2
    release.set(); publisher._worker.join(1); publisher.close()


def test_retry_exhaustion_counts_failure_and_worker_continues() -> None:
    class Redis:
        def __init__(self): self.calls = 0
        def xadd(self, name, fields): self.calls += 1; raise RuntimeError("down")
    redis = Redis(); publisher = DetectionEventPublisher(redis, STREAMS, max_retries=1, retry_backoff_seconds=0)
    publisher.publish(_event()); publisher.publish(_event()); publisher._queue.join()
    status = publisher.status()
    assert status["failed"] == 2 and status["retried"] == 2 and redis.calls == 4
    assert publisher.worker_alive
    publisher.close()


def test_serialization_failure_does_not_kill_worker(monkeypatch) -> None:
    class Redis:
        def __init__(self): self.calls = 0
        def xadd(self, name, fields): self.calls += 1
    original = HumanDetectionEvent.model_dump_json
    calls = {"count": 0}
    def flaky(self, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1: raise ValueError("bad serialization")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(HumanDetectionEvent, "model_dump_json", flaky)
    redis = Redis(); publisher = DetectionEventPublisher(redis, STREAMS)
    publisher.publish(_event()); publisher.publish(_event()); publisher._queue.join()
    status = publisher.status()
    assert status["failed"] == 1 and status["published"] == 1 and redis.calls == 1
    assert set(status) == {"enabled", "running", "closed", "published", "queued", "dropped", "retried", "failed", "error"}
    publisher.close()
