from types import SimpleNamespace

import numpy as np
import pytest

from app.runtime import Runtime, _observe_human_detection_event_once
from app.core.human_detection_event_observer import HumanDetectionEventObserver
from app.core.types import FramePacket, TaskName, TaskResult


class Closable:
    def __init__(self, name="", order=None): self.count = 0; self.name = name; self.order = order
    def close(self):
        self.count += 1
        if self.order is not None: self.order.append(self.name)


class Database:
    def __init__(self): self.count = 0
    def dispose(self): self.count += 1


def runtime_for_close(publisher, redis):
    runtime = Runtime.__new__(Runtime)
    runtime.broadcast = Closable(); runtime.personnel_zip_imports = Closable(); runtime.excel_imports = Closable()
    runtime.model_conversions = Closable(); runtime.detection_event_publisher = publisher
    runtime.detection_event_redis = redis; runtime.detection_event_error = None
    runtime._closed = False
    runtime._general_teardown_complete = False
    runtime.recording_coordinator = None; runtime.recording_redis = None; runtime.recording_error = None
    runtime.media_preview = None; runtime.live_branch = None; runtime.static_video_ingestor = None; runtime.video_ingestor = None
    runtime.static_video_lifecycle = Closable(); runtime.router = Closable(); runtime.fire_smoke_logs = Closable()
    runtime.plate_logs = Closable(); runtime.human_logs = Closable(); runtime.registry = Closable(); runtime.database = Database()
    return runtime


def test_runtime_closes_publisher_before_independent_redis_exactly_once() -> None:
    order = []
    class Publisher:
        def close(self): order.append("publisher"); return True
    redis = Closable("redis", order); runtime = runtime_for_close(Publisher(), redis)
    runtime.close()
    assert order == ["publisher", "redis"] and redis.count == 1
    assert runtime.detection_event_redis is None
    runtime.close()
    assert redis.count == 1


def test_runtime_blocked_publisher_fails_after_cleanup_then_retries_redis_ownership() -> None:
    class Publisher:
        def __init__(self): self.calls = 0
        def close(self): self.calls += 1; return self.calls > 1
    publisher = Publisher(); redis = Closable(); runtime = runtime_for_close(publisher, redis)
    with pytest.raises(RuntimeError, match="detection event publisher shutdown"):
        runtime.close()
    assert redis.count == 0 and runtime.router.count == 1 and runtime.database.count == 1
    runtime.close()
    assert publisher.calls == 2 and redis.count == 1 and runtime.detection_event_redis is None
    assert runtime.router.count == 1 and runtime.database.count == 1


def test_runtime_retries_redis_close_without_repeating_general_teardown() -> None:
    class Publisher:
        def __init__(self): self.calls = 0
        def close(self): self.calls += 1; return True
    class Redis:
        def __init__(self): self.calls = 0
        def close(self):
            self.calls += 1
            if self.calls == 1: raise RuntimeError("temporary")
    publisher, redis = Publisher(), Redis(); runtime = runtime_for_close(publisher, redis)
    with pytest.raises(RuntimeError, match="detection event publisher shutdown"):
        runtime.close()
    assert runtime.router.count == 1 and runtime.database.count == 1
    runtime.close()
    assert publisher.calls == 2 and redis.calls == 2 and runtime.detection_event_redis is None
    assert runtime.router.count == 1 and runtime.database.count == 1


def test_runtime_detection_status_has_stable_keys_for_disabled_and_init_error() -> None:
    runtime = Runtime.__new__(Runtime)
    runtime.settings = SimpleNamespace(detection_events_enabled=False, recording_enabled=False)
    runtime.detection_event_publisher = None; runtime.detection_event_error = None
    runtime.recording_coordinator = None; runtime.recording_error = None
    runtime.video_ingestor = runtime.static_video_ingestor = runtime.media_preview = runtime.live_branch = None
    runtime.router = SimpleNamespace(status=lambda: {})
    runtime.broadcast = SimpleNamespace(status=lambda: {})
    runtime.plate_logs = SimpleNamespace(count=lambda: 0, status=lambda: {})
    runtime.fire_smoke_logs = SimpleNamespace(status=lambda: {})
    runtime.human_logs = SimpleNamespace(status=lambda: {})
    runtime.personnel_store = SimpleNamespace(count=lambda: 0)
    runtime.location_store = SimpleNamespace(count_buildings=lambda: 0, count_sections=lambda: 0, count_rooms=lambda: 0)
    runtime.shift_store = SimpleNamespace(count=lambda: 0); runtime.holiday_store = SimpleNamespace(count_active=lambda: 0)
    runtime.request_store = SimpleNamespace(count=lambda: 0); runtime.detection_log_store = SimpleNamespace(count_by_status=lambda: {})
    keys = {"enabled", "running", "closed", "published", "queued", "dropped", "retried", "failed", "error"}
    assert set(runtime.status()["detection_events"]) == keys
    runtime.settings.detection_events_enabled = True; runtime.detection_event_error = "RuntimeError"
    status = runtime.status()["detection_events"]
    assert set(status) == keys and status["enabled"] is True and status["error"] == "RuntimeError"
    expected = {"enabled": True, "running": True, "closed": False, "published": 1,
                "queued": 2, "dropped": 3, "retried": 4, "failed": 5, "error": None}
    class Publisher:
        def status(self): return dict(expected)
    runtime.detection_event_publisher = Publisher(); runtime.detection_event_error = None
    assert runtime.status()["detection_events"] == expected


def test_face_event_wiring_is_once_while_legacy_dual_calls_remain_unchanged() -> None:
    class EventObserver:
        def __init__(self): self.calls = 0
        def observe(self, *args, **kwargs): self.calls += 1
    class LegacyStore:
        def __init__(self): self.calls = []
        def observe_result(self, *args, **kwargs): self.calls.append(kwargs)

    event_observer, legacy = EventObserver(), LegacyStore()
    pkt = FramePacket("cam", np.zeros((2, 2, 3), dtype=np.uint8), 1, 1, 0.0,
                      "2026-01-01T00:00:01+00:00")
    res = TaskResult(TaskName.FACE_RECOGNITION, "cam", 1, 1, "", "", 0, data={
        "tracking_session_id": "session", "source_frame_size": {"width": 2, "height": 2},
        "humans": [], "disappeared_humans": []})
    _observe_human_detection_event_once(event_observer, pkt, res, {}, False)
    legacy.observe_result(pkt, res, persist_human_log=False)
    legacy.observe_result(pkt, res)
    assert event_observer.calls == 1 and len(legacy.calls) == 2

    # The unavailable-publisher path retains no event state and does not alter legacy calls.
    disabled = HumanDetectionEventObserver(lambda: None)
    _observe_human_detection_event_once(disabled, pkt, res, {}, False)
    legacy.observe_result(pkt, res, persist_human_log=False)
    legacy.observe_result(pkt, res)
    assert disabled.status()["active"] == disabled.status()["pending"] == 0
    assert len(legacy.calls) == 4
