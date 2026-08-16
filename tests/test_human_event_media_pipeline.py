from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import hashlib
import threading
import numpy as np

from app.core.recording_segment_store import (
    IncompleteCoverageError, RecordingSegment, RecordingSegmentStore,
    match_covering_segments,
)
from app.core.human_event_media_worker import HumanEventMediaWorker
from app.core.human_event_audit_store import HumanEventAuditStore
from app.core.durable_event_outbox import DurableHumanEventOutbox
from app.core.detection_event_schemas import HumanDetectionEvent
from app.core.recording_segment_dispatcher import RecordingSegmentDispatcher
from app.database import Database
from app.core.live_branch import GpuLiveBranchManager
from app.api.detection_logs import _serve_media_file
from app.core.human_detection_event_observer import HumanDetectionEventObserver, _Track
from app.core.types import FramePacket, TaskName, TaskResult


NOW = datetime(2026, 8, 15, tzinfo=timezone.utc)


def human_event(event_id: str = "e") -> HumanDetectionEvent:
    return HumanDetectionEvent(
        event_id=event_id, camera_id="rtsp://camera/1", tracking_session_id="s", track_id=1,
        name="Unknown", recognition_status="unknown", recognition_confidence=0.0,
        first_seen_at_utc=NOW, last_seen_at_utc=NOW, best_frame_at_utc=NOW,
        best_frame_index=1, bounding_box=(0., 0., 2., 2.), frame_width=2, frame_height=2,
        snapshot_quality=.5, clip_start_at_utc=NOW - timedelta(seconds=5),
        clip_end_at_utc=NOW + timedelta(seconds=5), counts_for_attendance=True,
        created_at_utc=NOW,
    )


def segment(name: str, start: int, end: int) -> RecordingSegment:
    return RecordingSegment(name, "cam", "recordings", f"continuous/{name}.mp4",
                            NOW + timedelta(seconds=start), NOW + timedelta(seconds=end),
                            1920, 1080, 25.0, "a" * 64)


def test_half_open_segment_matching_and_full_coverage() -> None:
    values = [segment("a", 0, 10), segment("boundary", 10, 20), segment("outside", 20, 30)]
    assert [item.segment_id for item in match_covering_segments(
        values, NOW + timedelta(seconds=5), NOW + timedelta(seconds=20), max_segments=3
    )] == ["a", "boundary"]
    with pytest.raises(IncompleteCoverageError):
        match_covering_segments([values[0], values[2]], NOW + timedelta(seconds=5),
                                NOW + timedelta(seconds=25), max_segments=3)


def test_segment_matching_tolerates_splitmux_clock_jitter() -> None:
    values = [
        segment("first", 0, 10),
        RecordingSegment("second", "cam", "recordings", "continuous/second.mp4",
                         NOW + timedelta(seconds=10, microseconds=100),
                         NOW + timedelta(seconds=20), 1920, 1080, 25.0, "b" * 64),
    ]

    assert [item.segment_id for item in match_covering_segments(
        values, NOW + timedelta(seconds=5), NOW + timedelta(seconds=15), max_segments=3
    )] == ["first", "second"]


@pytest.mark.postgresql
def test_recording_segment_upsert_is_idempotent_with_postgresql_timestamps(
    postgres_database,
) -> None:
    store = RecordingSegmentStore(postgres_database)
    value = segment("idempotent", 0, 10)

    assert store.upsert(value) == value
    assert store.upsert(value) == value


class Redis:
    def __init__(self): self.calls = []
    def xgroup_create(self, *args, **kwargs): self.calls.append(("group", args, kwargs))
    def xadd(self, *args, **kwargs): self.calls.append(("dead", args, kwargs)); return "2-0"
    def xack(self, *args): self.calls.append(("ack", args, {}))
    def xpending_range(self, *args): return [{"times_delivered": 1}]


def test_group_starts_at_dollar_and_retry_does_not_ack() -> None:
    redis = Redis()
    worker = HumanEventMediaWorker(redis, "human", SimpleNamespace(), SimpleNamespace(), lambda *_: None)
    # Avoid starting the thread while verifying the creation contract.
    worker._thread = SimpleNamespace()
    worker.start()
    assert redis.calls == []
    worker._thread = None
    original = __import__("threading").Thread
    class Thread:
        def __init__(self, **kwargs): pass
        def start(self): pass
    import app.core.human_event_media_worker as module
    module.threading.Thread = Thread
    try: worker.start()
    finally: module.threading.Thread = original
    assert redis.calls[0][0] == "group" and redis.calls[0][2]["id"] == "$"
    # `$` is intentional: this approved group consumes only entries appended
    # after first group creation; historical entries are not backfilled.
    worker._handle("1-0", {"event": "not-json"})
    # Invalid envelopes are terminal: dead-letter must precede ACK.
    assert [call[0] for call in redis.calls[-2:]] == ["dead", "ack"]


def test_human_media_uses_configured_temporary_root(tmp_path: Path) -> None:
    root = tmp_path / "saved_media" / "temporary_minIO" / "human_track"
    worker = HumanEventMediaWorker(
        Redis(), "human", SimpleNamespace(), SimpleNamespace(), lambda *_: None,
        temp_root=root,
    )

    assert worker.temp_root == root


@pytest.mark.parametrize("upload_fails", [False, True])
def test_human_media_files_are_removed_only_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, upload_fails: bool,
) -> None:
    event = human_event()

    class Store:
        def match(self, *args, **kwargs):
            return [segment("source", -5, 5)]

    class Storage:
        settings = SimpleNamespace(bucket_name="recordings")

        def __init__(self):
            self.uploaded = []

        def stat(self, _key):
            return SimpleNamespace(size=4)

        def download_object(self, _key, destination, **_kwargs):
            destination.write_bytes(b"data")
            return destination

        def upload_object(self, _key, _path, _content_type):
            self.uploaded.append(_key)
            if upload_fails:
                raise RuntimeError("MinIO unavailable")

    def extract(_downloads, _starts, _clip_start, _clip_end, _best, clip, snapshot, **_kwargs):
        clip.write_bytes(b"clip")
        snapshot.write_bytes(b"snapshot")
        return SimpleNamespace(frames=3)

    import app.core.human_event_media_worker as module
    monkeypatch.setattr(module, "extract_human_media", extract)
    redis = Redis()
    root = tmp_path / "saved_media" / "temporary_minIO" / "human_track"
    storage = Storage()
    local_root = tmp_path / "saved_media" / "human_track"
    worker = HumanEventMediaWorker(
        redis, "human", Store(), storage, lambda *_: None,
        temp_root=root, local_root=local_root,
    )

    worker._handle("1-0", {"event": event.model_dump_json()})

    camera_root = root / "rtsp___camera_1"
    if upload_fails:
        assert (camera_root / "e.mp4").is_file()
        assert (camera_root / "e.jpg").is_file()
        assert not any(call[0] == "ack" for call in redis.calls)
    else:
        assert not camera_root.exists()
        assert storage.uploaded == [
            "human_track/rtsp://camera/1/2026/08/15/e/clip.mp4",
            "human_track/rtsp://camera/1/2026/08/15/e/snapshot.jpg",
        ]
        assert (local_root / "rtsp___camera_1/2026/08/15/e/clip.mp4").read_bytes() == b"clip"
        assert (local_root / "rtsp___camera_1/2026/08/15/e/snapshot.jpg").read_bytes() == b"snapshot"
        assert any(call[0] == "ack" for call in redis.calls)


def test_human_event_audit_retains_event_and_merge_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = human_event("audit-event")
    segments = [segment("first", -5, 0), segment("second", 0, 5)]

    class Store:
        matched_camera_id = None

        def match(self, *args, **kwargs):
            self.matched_camera_id = args[0]
            return segments

    class Storage:
        settings = SimpleNamespace(bucket_name="recordings")

        def __init__(self):
            self.uploaded = []

        def stat(self, _key):
            return SimpleNamespace(size=4)

        def download_object(self, _key, destination, **_kwargs):
            destination.write_bytes(b"data")
            return destination

        def upload_object(self, _key, _path, _content_type):
            self.uploaded.append(_key)
            return None

    def extract(_downloads, _starts, _clip_start, _clip_end, _best, clip, snapshot, **_kwargs):
        clip.write_bytes(b"clip")
        snapshot.write_bytes(b"snapshot")
        return SimpleNamespace(frames=27)

    import json
    import app.core.human_event_media_worker as module
    monkeypatch.setattr(module, "extract_human_media", extract)
    audit_root = tmp_path / "saved_media/temporary_minIO/human_track"
    resolver = lambda _source_id: 1
    class CapturingAudit(HumanEventAuditStore):
        completed_status = None

        def remove(self, value):
            import json
            self.completed_status = json.loads(self.paths(value)[1].read_text("utf-8"))
            super().remove(value)

    audit = CapturingAudit(audit_root, camera_id_resolver=resolver)
    redis = Redis()
    store = Store()
    storage = Storage()
    finalized = []
    worker = HumanEventMediaWorker(
        redis, "human", store, storage, lambda value, *_: finalized.append(value),
        temp_root=audit_root, local_root=tmp_path / "saved_media/human_track",
        audit_store=audit, storage_camera_id_resolver=resolver,
    )

    worker._handle("1-0", {"event": event.model_dump_json()})

    event_path, status_path = audit.paths(event)
    assert event_path.parent.name == "1"
    assert not event_path.exists() and not status_path.exists()
    status = audit.completed_status
    assert status["state"] == "completed"
    assert status["segments_found"] == 2
    assert status["segments_used"] == ["first", "second"]
    assert status["concatenation_required"] is True
    assert status["concatenation_succeeded"] is True
    assert status["clip_created"] is True and status["clip_frames"] == 27
    assert status["minio_uploaded"] is True
    assert status["database_finalized"] is True
    assert status["local_archived"] is True
    assert status["completed"] is True and status["ack_pending"] is True
    assert status["storage_camera_id"] == "1"
    assert store.matched_camera_id == event.camera_id
    assert finalized == [event]
    assert storage.uploaded == [
        "human_track/1/2026/08/15/audit-event/clip.mp4",
        "human_track/1/2026/08/15/audit-event/snapshot.jpg",
    ]
    assert (tmp_path / "saved_media/human_track/1/2026/08/15/audit-event/clip.mp4").is_file()


def test_exhausted_human_media_failure_moves_audit_json_to_failed(
    tmp_path: Path,
) -> None:
    event = human_event("failed-event")

    class Store:
        def match(self, *args, **kwargs):
            raise RuntimeError("segment service unavailable")

    resolver = lambda _source_id: 2
    audit_root = tmp_path / "saved_media/temporary_minIO/human_track"
    audit = HumanEventAuditStore(audit_root, camera_id_resolver=resolver)
    audit.persist_event(event)
    worker = HumanEventMediaWorker(
        Redis(), "human", Store(), SimpleNamespace(), lambda *_: None,
        temp_root=audit_root, audit_store=audit,
        storage_camera_id_resolver=resolver, max_attempts=1,
    )

    worker._handle("1-0", {"event": event.model_dump_json()})

    active_event, active_status = audit.paths(event)
    failed_root = audit_root / "failed" / "2"
    assert not active_event.exists() and not active_status.exists()
    assert (failed_root / active_event.name).is_file()
    assert (failed_root / active_status.name).is_file()


def test_outbox_retains_event_audit_before_accepting_publish(tmp_path: Path) -> None:
    event = human_event("retained-event")
    audit = HumanEventAuditStore(tmp_path / "audit")
    outbox = DurableHumanEventOutbox(
        Database.__new__(Database), Redis(), "human", tmp_path / "outbox",
        audit_store=audit,
    )

    assert outbox.publish(event) is True
    event_path, status_path = audit.paths(event)
    assert event_path.is_file() and status_path.is_file()


def test_audit_paths_do_not_collide_after_sanitization(tmp_path: Path) -> None:
    audit = HumanEventAuditStore(tmp_path)
    first = human_event("event/a").model_copy(update={"camera_id": "camera/a"})
    second = human_event("event:a").model_copy(update={"camera_id": "camera:a"})

    assert audit.paths(first) != audit.paths(second)


def test_retryable_coverage_error_is_not_acked() -> None:
    class Store:
        def match(self, *args, **kwargs): raise IncompleteCoverageError("open segment")
    redis = Redis()
    worker = HumanEventMediaWorker(redis, "human", Store(), SimpleNamespace(), lambda *_: None)
    event = {
        "schema_version": 1, "event_id": "e", "event_name": "human", "event_type": "track_ended",
        "camera_id": "rtsp://camera/1", "room_id": None, "tracking_session_id": "s", "track_id": 1,
        "personnel_id": None, "ref_img_id": None, "name": "Unknown", "recognition_status": "unknown",
        "recognition_confidence": 0.0, "first_seen_at_utc": NOW.isoformat(), "last_seen_at_utc": NOW.isoformat(),
        "best_frame_at_utc": NOW.isoformat(), "best_frame_index": 1, "bounding_box": [0., 0., 2., 2.],
        "bounding_box_format": "xyxy", "frame_width": 2, "frame_height": 2, "snapshot_quality": 0.5,
        "clip_start_at_utc": (NOW - timedelta(seconds=5)).isoformat(),
        "clip_end_at_utc": (NOW + timedelta(seconds=5)).isoformat(), "counts_for_attendance": True,
        "created_at_utc": NOW.isoformat(),
    }
    import json
    worker._handle("1-0", {"event": json.dumps(event)})
    assert not any(call[0] == "ack" for call in redis.calls)


def test_callback_fsync_spool_survives_restart_and_reports_full(tmp_path) -> None:
    event = human_event("durable-event")
    database = Database.__new__(Database)
    first = DurableHumanEventOutbox(database, Redis(), "human", tmp_path, max_spool_files=1)
    assert first.publish(event) is True
    safe = hashlib.sha256(b"durable-event").hexdigest()
    assert (tmp_path / f"{safe}.json").is_file()
    restarted = DurableHumanEventOutbox(database, Redis(), "human", tmp_path, max_spool_files=1)
    assert restarted.publish(event) is True  # same durable event is idempotent
    assert restarted.publish(human_event("other")) is False
    assert restarted.status()["spool_failed"] == 1


def test_dead_letter_failure_never_acks_terminal_message() -> None:
    class BrokenDeadLetter(Redis):
        def xadd(self, *args, **kwargs):
            self.calls.append(("dead_failed", args, kwargs)); raise RuntimeError("redis down")
    redis = BrokenDeadLetter()
    worker = HumanEventMediaWorker(redis, "human", SimpleNamespace(), SimpleNamespace(), lambda *_: None)
    worker._handle("1-0", {"event": "invalid"})
    assert not any(call[0] == "ack" for call in redis.calls)


def test_manifest_recovery_marks_only_after_confirmed_xadd() -> None:
    item = segment("manifest", 0, 10)
    class Store:
        def __init__(self): self.marked = []
        def unpublished(self): return (item,)
        def mark_published(self, value): self.marked.append(value)
    store = Store(); redis = Redis()
    dispatcher = RecordingSegmentDispatcher(store, redis, "segments")
    assert dispatcher.dispatch_once() == 1
    assert [call[0] for call in redis.calls] == ["dead"]
    assert store.marked == ["manifest"]
    class Down(Redis):
        def xadd(self, *args, **kwargs): raise RuntimeError("down")
    store = Store(); dispatcher = RecordingSegmentDispatcher(store, Down(), "segments")
    assert dispatcher.dispatch_once() == 0 and store.marked == []


def test_crash_after_xadd_before_mark_republishes_deterministic_manifest() -> None:
    item = segment("same-id", 0, 10)
    class Store:
        def __init__(self): self.calls = 0; self.done = False
        def unpublished(self): return () if self.done else (item,)
        def mark_published(self, value):
            self.calls += 1
            if self.calls == 1: raise RuntimeError("crash before commit")
            self.done = True
    store = Store(); redis = Redis(); dispatcher = RecordingSegmentDispatcher(store, redis, "segments")
    assert dispatcher.dispatch_once() == 0
    assert dispatcher.dispatch_once() == 1
    payloads = [call[1][1]["event"] for call in redis.calls]
    assert len(payloads) == 2 and payloads[0] == payloads[1]


def test_segment_uses_canonical_source_id_without_changing_object_name(tmp_path) -> None:
    upload = GpuLiveBranchManager._recording_upload(
        tmp_path / "camera-12-part.mp4", "12", NOW, NOW + timedelta(seconds=10),
        "rtsp://camera.example/live",
    )
    assert upload.source_id == "rtsp://camera.example/live"
    assert upload.object_name.startswith("continuous/12/")


def test_proxy_temp_is_cleaned_on_invalid_range(tmp_path) -> None:
    path = tmp_path / "proxy.mp4"; path.write_bytes(b"1234")
    response = _serve_media_file(path, SimpleNamespace(headers={"range": "invalid"}), False, True)
    assert response.status_code == 416 and not path.exists()


def test_orphan_after_fsync_before_rename_is_promoted_on_restart(tmp_path) -> None:
    database = Database.__new__(Database)
    def crash(_path): raise KeyboardInterrupt("simulated process death")
    outbox = DurableHumanEventOutbox(database, Redis(), "human", tmp_path, after_fsync=crash)
    with pytest.raises(KeyboardInterrupt): outbox.publish(human_event("crash-window"))
    assert list(tmp_path.glob("*.tmp"))
    DurableHumanEventOutbox(database, Redis(), "human", tmp_path)
    digest = hashlib.sha256(b"crash-window").hexdigest()
    assert (tmp_path / f"{digest}.json").is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_completed_event_retries_without_another_frame() -> None:
    class Publisher:
        def __init__(self): self.calls = 0
        def publish(self, event): self.calls += 1; return self.calls >= 2
    publisher = Publisher(); observer = HumanDetectionEventObserver(lambda: publisher, retry_seconds=.01)
    key = ("s", "cam", 1)
    with observer._lock:
        observer._completions[key] = human_event("retry")
        observer._counts["undurable"] = 1
    observer._wake.set()
    assert observer.close(1.0)
    assert publisher.calls >= 2 and observer.status()["undurable"] == 0


def test_immediate_shutdown_reports_undurable_completion() -> None:
    publisher = SimpleNamespace(publish=lambda event: False)
    observer = HumanDetectionEventObserver(lambda: publisher, retry_seconds=.01)
    with observer._lock:
        observer._completions[("s", "cam", 1)] = human_event("blocked")
        observer._counts["undurable"] = 1
    observer._wake.set()
    assert observer.close(.02) is False
    assert observer.status()["undurable"] == 1


def test_completion_capacity_is_explicit_and_does_not_evict_undurable() -> None:
    observer = HumanDetectionEventObserver(lambda: None, completion_capacity=1, retry_seconds=.1)
    first, second = ("s", "cam", 1), ("s", "cam", 2)
    with observer._lock:
        observer._completions[first] = human_event("first")
        observer._active[second] = _Track(NOW, NOW, NOW, 1, (0., 0., 2., 2.), 2, 2, .5, False)
        observer._finalize(second, object())
        assert first in observer._completions and second in observer._waiting_completions
        assert observer._counts["handoff_full"] == 1
    assert observer.close(.01) is False


def test_capacity_one_a_then_b_automatically_promotes_without_new_callback() -> None:
    class Publisher:
        def __init__(self): self.track_ids = []
        def publish(self, event): self.track_ids.append(event.track_id); return True
    publisher = Publisher()
    observer = HumanDetectionEventObserver(lambda: publisher, completion_capacity=1, retry_seconds=.01)
    a, b = ("s", "cam", 1), ("s", "cam", 2)
    with observer._lock:
        for key in (a, b):
            observer._active[key] = _Track(NOW, NOW, NOW, 1, (0., 0., 2., 2.), 2, 2, .5, False)
            observer._pending[key] = None
            observer._finalize(key, publisher)
        assert list(observer._completions) == [a]
        assert list(observer._waiting_completions) == [b]
        assert observer._counts["undurable"] == 2
    assert observer.close(1.0) is True
    assert publisher.track_ids == [1, 2]
    assert observer.status()["undurable"] == 0


def test_shutdown_fails_when_primary_or_waiting_completion_remains() -> None:
    observer = HumanDetectionEventObserver(
        lambda: SimpleNamespace(publish=lambda event: False), completion_capacity=1, retry_seconds=.01)
    with observer._lock:
        observer._completions[("s", "cam", 1)] = human_event("a")
        observer._waiting_completions[("s", "cam", 2)] = human_event("b")
        observer._counts["undurable"] = 2
    observer._wake.set()
    assert observer.close(.02) is False
    assert observer.status()["undurable"] == 2


def test_capacity_one_a_primary_b_waiting_c_drop_new_then_drain() -> None:
    allow = threading.Event()
    class Publisher:
        def __init__(self): self.track_ids = []
        def publish(self, event):
            if not allow.is_set(): return False
            self.track_ids.append(event.track_id); return True
    publisher = Publisher()
    observer = HumanDetectionEventObserver(
        lambda: publisher, active_capacity=2, completion_capacity=1, retry_seconds=.01)
    a, b = ("s", "cam", 1), ("s", "cam", 2)
    with observer._lock:
        for key in (a, b):
            observer._active[key] = _Track(NOW, NOW, NOW, 1, (0., 0., 2., 2.), 2, 2, .5, False)
            observer._pending[key] = None
            observer._finalize(key, publisher)
    packet = FramePacket("cam", np.zeros((2, 2, 3), dtype=np.uint8), 1, 1, 0.0, NOW.isoformat())
    result = TaskResult(TaskName.FACE_RECOGNITION, "cam", 1, 1, "", "", 0, data={
        "tracking_session_id": "s", "source_frame_size": {"width": 2, "height": 2},
        "humans": [{"track_id": 3, "source_bbox": [0, 0, 2, 2], "confidence": .5}],
        "disappeared_humans": [{"track_id": 3}],
    })
    observer.observe(packet, result, room_ids_by_track={}, counts_for_attendance=True)
    status = observer.status()
    assert status["admission_dropped"] == 1 and status["rejected_cached"] == 1
    assert ("s", "cam", 3) not in observer._active
    assert ("s", "cam", 3) not in observer._pending
    assert ("s", "cam", 3) not in observer._completions
    assert ("s", "cam", 3) not in observer._waiting_completions
    # A repeated packet for the rejected observation cannot partially re-enter.
    observer.observe(packet, result, room_ids_by_track={}, counts_for_attendance=True)
    assert observer.status()["admission_dropped"] == 1
    allow.set(); observer._wake.set()
    assert observer.close(1.0) is True
    assert publisher.track_ids == [1, 2]
    assert observer.status()["admission_dropped"] == 1
