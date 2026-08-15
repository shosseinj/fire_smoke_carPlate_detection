from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.core.human_detection_event_observer import HumanDetectionEventObserver
from app.core.types import FramePacket, TaskName, TaskResult


class Publisher:
    def __init__(self, accepted=True): self.accepted, self.events = accepted, []
    def publish(self, event): self.events.append(event); return self.accepted


def packet(index=1, at="2026-01-01T00:00:01+00:00"):
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    return FramePacket("cam", frame, 1, index, 0.0, at, metadata={"source_frame": np.zeros((20, 30, 3), dtype=np.uint8)})


def result(*, track=1, humans=True, disappeared=False, session="session", camera="cam", index=1,
           ref=None, person="Unknown", face=False, quality=.2, bbox=(1, 2, 10, 12), width=30, height=20):
    human = {"track_id": track, "source_bbox": bbox, "confidence": quality, "face_visible": face,
             "best_face_quality": quality, "identity_stable": person != "Unknown", "person": person,
             "ref_img_id": ref, "recognition_score": .8}
    return TaskResult(TaskName.FACE_RECOGNITION, camera, 1, index, "", "", 0, data={
        "tracking_session_id": session, "source_frame_size": {"width": width, "height": height},
        "humans": [human] if humans else [], "disappeared_humans": [{"track_id": track}] if disappeared else []})


def observe(observer, publisher, res, pkt=None, rooms=None, attendance=False):
    observer.observe(pkt or packet(res.frame_index), res, room_ids_by_track=rooms or {}, counts_for_attendance=attendance)


def test_recognized_unknown_and_unresolved_identity_policy():
    pub = Publisher(); observer = HumanDetectionEventObserver(lambda: pub)
    observe(observer, pub, result(track=1, person="Alice", ref="personnel_12"))
    observe(observer, pub, result(track=1, humans=False, disappeared=True))
    observe(observer, pub, result(track=2, person="Numeric", ref="123"))
    observe(observer, pub, result(track=2, humans=False, disappeared=True))
    observe(observer, pub, result(track=3))
    observe(observer, pub, result(track=3, humans=False, disappeared=True))
    assert [(e.name, e.personnel_id, e.ref_img_id) for e in pub.events] == [
        ("Alice", 12, "personnel_12"), ("Unknown", None, None)]
    assert observer.status()["unresolved_recognized"] == 1


def test_exactly_once_deterministic_id_and_session_camera_independence():
    pub = Publisher(); observer = HumanDetectionEventObserver(lambda: pub)
    for session, camera in (("s1", "c1"), ("s1", "c2"), ("s2", "c1")):
        observe(observer, pub, result(session=session, camera=camera))
        terminal = result(session=session, camera=camera, humans=False, disappeared=True)
        observe(observer, pub, terminal); observe(observer, pub, terminal)
    assert len(pub.events) == 3 and len({e.event_id for e in pub.events}) == 3
    first_id = pub.events[0].event_id
    other = Publisher(); replay = HumanDetectionEventObserver(lambda: other)
    observe(replay, other, result(session="s1", camera="c1")); observe(replay, other, result(session="s1", camera="c1", humans=False, disappeared=True))
    assert other.events[0].event_id == first_id and observer.status()["duplicate"] == 3


def test_timing_best_frame_source_metadata_room_and_no_frame_retention():
    pub = Publisher(); observer = HumanDetectionEventObserver(lambda: pub)
    observe(observer, pub, result(index=2, quality=.9, bbox=(2, 3, 20, 15)), packet(2, "2026-01-01T00:00:02+00:00"))
    observe(observer, pub, result(index=3, face=True, quality=.4, bbox=(3, 4, 25, 18), width=40, height=25),
            packet(3, "2026-01-01T00:00:03+00:00"), {1: 7}, True)
    # Equal face quality does not replace the earlier selected frame.
    observe(observer, pub, result(index=4, face=True, quality=.4, bbox=(4, 5, 29, 19)),
            packet(4, "2026-01-01T00:00:04+00:00"), attendance=True)
    assert not any(isinstance(value, np.ndarray) for state in observer._active.values() for value in state.__slots__ if isinstance(getattr(state, value), np.ndarray))
    observe(observer, pub, result(humans=False, disappeared=True, index=9), packet(9, "2026-01-01T00:01:00+00:00"))
    event = pub.events[0]
    assert event.first_seen_at_utc.second == 2 and event.last_seen_at_utc.second == 4
    assert event.best_frame_index == 3 and event.bounding_box == (3, 4, 25, 18)
    assert (event.frame_width, event.frame_height, event.room_id, event.counts_for_attendance) == (40, 25, 7, True)
    assert event.clip_start_at_utc == event.first_seen_at_utc and event.clip_end_at_utc == event.last_seen_at_utc
    assert event.created_at_utc >= event.last_seen_at_utc


def test_out_of_order_completion_reconciles_and_old_packets_are_ignored():
    pub = Publisher(); observer = HumanDetectionEventObserver(lambda: pub)
    observe(observer, pub, result(humans=False, disappeared=True))
    assert observer.status()["pending"] == 1 and not pub.events
    observe(observer, pub, result())
    observe(observer, pub, result(index=2))
    assert len(pub.events) == 1 and observer.status()["active"] == observer.status()["pending"] == 0


def test_rejection_is_terminal_but_malformed_input_recovers_and_isolated():
    pub = Publisher(False); observer = HumanDetectionEventObserver(lambda: pub)
    bad = result(track=1, bbox=(5, 5, 5, 8)); good = result(track=2)
    bad.data["humans"].append(good.data["humans"][0])
    observe(observer, pub, bad)
    observe(observer, pub, result(track=2, humans=False, disappeared=True))
    observe(observer, pub, result(track=2, humans=False, disappeared=True))
    assert len(pub.events) == 1 and observer.status()["rejected"] == 1
    assert observer.status()["malformed"] >= 1 and observer.status()["duplicate"] == 1


def test_active_completed_pending_bounds_and_basic_thread_safety():
    pub = Publisher(); observer = HumanDetectionEventObserver(lambda: pub, active_capacity=2, completed_capacity=2, pending_capacity=2)
    for track in range(4): observe(observer, pub, result(track=track))
    for track in range(10, 14): observe(observer, pub, result(track=track, humans=False, disappeared=True))
    assert observer.status()["active"] == 2 and observer.status()["pending"] == 2
    assert observer.status()["active_evicted"] == 2 and observer.status()["pending_evicted"] == 2
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda track: observe(observer, pub, result(track=track), rooms={track: 1}), range(20, 30)))
    for track in (28, 29): observe(observer, pub, result(track=track, humans=False, disappeared=True))
    assert observer.status()["active"] <= 2 and observer.status()["completed"] <= 2


def test_disabled_publisher_accumulates_no_state():
    observer = HumanDetectionEventObserver(lambda: None)
    observe(observer, None, result())
    assert observer.status()["active"] == observer.status()["pending"] == 0


def test_pending_malformed_same_key_then_corrected_metadata_recovers_once():
    pub = Publisher(); observer = HumanDetectionEventObserver(lambda: pub)
    observe(observer, pub, result(humans=False, disappeared=True))
    # Invalid admitted room makes event schema construction fail without tombstoning.
    observe(observer, pub, result(), rooms={1: 0})
    assert not pub.events and observer.status()["pending"] == 1
    observe(observer, pub, result(index=2), rooms={1: 9})
    observe(observer, pub, result(humans=False, disappeared=True))
    assert len(pub.events) == 1 and pub.events[0].room_id == 9
    assert observer.status()["completed"] == 1
