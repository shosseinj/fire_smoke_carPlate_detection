from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time

from app.core.stream_demand import StreamDemandController


def test_video_lease_is_idempotent_and_never_goes_negative() -> None:
    controller = StreamDemandController()

    first = controller.acquire_video()
    second = controller.acquire_video()
    assert controller.snapshot()["video_subscribers"] == 2
    assert controller.video_required()

    assert first.release() is True
    assert first.release() is False
    assert controller.release_video() is True
    assert controller.release_video() is False
    assert second.release() is True  # already removed by the raw guarded release

    assert controller.snapshot()["video_subscribers"] == 0
    assert not controller.video_required()


def test_context_manager_supports_reconnect_and_disconnect() -> None:
    controller = StreamDemandController()

    for _ in range(3):
        with controller.acquire_video():
            assert controller.video_required()
            assert controller.snapshot()["video_subscribers"] == 1
        assert not controller.video_required()
        assert controller.snapshot()["video_subscribers"] == 0


def test_ai_and_video_demand_are_independent() -> None:
    controller = StreamDemandController()

    with controller.acquire_ai():
        assert controller.ai_required()
        assert not controller.video_required()
        with controller.acquire_video():
            assert controller.ai_required()
            assert controller.video_required()

    status = controller.status()
    assert status["video_subscribers"] == 0
    assert status["global_video_subscribers"] == 0
    assert status["video_subscribers_by_source"] == {}
    assert status["ai_subscribers"] == 0
    assert status["video_required"] is False
    assert status["ai_required"] is False


def test_listener_observes_changes_and_can_be_removed() -> None:
    controller = StreamDemandController()
    snapshots: list[dict[str, int | bool]] = []
    unsubscribe = controller.add_listener(snapshots.append, notify_immediately=True)

    with controller.acquire_video():
        pass
    unsubscribe()
    with controller.acquire_ai():
        pass

    assert [item["video_subscribers"] for item in snapshots] == [0, 1, 0]


def test_accounting_is_thread_safe() -> None:
    controller = StreamDemandController()

    def connect_and_disconnect(_: int) -> None:
        lease = controller.acquire_video()
        lease.release()
        lease.release()

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(connect_and_disconnect, range(200)))

    assert controller.snapshot()["video_subscribers"] == 0
    assert not controller.video_required()


def test_source_demand_does_not_enable_unrelated_source() -> None:
    controller = StreamDemandController()

    camera_1 = controller.acquire_video("camera-1")
    camera_1_again = controller.acquire_video("camera-1")
    camera_2 = controller.acquire_video("camera-2")

    assert controller.video_required()
    assert controller.video_required("camera-1")
    assert controller.video_required("camera-2")
    assert not controller.video_required("camera-3")
    assert controller.snapshot()["video_subscribers_by_source"] == {
        "camera-1": 2,
        "camera-2": 1,
    }

    camera_1.release()
    camera_1.release()
    camera_1_again.release()
    assert not controller.video_required("camera-1")
    assert controller.video_required("camera-2")
    camera_2.release()
    assert not controller.video_required()


def test_global_wall_demand_enables_every_source() -> None:
    controller = StreamDemandController()

    with controller.acquire_video():
        assert controller.snapshot()["global_video_subscribers"] == 1
        assert controller.video_required("camera-1")
        assert controller.video_required("camera-2")

    assert not controller.video_required("camera-1")


def test_wall_and_fullscreen_demands_are_composed_and_released_independently() -> None:
    controller = StreamDemandController()
    wall = controller.acquire_wall()
    fullscreen = controller.acquire_fullscreen("camera-1")

    assert controller.wall_required("camera-1")
    assert controller.wall_required("camera-2")
    assert controller.fullscreen_required("camera-1")
    assert not controller.fullscreen_required("camera-2")
    snapshot = controller.snapshot()
    assert snapshot["wall_subscribers"] == 1
    assert snapshot["fullscreen_subscribers_by_source"] == {"camera-1": 1}

    fullscreen.release()
    assert controller.wall_required("camera-1")
    assert not controller.fullscreen_required("camera-1")
    wall.release()
    assert not controller.video_required()


def test_source_release_is_guarded_and_cannot_decrement_another_source() -> None:
    controller = StreamDemandController()
    lease = controller.acquire_video("camera-1")

    assert controller.release_video("camera-2") is False
    assert controller.snapshot()["video_subscribers_by_source"] == {"camera-1": 1}
    assert lease.release() is True
    assert lease.release() is False
    assert controller.release_video("camera-1") is False
    assert controller.snapshot()["video_subscribers"] == 0


def test_release_grace_keeps_effective_demand_without_inflating_count() -> None:
    controller = StreamDemandController(video_release_grace_seconds=0.04)
    changes: list[dict[str, object]] = []
    controller.add_listener(changes.append)

    lease = controller.acquire_video("camera-1")
    lease.release()

    status = controller.snapshot()
    assert status["video_subscribers"] == 0
    assert status["video_subscribers_by_source"] == {}
    assert status["video_required"] is True
    assert controller.video_required("camera-1")
    assert not controller.video_required("camera-2")
    assert status["video_grace_remaining_by_source"]["camera-1"] > 0

    deadline = time.monotonic() + 1.0
    while controller.video_required("camera-1") and time.monotonic() < deadline:
        time.sleep(0.005)

    assert not controller.video_required()
    while changes[-1]["video_required"] is not False and time.monotonic() < deadline:
        time.sleep(0.005)
    assert changes[-1]["video_required"] is False


def test_reconnect_during_grace_invalidates_stale_expiry_timer() -> None:
    controller = StreamDemandController(video_release_grace_seconds=0.04)
    first = controller.acquire_video("camera-1")
    first.release()
    time.sleep(0.01)

    second = controller.acquire_video("camera-1")
    time.sleep(0.05)
    assert controller.video_required("camera-1")
    assert controller.snapshot()["video_subscribers_by_source"] == {"camera-1": 1}

    second.release()
