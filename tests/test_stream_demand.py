from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

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

    assert controller.status() == {
        "video_subscribers": 0,
        "ai_subscribers": 0,
        "video_required": False,
        "ai_required": False,
    }


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
