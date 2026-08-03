from __future__ import annotations

import pytest

from app.core.live_branch import GpuLiveBranchManager, live_stream_path


class _Caps:
    def __init__(self, text: str) -> None:
        self.text = text

    def to_string(self) -> str:
        return self.text


class _Pad:
    def __init__(self, caps: _Caps | None = None) -> None:
        self.caps = caps
        self.linked = False

    def get_current_caps(self) -> _Caps | None:
        return self.caps

    def link(self, _other: object) -> int:
        self.linked = True
        return 0

    def unlink(self, _other: object) -> None:
        self.linked = False


class _Element:
    def __init__(self, factory: str) -> None:
        self.factory = factory
        self.props: dict[str, object] = {}
        self.sink = _Pad()

    def set_property(self, name: str, value: object) -> None:
        self.props[name] = value

    def get_static_pad(self, name: str) -> _Pad | None:
        return self.sink if name == "sink" else _Pad()

    def link(self, _other: object) -> bool:
        return True

    def sync_state_with_parent(self) -> None:
        return None

    def set_state(self, _state: object) -> None:
        return None


class _Tee(_Element):
    def __init__(self) -> None:
        super().__init__("tee")
        self.sink = _Pad(_Caps("video/x-raw(memory:NVMM),format=NV12,width=1920,height=1080"))
        self.requested: list[_Pad] = []

    def get_request_pad(self, _template: str) -> _Pad:
        pad = _Pad()
        self.requested.append(pad)
        return pad

    def release_request_pad(self, pad: _Pad) -> None:
        self.requested.remove(pad)


class _Pipeline:
    def __init__(self) -> None:
        self.elements: list[object] = []

    def add(self, element: object) -> None:
        self.elements.append(element)

    def remove(self, element: object) -> None:
        self.elements.remove(element)


class _CapsFactory:
    @staticmethod
    def from_string(value: str) -> _Caps:
        return _Caps(value)


class _Gst:
    Caps = _CapsFactory
    PadLinkReturn = type("PadLinkReturn", (), {"OK": 0})
    State = type("State", (), {"NULL": 0})
    PadProbeType = type("PadProbeType", (), {"BLOCK_DOWNSTREAM": 1})
    PadProbeReturn = type("PadProbeReturn", (), {"OK": 0})


def _manager() -> GpuLiveBranchManager:
    manager = GpuLiveBranchManager(
        publish_base="rtsp://mediamtx:8554",
        browser_base="http://host:8889",
        enabled=True,
    )
    manager._gst = _Gst()
    manager._make = lambda factory, _name: _Element(factory)  # type: ignore[method-assign]
    return manager


def test_namespace_is_distinct_and_opaque() -> None:
    assert live_stream_path("camera", "wall").startswith("live-branch/wall/")
    assert "preview-" not in live_stream_path("camera", "wall")


def test_disabled_mode_does_not_create_branch() -> None:
    manager = GpuLiveBranchManager(publish_base="rtsp://mediamtx:8554", enabled=False)
    assert manager.acquire("camera", "wall", "viewer") == {
        "enabled": False,
        "source_id": "camera",
        "profile": "wall",
    }


def test_source_requires_confirmed_nvmm() -> None:
    manager = _manager()
    tee = _Tee()
    pipeline = _Pipeline()
    assert manager.attach_source("camera", tee, pipeline)
    assert manager.acquire("camera", "wall", "viewer")["path"].startswith("live-branch/wall/")
    assert manager.has_source("camera") is True
    assert manager.has_source("missing") is False


def test_confirmed_decoder_caps_register_when_tee_sink_is_not_negotiated() -> None:
    manager = _manager()
    tee = _Tee()
    tee.sink.caps = None
    pipeline = _Pipeline()
    decoded_caps = _Caps("video/x-raw(memory:NVMM),format=NV12,width=(int)1280,height=(int)720")
    assert manager.attach_source("static-file", tee, pipeline, confirmed_caps=decoded_caps)
    assert manager.acquire("static-file", "fullscreen", "viewer")["width"] == 1280


def test_non_nvmm_decoder_output_is_not_registered_or_broadcast() -> None:
    manager = _manager()
    tee = _Tee()
    tee.sink.caps = None
    assert not manager.attach_source(
        "rtsp-camera", tee, _Pipeline(),
        confirmed_caps=_Caps("video/x-raw,format=NV12,width=1280,height=720"),
    )
    with pytest.raises(RuntimeError, match="confirmed NVMM"):
        manager.acquire("rtsp-camera", "wall", "viewer")


def test_wall_caps_are_exact_and_fullscreen_has_no_resize() -> None:
    manager = _manager()
    source = type("Source", (), {"source_id": "camera", "tee": _Tee(), "pipeline": _Pipeline()})()
    wall = manager._build(source, "wall")
    fullscreen = manager._build(source, "fullscreen")
    assert wall.elements[2].props["caps"].to_string() == (
        "video/x-raw(memory:NVMM),format=NV12,width=320,height=320"
    )
    assert "width=" not in fullscreen.elements[2].props["caps"].to_string()
    assert "height=" not in fullscreen.elements[2].props["caps"].to_string()
    assert wall.elements[3].props["idrinterval"] == 15
    assert wall.elements[3].props["iframeinterval"] == 15
    assert fullscreen.elements[3].props["idrinterval"] == 15
    assert fullscreen.elements[3].props["iframeinterval"] == 15


def test_branch_reuse_and_grace_release() -> None:
    manager = _manager()
    tee = _Tee()
    manager.attach_source("camera", tee, _Pipeline())
    first = manager.acquire("camera", "wall", "one")
    second = manager.acquire("camera", "wall", "two")
    assert first["path"] == second["path"]
    manager.release("camera", "wall", "one")
    assert manager.heartbeat("camera", "wall", "two")


def test_expired_heartbeat_schedules_branch_for_grace_cleanup(monkeypatch) -> None:
    clock = {"now": 100.0}
    monkeypatch.setattr("app.core.live_branch.time.monotonic", lambda: clock["now"])
    manager = _manager()
    manager.heartbeat_timeout_seconds = 2.0
    manager.grace_seconds = 1.0
    manager.attach_source("camera", _Tee(), _Pipeline())
    manager.acquire("camera", "wall", "viewer")

    clock["now"] = 103.0
    manager._expire()
    assert ("camera", "wall") in manager._pending_removal

    clock["now"] = 105.0
    manager._expire()
    assert manager.status()["branches"] == {}


def test_invalid_profile_rolls_back_without_branch() -> None:
    manager = _manager()
    with pytest.raises(ValueError):
        manager.acquire("camera", "bad", "viewer")
    assert manager.status()["branches"] == {}


def test_source_refresh_detaches_all_owned_branches() -> None:
    manager = _manager()
    manager.attach_source("camera", _Tee(), _Pipeline())
    manager.acquire("camera", "wall", "viewer")
    manager.detach_source("camera")
    assert manager.status()["sources"] == 0
    assert manager.status()["branches"] == {}
