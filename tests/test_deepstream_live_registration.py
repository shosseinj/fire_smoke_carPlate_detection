from __future__ import annotations

import threading
from types import SimpleNamespace

from app.core.deepstream_ingestor import DeepStreamIngestor


class _Caps:
    def __init__(self, text: str) -> None:
        self.text = text

    def to_string(self) -> str:
        return self.text


class _Pad:
    def __init__(self, caps: list[_Caps]) -> None:
        self.caps = caps
        self.current_index = 0
        self.linked = False

    def get_current_caps(self) -> _Caps:
        return self.caps[min(self.current_index, len(self.caps) - 1)]

    def advance_caps(self) -> None:
        self.current_index = min(self.current_index + 1, len(self.caps) - 1)

    def query_caps(self, _filter: object) -> _Caps:
        return self.caps[-1]

    def is_linked(self) -> bool:
        return self.linked

    def link(self, _other: object) -> int:
        self.linked = True
        return 0


class _Tee:
    def __init__(self) -> None:
        self.sink = _Pad([_Caps("video/x-raw")])
        self.requested: list[_Pad] = []

    def get_static_pad(self, name: str) -> _Pad:
        return self.sink if name == "sink" else _Pad([_Caps("")])

    def get_request_pad(self, _template: str) -> _Pad:
        pad = _Pad([_Caps("")])
        self.requested.append(pad)
        return pad

    def release_request_pad(self, pad: _Pad) -> None:
        self.requested.remove(pad)


class _Gst:
    PadLinkReturn = SimpleNamespace(OK=0)


class _GLib:
    def __init__(self) -> None:
        self.callbacks: dict[int, object] = {}
        self.removed: list[int] = []
        self.next_id = 1

    def timeout_add(self, _interval: int, callback: object) -> int:
        source_id = self.next_id
        self.next_id += 1
        self.callbacks[source_id] = callback
        return source_id

    def source_remove(self, source_id: int) -> bool:
        self.removed.append(source_id)
        self.callbacks.pop(source_id, None)
        return True


class _Manager:
    enabled = True

    def __init__(self) -> None:
        self.attachments: list[tuple[str, object]] = []

    def set_runtime(self, _gst: object, _glib: object) -> None:
        return None

    def attach_source(self, source_id: str, _tee: object, _pipeline: object, *, confirmed_caps: _Caps) -> bool:
        self.attachments.append((source_id, confirmed_caps))
        return True


def _ingestor(manager: _Manager, glib: _GLib, state: object) -> DeepStreamIngestor:
    ingestor = object.__new__(DeepStreamIngestor)
    ingestor._gst = _Gst()
    ingestor._glib = glib
    ingestor._lock = threading.RLock()
    ingestor._states = {"source": state}
    ingestor._closing_sources = set()
    ingestor.live_branch_manager = manager
    return ingestor


def _state(tee: _Tee) -> SimpleNamespace:
    return SimpleNamespace(
        source_id="source",
        pipeline=object(),
        tee=tee,
        live_registration_retry_id=None,
        live_registration_attempts=0,
    )


def _native_caps() -> SimpleNamespace:
    pads = {
        "sink": _Pad([_Caps("")]),
        "src": _Pad([_Caps("")]),
    }
    return SimpleNamespace(get_static_pad=lambda name: pads[name])


def test_early_video_caps_defer_until_nvmm_and_preserve_ai_link() -> None:
    tee = _Tee()
    decoded_pad = _Pad([
        _Caps("video/x-raw,format=NV12"),
        _Caps("video/x-raw(memory:NVMM),format=NV12,width=(int)1280,height=(int)720,framerate=25/1"),
    ])
    decoded_pad.query_caps = lambda _filter: decoded_pad.get_current_caps()  # type: ignore[method-assign]
    ai_pacer = SimpleNamespace(get_static_pad=lambda _name: _Pad([_Caps("")]))
    manager = _Manager()
    glib = _GLib()
    ingestor = _ingestor(manager, glib, _state(tee))

    ingestor._on_decoded_pad_added(None, decoded_pad, tee, _native_caps(), ai_pacer, "source")

    assert decoded_pad.linked
    assert len(tee.requested) == 1
    assert manager.attachments == []
    retry = next(iter(glib.callbacks.values()))
    decoded_pad.advance_caps()
    assert retry() is False
    assert manager.attachments[0][0] == "source"
    assert "memory:NVMM" in manager.attachments[0][1].to_string()


def test_deferred_registration_uses_negotiated_tee_sink_caps() -> None:
    tee = _Tee()
    decoded_pad = _Pad([_Caps("video/x-raw,format=NV12")])
    tee.sink.caps = [_Caps("video/x-raw(memory:NVMM),format=NV12,width=1280,height=720,framerate=25/1")]
    ai_pacer = SimpleNamespace(get_static_pad=lambda _name: _Pad([_Caps("")]))
    manager = _Manager()
    glib = _GLib()
    ingestor = _ingestor(manager, glib, _state(tee))

    ingestor._on_decoded_pad_added(None, decoded_pad, tee, _native_caps(), ai_pacer, "source")

    assert manager.attachments[0][0] == "source"
    assert "width=1280" in manager.attachments[0][1].to_string()


def test_no_nvmm_retry_exhaustion_does_not_affect_ai_branch() -> None:
    tee = _Tee()
    decoded_pad = _Pad([_Caps("video/x-raw,format=NV12")])
    ai_pacer = SimpleNamespace(get_static_pad=lambda _name: _Pad([_Caps("")]))
    manager = _Manager()
    glib = _GLib()
    ingestor = _ingestor(manager, glib, _state(tee))
    ingestor.LIVE_REGISTRATION_MAX_RETRIES = 2

    ingestor._on_decoded_pad_added(None, decoded_pad, tee, _native_caps(), ai_pacer, "source")
    retry = next(iter(glib.callbacks.values()))
    assert retry() is True
    assert retry() is False
    assert decoded_pad.linked
    assert len(tee.requested) == 1
    assert manager.attachments == []


def test_slow_static_caps_can_register_after_original_one_second_window() -> None:
    tee = _Tee()
    decoded_pad = _Pad([
        _Caps("video/x-raw,format=NV12"),
        _Caps("video/x-raw(memory:NVMM),format=NV12,width=1920,height=1080"),
    ])
    decoded_pad.query_caps = lambda _filter: decoded_pad.get_current_caps()  # type: ignore[method-assign]
    manager = _Manager()
    glib = _GLib()
    ingestor = _ingestor(manager, glib, _state(tee))
    ai_pacer = SimpleNamespace(get_static_pad=lambda _name: _Pad([_Caps("")]))

    ingestor._on_decoded_pad_added(None, decoded_pad, tee, _native_caps(), ai_pacer, "source")
    retry = next(iter(glib.callbacks.values()))
    for _ in range(25):
        assert retry() is True
    decoded_pad.advance_caps()
    assert retry() is False
    assert manager.attachments[0][0] == "source"

import pytest

pytestmark = pytest.mark.streaming
