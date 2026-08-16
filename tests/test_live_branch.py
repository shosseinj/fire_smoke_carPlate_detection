from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest

from app.core.live_branch import GpuLiveBranchManager, RecordingUpload, live_stream_path


def test_recovered_fragment_uses_media_duration_when_interval_is_missing(tmp_path: Path) -> None:
    instant = "2026-08-16T08:35:56+00:00"
    upload = RecordingUpload(
        tmp_path / "fragment.mp4", "4", instant, instant,
        "continuous/4/2026/08/16/fragment.mp4", "camera",
    )

    repaired = GpuLiveBranchManager._repair_upload_interval(upload, 19.8)

    assert repaired.end_time == instant
    assert repaired.start_time == "2026-08-16T08:35:36.200000+00:00"
    assert repaired.object_name == upload.object_name


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
        self.signals: dict[str, object] = {}
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

    def connect(self, name: str, callback: object) -> None:
        self.signals[name] = callback


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
    manager = GpuLiveBranchManager(
        publish_base="rtsp://mediamtx:8554",
        enabled=False,
        recording_spool_path="saved_media/temporary_minIO/continuous",
    )
    assert manager.recording_spool_path == Path("saved_media/temporary_minIO/continuous")
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
    assert manager.acquire("camera", "wall", "viewer")["path"].startswith("live-branch/fullscreen/")
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
        "video/x-raw(memory:NVMM),format=NV12,width=320,height=260"
    )
    assert "width=" not in fullscreen.elements[2].props["caps"].to_string()
    assert "height=" not in fullscreen.elements[2].props["caps"].to_string()
    assert wall.elements[3].props["idrinterval"] == 15
    assert wall.elements[3].props["iframeinterval"] == 15
    assert fullscreen.elements[3].props["idrinterval"] == 15
    assert fullscreen.elements[3].props["iframeinterval"] == 15


def test_fullscreen_fragment_rotation_keeps_canonical_source_id(tmp_path) -> None:
    manager = _manager()
    manager.recording_spool_path = tmp_path
    source = type(
        "Source",
        (),
        {
            "source_id": "rtsp://camera.example/live",
            "camera_id": "12",
            "tee": _Tee(),
            "pipeline": _Pipeline(),
        },
    )()
    branch = manager._build(source, "fullscreen")
    splitmux = next(element for element in branch.elements if element.factory == "splitmuxsink")
    callback = splitmux.signals["format-location"]

    callback(splitmux, 0)
    callback(splitmux, 1)

    assert manager._upload_queue.get_nowait().source_id == "rtsp://camera.example/live"


def test_terminal_recording_failure_is_quarantined_without_data_loss(tmp_path) -> None:
    manager = _manager()
    manager.recording_spool_path = tmp_path
    recording = tmp_path / "camera-12-broken.mp4"
    sidecar = recording.with_suffix(".json")
    recording.write_bytes(b"broken")
    sidecar.write_text("{}", encoding="utf-8")

    quarantined = manager._quarantine_recording(recording)

    assert quarantined.read_bytes() == b"broken"
    assert quarantined.parent == tmp_path / "failed"
    assert quarantined.with_suffix(".json").read_text(encoding="utf-8") == "{}"
    assert not recording.exists()


def test_successful_recording_is_archived_outside_retry_spool(tmp_path) -> None:
    manager = _manager()
    manager.recording_archive_path = tmp_path / "saved_media" / "continuous"
    spool = tmp_path / "temporary"
    spool.mkdir()
    recording = spool / "camera-12-part.mp4"
    sidecar = recording.with_suffix(".json")
    recording.write_bytes(b"video")
    sidecar.write_text('{"durable": true}', encoding="utf-8")
    upload = manager._recording_upload(
        recording, "12", __import__("datetime").datetime(2026, 8, 16,
            tzinfo=__import__("datetime").timezone.utc),
        __import__("datetime").datetime(2026, 8, 16, 0, 0, 10,
            tzinfo=__import__("datetime").timezone.utc),
    )

    archived = manager._archive_upload(upload)

    assert archived == manager.recording_archive_path / "12/2026/08/16/camera-12-part.mp4"
    assert archived.read_bytes() == b"video"
    assert archived.with_suffix(".json").read_text(encoding="utf-8") == '{"durable": true}'
    assert not recording.exists()


def _retained_upload(manager: GpuLiveBranchManager, tmp_path: Path, age_seconds: float) -> RecordingUpload:
    manager.recording_spool_path = tmp_path / "temporary"
    manager.recording_archive_path = tmp_path / "archive"
    manager.recording_spool_path.mkdir()
    recording = manager.recording_spool_path / "camera-12-retained.mp4"
    recording.write_bytes(b"video")
    upload = manager._recording_upload(
        recording, "12", datetime(2026, 8, 16, tzinfo=timezone.utc),
        datetime(2026, 8, 16, 0, 0, 10, tzinfo=timezone.utc),
    )
    manager._persist_upload(upload)
    sidecar = upload.path.with_suffix(".json")
    import json
    value = json.loads(sidecar.read_text(encoding="utf-8"))
    value["upload_succeeded_at"] = (datetime.now(timezone.utc) - timedelta(seconds=age_seconds)).isoformat()
    sidecar.write_text(json.dumps(value), encoding="utf-8")
    return upload


def test_acknowledged_recording_is_retained_until_expiry(tmp_path) -> None:
    manager = _manager()
    manager.recording_success_retention_seconds = 180
    upload = _retained_upload(manager, tmp_path, 10)

    manager._cleanup_retained_uploads()

    assert upload.path.is_file()
    assert manager._upload_succeeded_at(upload.path) is not None


@pytest.mark.parametrize("retention,age", [(180, 181), (0, 0)])
def test_acknowledged_recording_archives_after_retention(
    tmp_path, retention: float, age: float,
) -> None:
    manager = _manager()
    manager.recording_success_retention_seconds = retention
    upload = _retained_upload(manager, tmp_path, age)

    manager._cleanup_retained_uploads()

    assert not upload.path.exists()
    assert (manager.recording_archive_path / "12/2026/08/16/camera-12-retained.mp4").is_file()


def test_missing_or_malformed_acknowledgement_remains_recoverable(tmp_path) -> None:
    manager = _manager()
    upload = _retained_upload(manager, tmp_path, 10)
    sidecar = upload.path.with_suffix(".json")
    sidecar.write_text(sidecar.read_text(encoding="utf-8").replace(
        '"upload_succeeded_at":', '"upload_succeeded_at": "bad", "replaced":'
    ), encoding="utf-8")

    assert manager._upload_succeeded_at(upload.path) is None
    assert manager._recover_upload(upload.path).path == upload.path


def test_cleanup_failure_retains_acknowledgement_and_retries_without_upload(
    tmp_path, monkeypatch,
) -> None:
    manager = _manager()
    manager.recording_success_retention_seconds = 0
    upload = _retained_upload(manager, tmp_path, 10)
    archive = manager._archive_upload
    attempts = {"count": 0}

    def fail_once(candidate):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise OSError("busy")
        return archive(candidate)

    monkeypatch.setattr(manager, "_archive_upload", fail_once)
    manager._cleanup_retained_uploads()
    assert upload.path.is_file()
    assert manager._upload_succeeded_at(upload.path) is not None

    manager._cleanup_retained_uploads()
    assert not upload.path.exists()
    assert manager._upload_queue.empty()


def test_branch_reuse_and_grace_release() -> None:
    manager = _manager()
    tee = _Tee()
    manager.attach_source("camera", tee, _Pipeline())
    first = manager.acquire("camera", "wall", "one")
    second = manager.acquire("camera", "wall", "two")
    assert first["path"] == second["path"]
    manager.release("camera", "wall", "one")
    assert manager.heartbeat("camera", "wall", "two")


def test_wall_viewer_reuses_fullscreen_recording_encoder() -> None:
    manager = _manager()
    manager.attach_source("camera", _Tee(), _Pipeline())

    wall = manager.acquire("camera", "wall", "wall-viewer")
    fullscreen = manager.acquire("camera", "fullscreen", "fullscreen-viewer")

    assert wall["path"] == fullscreen["path"]
    assert list(manager.status()["branches"]) == ["camera/fullscreen"]
    assert manager.release("camera", "wall", "wall-viewer")


def test_durable_fullscreen_owner_does_not_expire_without_heartbeat(monkeypatch) -> None:
    clock = {"now": 100.0}
    monkeypatch.setattr("app.core.live_branch.time.monotonic", lambda: clock["now"])
    manager = _manager()
    manager.heartbeat_timeout_seconds = 1.0
    manager.attach_source("camera", _Tee(), _Pipeline())
    manager.acquire_durable("camera", "recording:job")

    clock["now"] = 200.0
    manager._expire()

    assert "camera/fullscreen" in manager.status()["branches"]
    assert manager.release_durable("camera", "recording:job")


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
    branch = manager._branches[("camera", "fullscreen")]
    assert "viewer" not in branch.references
    assert ("camera", "fullscreen") not in manager._pending_removal

    clock["now"] = 105.0
    manager._expire()
    assert "camera/fullscreen" in manager.status()["branches"]


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
