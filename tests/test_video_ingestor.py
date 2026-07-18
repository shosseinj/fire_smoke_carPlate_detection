from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from app.core.source_registry import SourceRecord, SourceRegistry
from app.core.deepstream_ingestor import DeepStreamIngestor
from app.core.types import TaskName
from app.core.video_ingestor import VideoFileIngestor


class FakeCapture:
    def __init__(self, _: str) -> None:
        self.position = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self):
        self.position += 1
        return True, np.full((4, 6, 3), self.position, dtype=np.uint8)

    def grab(self) -> bool:
        self.position += 1
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return 10.0
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.position)
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.position * 100.0
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.position = int(value)
        return True

    def release(self) -> None:
        self.released = True


class RecordingRouter:
    def __init__(self, registry: SourceRegistry) -> None:
        self.registry = registry
        self.calls: list[dict] = []

    def submit_round(self, **kwargs):
        self.calls.append(kwargs)
        task_submissions = sum(
            len(self.registry.require(source_id).tasks) for source_id in kwargs["source_ids"]
        )
        return {
            "received_frames": len(kwargs["frames"]),
            "accepted_sources": len(kwargs["frames"]),
            "task_submissions": task_submissions,
        }


class FakeCaps:
    def __init__(self, value: str) -> None:
        self.value = value

    def to_string(self) -> str:
        return self.value


class FakeElementFactory:
    def __init__(self, name: str) -> None:
        self.name = name

    def get_name(self) -> str:
        return self.name


class FakeElement:
    def __init__(self, factory_name: str) -> None:
        self.factory = FakeElementFactory(factory_name)
        self.connections: list[tuple[str, object]] = []

    def get_factory(self) -> FakeElementFactory:
        return self.factory

    def connect(self, signal: str, callback: object) -> None:
        self.connections.append((signal, callback))


def test_video_files_are_sampled_as_one_camera_round(tmp_path: Path) -> None:
    registry = SourceRegistry()
    registry.create(
        SourceRecord(
            source_id="camera-01",
            name="Fire camera",
            tasks={TaskName.FIRE_SMOKE},
            source_uri="data/fire.mp4",
            metadata={"kind": "video_file"},
        )
    )
    registry.create(
        SourceRecord(
            source_id="camera-02",
            name="Mixed camera",
            tasks={TaskName.FIRE_SMOKE, TaskName.PLATE_RECOGNITION},
            source_uri="data/mixed.mp4",
            metadata={"kind": "video_file"},
        )
    )
    registry.create(
        SourceRecord(
            source_id="camera-03",
            name="Disabled camera",
            enabled=False,
            tasks={TaskName.PLATE_RECOGNITION},
            source_uri="data/disabled.mp4",
            metadata={"kind": "video_file"},
        )
    )
    router = RecordingRouter(registry)
    ingestor = VideoFileIngestor(
        registry=registry,
        router=router,  # type: ignore[arg-type]
        project_root=tmp_path,
        target_fps=5.0,
        capture_factory=FakeCapture,
    )

    summary = ingestor.process_once()

    assert summary == {
        "received_frames": 2,
        "accepted_sources": 2,
        "task_submissions": 3,
    }
    assert router.calls[0]["source_ids"] == ["camera-01", "camera-02"]
    assert router.calls[0]["frame_indexes"] == [0, 0]
    assert router.calls[0]["source_times_seconds"] == [0.1, 0.1]
    assert all(frame.shape == (640, 640, 3) for frame in router.calls[0]["frames"])
    assert ingestor.status()["sources"]["camera-01"]["stride"] == 2
    ingestor.close()


def test_deepstream_accepts_rtsp_and_local_sources_without_metadata(tmp_path: Path) -> None:
    registry = SourceRegistry()
    router = RecordingRouter(registry)
    ingestor = DeepStreamIngestor(
        registry=registry,
        router=router,  # type: ignore[arg-type]
        project_root=tmp_path,
    )
    rtsp = SourceRecord(
        source_id="camera-01",
        name="RTSP",
        source_uri="rtsp://user:password@192.0.2.10:554/live",
        metadata={},
    )
    local = SourceRecord(
        source_id="camera-02",
        name="File",
        source_uri="data/example.mp4",
        metadata={},
    )
    registry.create(rtsp)
    registry.create(local)

    assert ingestor.is_supported_source(rtsp) is True
    assert ingestor.is_supported_source(local) is True
    assert [record.source_id for record in ingestor._active_records()] == [
        "camera-01",
        "camera-02",
    ]
    assert ingestor._resolve_uri(rtsp.source_uri or "") == rtsp.source_uri
    assert ingestor._resolve_uri(local.source_uri or "").startswith("file:")
    assert "example.mp4" in ingestor._resolve_uri(local.source_uri or "")


def test_deepstream_static_only_mode_excludes_rtsp_before_open(tmp_path: Path) -> None:
    registry = SourceRegistry()
    registry.create(
        SourceRecord(
            source_id="camera-01",
            name="RTSP",
            source_uri="rtsp://user:password@192.0.2.10:554/live",
            metadata={},
        )
    )
    registry.create(
        SourceRecord(
            source_id="camera-02",
            name="File",
            source_uri="data/example.mp4",
            metadata={},
        )
    )
    ingestor = DeepStreamIngestor(
        registry=registry,
        router=RecordingRouter(registry),  # type: ignore[arg-type]
        project_root=tmp_path,
        rtsp_enabled=False,
    )

    assert [record.source_id for record in ingestor._active_records()] == ["camera-02"]


def test_deepstream_frame_index_remains_monotonic_across_file_reopen(
    tmp_path: Path,
) -> None:
    registry = SourceRegistry()
    ingestor = DeepStreamIngestor(
        registry=registry,
        router=RecordingRouter(registry),  # type: ignore[arg-type]
        project_root=tmp_path,
    )

    assert ingestor._next_frame_index_locked("camera-loop") == 0
    assert ingestor._next_frame_index_locked("camera-loop") == 1
    # A replacement Gst pipeline uses the same per-camera sequence.
    assert ingestor._next_frame_index_locked("camera-loop") == 2


def test_deepstream_uses_stable_numeric_source_ids_and_separate_stall_timeout(
    tmp_path: Path,
) -> None:
    registry = SourceRegistry()
    ingestor = DeepStreamIngestor(
        registry=registry,
        router=RecordingRouter(registry),  # type: ignore[arg-type]
        project_root=tmp_path,
        rtsp_reconnect_seconds=3,
        rtsp_stall_timeout_seconds=30,
    )

    assert ingestor.rtsp_reconnect_seconds == 3
    assert ingestor.rtsp_stall_timeout_seconds == 30
    assert ingestor._gst_source_id("camera-03") == 3
    assert ingestor._gst_source_id("camera-03") == 3
    assert ingestor._gst_source_id("warehouse") == 0
    assert ingestor._gst_source_id("video-03") == 1
    assert ingestor.status()["rtsp_stall_timeout_seconds"] == 30


def test_deepstream_skips_unused_audio_during_decoder_autoplug(tmp_path: Path) -> None:
    registry = SourceRegistry()
    ingestor = DeepStreamIngestor(
        registry=registry,
        router=RecordingRouter(registry),  # type: ignore[arg-type]
        project_root=tmp_path,
    )
    decodebin = FakeElement("uridecodebin")
    other = FakeElement("nvv4l2decoder")

    ingestor._on_deep_element_added(None, None, decodebin)
    ingestor._on_deep_element_added(None, None, other)

    assert len(decodebin.connections) == 1
    signal, callback = decodebin.connections[0]
    assert signal == "autoplug-continue"
    assert callable(callback)
    assert callback(None, None, FakeCaps("audio/mpeg, mpegversion=(int)4")) is False
    assert callback(None, None, FakeCaps("video/x-h264")) is True
    assert other.connections == []


def test_deepstream_applies_camera_crud_and_uri_changes_without_restart(
    tmp_path: Path,
) -> None:
    registry = SourceRegistry()
    ingestor = DeepStreamIngestor(
        registry=registry,
        router=RecordingRouter(registry),  # type: ignore[arg-type]
        project_root=tmp_path,
    )
    opened: list[tuple[str, str | None]] = []
    closed: list[str] = []

    def open_source(record: SourceRecord) -> None:
        opened.append((record.source_id, record.source_uri))
        ingestor._states[record.source_id] = SimpleNamespace(
            source_uri=record.source_uri,
            frame_width=record.frame_width,
            frame_height=record.frame_height,
        )

    def close_source(source_id: str) -> None:
        closed.append(source_id)
        ingestor._states.pop(source_id, None)

    ingestor._open_source = open_source  # type: ignore[method-assign]
    ingestor._close_source = close_source  # type: ignore[method-assign]

    registry.create(
        SourceRecord(
            source_id="camera-live",
            name="Live camera",
            tasks={TaskName.FIRE_SMOKE},
            source_uri="rtsp://example.test/first",
        )
    )
    ingestor._sync_sources()
    assert opened == [("camera-live", "rtsp://example.test/first")]

    registry.update("camera-live", source_uri="rtsp://example.test/second")
    ingestor._sync_sources()
    assert closed == ["camera-live"]
    assert opened[-1] == ("camera-live", "rtsp://example.test/second")

    registry.update("camera-live", enabled=False)
    ingestor._sync_sources()
    assert closed == ["camera-live", "camera-live"]

    registry.update("camera-live", enabled=True)
    ingestor._sync_sources()
    assert opened[-1] == ("camera-live", "rtsp://example.test/second")

    registry.delete("camera-live")
    ingestor._sync_sources()
    assert closed == ["camera-live", "camera-live", "camera-live"]
