from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from app.core.source_registry import STATIC_VIDEO, SourceRecord, SourceRegistry
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


def test_video_files_are_sampled_as_one_camera_round(
    tmp_path: Path, source_registry: SourceRegistry
) -> None:
    registry = source_registry
    registry.create(
        SourceRecord(
            name="Fire camera",
            tasks={TaskName.FIRE_SMOKE},
            source_uri="data/fire.mp4",
            metadata={"kind": "video_file"},
        )
    )
    registry.create(
        SourceRecord(
            name="Mixed camera",
            tasks={TaskName.FIRE_SMOKE, TaskName.PLATE_RECOGNITION},
            source_uri="data/mixed.mp4",
            metadata={"kind": "video_file"},
        )
    )
    registry.create(
        SourceRecord(
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
        capture_factory=FakeCapture,
    )

    summary = ingestor.process_once()

    assert summary == {
        "received_frames": 2,
        "accepted_sources": 2,
        "task_submissions": 3,
    }
    assert router.calls[0]["source_ids"] == ["data/fire.mp4", "data/mixed.mp4"]
    assert router.calls[0]["frame_indexes"] == [0, 0]
    assert router.calls[0]["source_times_seconds"] == [0.1, 0.1]
    assert all(frame.shape == (640, 640, 3) for frame in router.calls[0]["frames"])
    assert all(
        item["source_frame"].shape == (4, 6, 3)
        for item in router.calls[0]["metadata"]
    )
    assert router.calls[0]["metadata"][0]["source_frame_width"] == 6
    assert router.calls[0]["metadata"][0]["source_frame_height"] == 4
    assert ingestor.status()["sources"]["data/fire.mp4"]["source_fps"] == 10.0
    assert ingestor.status()["sources"]["data/fire.mp4"]["effective_fps"] == 10.0
    assert ingestor.status()["sources"]["data/fire.mp4"]["stride"] == 1
    ingestor.close()


def test_video_file_uses_explicit_source_fps_override_for_slowdown(
    tmp_path: Path, source_registry: SourceRegistry
) -> None:
    registry = source_registry
    registry.create(
        SourceRecord(
            name="Slow fire camera",
            tasks={TaskName.FIRE_SMOKE},
            source_uri="data/fire.mp4",
            metadata={"kind": "video_file"},
            fps=5.0,
        )
    )
    router = RecordingRouter(registry)
    ingestor = VideoFileIngestor(
        registry=registry,
        router=router,  # type: ignore[arg-type]
        project_root=tmp_path,
        capture_factory=FakeCapture,
    )

    summary = ingestor.process_once()

    assert summary["received_frames"] == 1
    status = ingestor.status()["sources"]["data/fire.mp4"]
    assert status["source_fps"] == 10.0
    assert status["effective_fps"] == 5.0
    assert status["stride"] == 2
    ingestor.close()


def test_video_file_source_fps_override_can_accelerate_playback(
    tmp_path: Path, source_registry: SourceRegistry
) -> None:
    registry = source_registry
    registry.create(
        SourceRecord(
            name="Fast file",
            source_uri="data/fast.mp4",
            source_type=STATIC_VIDEO,
            fps=20.0,
        )
    )
    ingestor = VideoFileIngestor(
        registry=registry,
        router=RecordingRouter(registry),  # type: ignore[arg-type]
        project_root=tmp_path,
        source_type_filter=STATIC_VIDEO,
        capture_factory=FakeCapture,
    )

    assert ingestor.process_once()["received_frames"] == 1
    status = ingestor.status()["sources"]["data/fast.mp4"]
    assert status["native_fps"] == 10.0
    assert status["configured_fps"] == 20.0
    assert status["effective_fps"] == 20.0
    assert status["fps_mode"] == "override"
    assert status["stride"] == 1
    ingestor.close()


def test_deepstream_accepts_rtsp_and_local_sources_without_metadata(
    tmp_path: Path, source_registry: SourceRegistry
) -> None:
    registry = source_registry
    router = RecordingRouter(registry)
    ingestor = DeepStreamIngestor(
        registry=registry,
        router=router,  # type: ignore[arg-type]
        project_root=tmp_path,
    )
    rtsp = SourceRecord(
        name="RTSP",
        source_uri="rtsp://user:password@192.0.2.10:554/live",
        metadata={},
    )
    local = SourceRecord(
        name="File",
        source_uri="data/example.mp4",
        metadata={},
    )
    registry.create(rtsp)
    registry.create(local)

    assert ingestor.is_supported_source(rtsp) is True
    assert ingestor.is_supported_source(local) is True
    assert [record.source_uri for record in ingestor._active_records()] == [
        rtsp.source_uri,
        local.source_uri,
    ]
    assert ingestor._resolve_uri(rtsp.source_uri or "") == rtsp.source_uri
    assert ingestor._resolve_uri(local.source_uri or "").startswith("file:")
    assert "example.mp4" in ingestor._resolve_uri(local.source_uri or "")


def test_rtsp_uri_overrides_stale_static_video_classification(
    tmp_path: Path, source_registry: SourceRegistry
) -> None:
    registry = source_registry
    record = registry.create(
        SourceRecord(
            name="Misclassified RTSP",
            source_uri="rtsp://user:password@192.0.2.10:554/live",
            source_type=STATIC_VIDEO,
        )
    )

    assert record.source_type == "rtsp"
    assert registry.require(record.source_uri).source_type == "rtsp"

    deepstream = DeepStreamIngestor(
        registry=registry,
        router=RecordingRouter(registry),  # type: ignore[arg-type]
        project_root=tmp_path,
    )
    static_ingestor = VideoFileIngestor(
        registry=registry,
        router=RecordingRouter(registry),  # type: ignore[arg-type]
        project_root=tmp_path,
        source_type_filter=STATIC_VIDEO,
        capture_factory=lambda _: (_ for _ in ()).throw(
            AssertionError("Static ingestor must not open an RTSP URI")
        ),
    )

    assert [item.source_uri for item in deepstream._active_records()] == [
        record.source_uri
    ]
    assert static_ingestor.process_once()["received_frames"] == 0
    static_ingestor.close()


def test_deepstream_static_only_mode_excludes_rtsp_before_open(
    tmp_path: Path, source_registry: SourceRegistry
) -> None:
    registry = source_registry
    registry.create(
        SourceRecord(
            name="RTSP",
            source_uri="rtsp://user:password@192.0.2.10:554/live",
            metadata={},
        )
    )
    registry.create(
        SourceRecord(
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

    assert [record.source_uri for record in ingestor._active_records()] == ["data/example.mp4"]


def test_deepstream_frame_index_remains_monotonic_across_file_reopen(
    tmp_path: Path, source_registry: SourceRegistry
) -> None:
    registry = source_registry
    ingestor = DeepStreamIngestor(
        registry=registry,
        router=RecordingRouter(registry),  # type: ignore[arg-type]
        project_root=tmp_path,
    )

    assert ingestor._next_frame_index_locked("camera-loop") == 0
    assert ingestor._next_frame_index_locked("camera-loop") == 1
    # A replacement Gst pipeline uses the same per-camera sequence.
    assert ingestor._next_frame_index_locked("camera-loop") == 2


def test_deepstream_failed_source_disposal_does_not_block_healthy_source(
    tmp_path: Path, source_registry: SourceRegistry
) -> None:
    ingestor = DeepStreamIngestor(
        registry=source_registry,
        router=RecordingRouter(source_registry),  # type: ignore[arg-type]
        project_root=tmp_path,
    )
    healthy = SimpleNamespace(source_id="healthy")
    failed = SimpleNamespace(source_id="failed")
    ingestor._states = {"healthy": healthy, "failed": failed}  # type: ignore[assignment]
    disposal_started = threading.Event()
    allow_disposal = threading.Event()

    def blocking_disposal(_: object) -> None:
        disposal_started.set()
        allow_disposal.wait(timeout=2.0)

    ingestor._dispose_state = blocking_disposal  # type: ignore[method-assign]

    started = time.monotonic()
    ingestor._close_source("failed")
    elapsed = time.monotonic() - started

    assert elapsed < 0.25
    assert disposal_started.wait(timeout=0.5)
    assert ingestor._states == {"healthy": healthy}
    assert ingestor._closing_sources == {"failed"}

    allow_disposal.set()
    ingestor._join_close_threads(timeout=1.0)
    assert ingestor._closing_sources == set()


def test_deepstream_failure_status_redacts_rtsp_credentials(
    tmp_path: Path, source_registry: SourceRegistry
) -> None:
    ingestor = DeepStreamIngestor(
        registry=source_registry,
        router=RecordingRouter(source_registry),  # type: ignore[arg-type]
        project_root=tmp_path,
    )
    source_uri = "rtsp://user:password@192.0.2.10:554/live"
    state = SimpleNamespace(
        source_uri=source_uri,
        display_uri="rtsp://***:***@192.0.2.10:554/live",
        last_error=None,
    )
    ingestor._states[source_uri] = state  # type: ignore[assignment]

    ingestor._mark_failed(source_uri, f"connection failed for {source_uri}")

    assert "password" not in str(ingestor._last_error)
    assert "password" not in str(state.last_error)
    assert "***:***@" in str(ingestor._last_error)
    assert "password" not in ingestor._safe_element_name(source_uri)


def test_deepstream_decodes_gpu_converted_bgrx_without_cpu_videoconvert() -> None:
    payload = bytes([10, 20, 30, 255, 40, 50, 60, 255])

    frame = DeepStreamIngestor._decode_cpu_sample(
        payload, width=2, height=1, pixel_format="BGRx"
    )

    assert frame.shape == (1, 2, 3)
    assert frame.tolist() == [[[10, 20, 30], [40, 50, 60]]]


def test_deepstream_resize_keeps_cpu_fallback_contract(tmp_path: Path) -> None:
    ingestor = DeepStreamIngestor(
        registry=None,  # type: ignore[arg-type]
        router=None,  # type: ignore[arg-type]
        project_root=tmp_path,
        gpu_resize_enabled=False,
    )
    frame = np.zeros((4, 8, 3), dtype=np.uint8)

    resized = ingestor._resize_frame(frame, (4, 2))

    assert resized.shape == (2, 4, 3)
    assert ingestor.status()["gpu_resize_active"] is False


def test_deepstream_submits_640_inference_view_with_native_source_frame(
    tmp_path: Path, source_registry: SourceRegistry
) -> None:
    registry = source_registry
    registry.create(
        SourceRecord(
            name="2K face camera",
            tasks={TaskName.FACE_RECOGNITION},
            source_uri="data/face.mp4",
            frame_width=640,
            frame_height=640,
        )
    )
    router = RecordingRouter(registry)
    ingestor = DeepStreamIngestor(
        registry=registry,
        router=router,  # type: ignore[arg-type]
        project_root=tmp_path,
    )
    source_frame = np.full((1080, 2048, 3), 31, dtype=np.uint8)
    ingestor._states["data/face.mp4"] = SimpleNamespace(
        latest_frame=source_frame,
        latest_version=1,
        submitted_version=0,
        source_id="data/face.mp4",
        frame_width=640,
        frame_height=640,
        frame_index=7,
        source_time_seconds=1.5,
        display_uri="data/face.mp4",
        source_type="video_file",
        source_frame_width=2048,
        source_frame_height=1080,
        submitted_frames=0,
    )

    ingestor._submit_latest_round()

    call = router.calls[0]
    assert call["frames"][0].shape == (640, 640, 3)
    assert call["metadata"][0]["source_frame"] is source_frame
    assert call["metadata"][0]["source_frame_width"] == 2048
    assert call["metadata"][0]["source_frame_height"] == 1080


def test_deepstream_uses_stable_numeric_source_ids_and_separate_stall_timeout(
    tmp_path: Path, source_registry: SourceRegistry
) -> None:
    registry = source_registry
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
    assert ingestor.status()["fps_control"] == "sources.fps"


def test_deepstream_skips_unused_audio_during_decoder_autoplug(
    tmp_path: Path, source_registry: SourceRegistry
) -> None:
    registry = source_registry
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


def test_deepstream_applies_source_enable_and_task_changes_without_restart(
    tmp_path: Path, source_registry: SourceRegistry
) -> None:
    registry = source_registry
    ingestor = DeepStreamIngestor(
        registry=registry,
        router=RecordingRouter(registry),  # type: ignore[arg-type]
        project_root=tmp_path,
    )
    opened: list[tuple[str, str | None]] = []
    closed: list[str] = []

    def open_source(record: SourceRecord) -> None:
        opened.append((record.source_uri, record.source_uri))
        ingestor._states[record.source_uri] = SimpleNamespace(
            source_uri=record.source_uri,
            frame_width=record.frame_width,
            frame_height=record.frame_height,
            delivery_target_fps=record.fps,
            next_frame_due_monotonic=0.0,
        )

    def close_source(source_id: str) -> None:
        closed.append(source_id)
        ingestor._states.pop(source_id, None)

    ingestor._open_source = open_source  # type: ignore[method-assign]
    ingestor._close_source = close_source  # type: ignore[method-assign]

    registry.create(
        SourceRecord(
            name="Live camera",
            tasks={TaskName.FIRE_SMOKE},
            source_uri="rtsp://example.test/first",
        )
    )
    ingestor._sync_sources()
    source_uri = "rtsp://example.test/first"
    assert opened == [(source_uri, source_uri)]

    ingestor._sync_sources()
    assert ingestor._states[source_uri].delivery_target_fps is None
    registry.update(source_uri, fps=40.0)
    ingestor._sync_sources()
    assert ingestor._states[source_uri].delivery_target_fps == 40.0
    registry.update(source_uri, tasks=set())
    ingestor._sync_sources()
    assert ingestor._states[source_uri].delivery_target_fps == 40.0
    registry.update(source_uri, fps=None)
    ingestor._sync_sources()
    assert ingestor._states[source_uri].delivery_target_fps is None
    assert closed == []

    registry.update(source_uri, enabled=False)
    ingestor._sync_sources()
    assert closed == [source_uri]

    registry.update(source_uri, enabled=True)
    ingestor._sync_sources()
    assert opened[-1] == (source_uri, source_uri)

    registry.delete(source_uri)
    ingestor._sync_sources()
    assert closed == [source_uri, source_uri]
