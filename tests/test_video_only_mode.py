from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np

from app.core.broadcast import AnnotatedBroadcastHub
from app.core.frontend_frame_worker import FrontendFrameWorker
from app.core.source_registry import STATIC_VIDEO, SourceRecord, SourceRegistry
from app.core.stream_demand import StreamDemandController
from app.core.types import TaskName
from app.core.video_ingestor import StaticVideoFileIngestor


class FakeCapture:
    instances: list["FakeCapture"] = []

    def __init__(self, _uri: str) -> None:
        self.position = 0
        self.released = False
        self.instances.append(self)

    def isOpened(self) -> bool:
        return True

    def read(self):
        self.position += 1
        return True, np.full((24, 32, 3), self.position, dtype=np.uint8)

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
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def submit_round(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "received_frames": len(kwargs["frames"]),
            "accepted_sources": len(kwargs["frames"]),
            "task_submissions": 0,
        }


class RecordingFrontendWorker:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def submit_frame(self, **kwargs) -> bool:
        self.calls.append(kwargs)
        return True


def test_static_video_pauses_without_demand_and_resumes_for_video_subscriber(
    tmp_path: Path,
    source_registry: SourceRegistry,
) -> None:
    FakeCapture.instances.clear()
    source_registry.create(
        SourceRecord(
            source_uri="data/video.mp4",
            source_type=STATIC_VIDEO,
            tasks={TaskName.FACE_RECOGNITION},
        )
    )
    demand = StreamDemandController()
    router = RecordingRouter()
    frontend = RecordingFrontendWorker()
    ingestor = StaticVideoFileIngestor(
        registry=source_registry,
        router=router,  # type: ignore[arg-type]
        project_root=tmp_path,
        capture_factory=FakeCapture,
        frontend_frame_worker=frontend,  # type: ignore[arg-type]
        demand_controller=demand,
        video_only_mode=True,
    )
    try:
        assert ingestor.process_once()["received_frames"] == 0
        assert FakeCapture.instances == []
        assert frontend.calls == []
        assert router.calls == []

        lease = demand.acquire_video()
        assert ingestor.process_once()["received_frames"] == 0
        assert len(FakeCapture.instances) == 1
        assert FakeCapture.instances[0].position == 1
        assert len(frontend.calls) == 1
        assert frontend.calls[0]["frame"].shape == (24, 32, 3)
        assert router.calls == []

        lease.release()
        paused_position = FakeCapture.instances[0].position
        published = len(frontend.calls)
        for _ in range(3):
            assert ingestor.process_once()["received_frames"] == 0
        assert FakeCapture.instances[0].position == paused_position
        assert len(frontend.calls) == published
        assert router.calls == []
    finally:
        ingestor.close()


def test_static_video_opens_only_one_new_source_per_scheduler_round(
    tmp_path: Path,
    source_registry: SourceRegistry,
) -> None:
    FakeCapture.instances.clear()
    for index in range(3):
        source_registry.create(
            SourceRecord(
                source_uri=f"data/video-{index}.mp4",
                source_type=STATIC_VIDEO,
            )
        )
    demand = StreamDemandController()
    lease = demand.acquire_video()
    ingestor = StaticVideoFileIngestor(
        registry=source_registry,
        router=RecordingRouter(),  # type: ignore[arg-type]
        project_root=tmp_path,
        capture_factory=FakeCapture,
        frontend_frame_worker=RecordingFrontendWorker(),  # type: ignore[arg-type]
        demand_controller=demand,
        video_only_mode=True,
    )
    try:
        ingestor.process_once()
        assert len(FakeCapture.instances) == 1
        ingestor.process_once()
        assert len(FakeCapture.instances) == 2
        ingestor.process_once()
        assert len(FakeCapture.instances) == 3
    finally:
        lease.release()
        ingestor.close()


def test_frontend_and_renderer_counters_stop_after_last_disconnect(
    tmp_path: Path,
    source_registry: SourceRegistry,
) -> None:
    FakeCapture.instances.clear()
    source_registry.create(
        SourceRecord(
            source_uri="data/video.mp4",
            source_type=STATIC_VIDEO,
        )
    )
    demand = StreamDemandController()
    router = RecordingRouter()
    hub = AnnotatedBroadcastHub(enabled=True, async_render=False)
    worker = FrontendFrameWorker(publish_callback=hub.publish_source_frame)
    ingestor = StaticVideoFileIngestor(
        registry=source_registry,
        router=router,  # type: ignore[arg-type]
        project_root=tmp_path,
        capture_factory=FakeCapture,
        frontend_frame_worker=worker,
        demand_controller=demand,
        video_only_mode=True,
    )
    worker.start()
    subscriber_id, _target = hub.subscribe_source_only(wall=True)
    lease = demand.acquire_video()
    try:
        ingestor.process_once()
        deadline = time.monotonic() + 2.0
        while worker.status()["published"] == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        assert worker.status()["submitted"] == 1
        assert worker.status()["published"] == 1
        assert hub.status()["source_only_rendered"] == 1

        lease.release()
        hub.unsubscribe_source_only(subscriber_id)
        submitted = worker.status()["submitted"]
        rendered = hub.status()["source_only_rendered"]
        for _ in range(3):
            ingestor.process_once()
        time.sleep(0.05)

        assert worker.status()["submitted"] == submitted
        assert hub.status()["source_only_rendered"] == rendered
    finally:
        lease.release()
        hub.unsubscribe_source_only(subscriber_id)
        ingestor.close()
        worker.close()
        hub.close()
