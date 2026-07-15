from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.core.source_registry import SourceRecord, SourceRegistry
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
    assert ingestor.status()["sources"]["camera-01"]["stride"] == 2
    ingestor.close()
