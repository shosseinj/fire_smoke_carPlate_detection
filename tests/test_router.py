from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from app.core.result_store import ResultStore
from app.core.router import TaskRouter
from app.core.source_registry import SourceRecord, SourceRegistry
from app.core.types import TaskName
from app.core.worker import TaskWorker
from app.processors.mock import MockProcessor


def build_router(tmp_path: Path) -> tuple[TaskRouter, ResultStore, SourceRegistry, MockProcessor, MockProcessor]:
    registry = SourceRegistry(tmp_path / "sources.json")
    for index in range(1, 15):
        tasks: set[TaskName]
        enabled = True
        if index == 1:
            tasks = {TaskName.PLATE_RECOGNITION}
        elif 2 <= index <= 5:
            tasks = {TaskName.FIRE_SMOKE}
        elif 7 <= index <= 14:
            tasks = {TaskName.PLATE_RECOGNITION, TaskName.FIRE_SMOKE}
        else:
            tasks = set()
            enabled = False
        registry.create(
            SourceRecord(
                source_id=f"camera-{index:02d}",
                name=f"Camera {index}",
                enabled=enabled,
                tasks=tasks,
            )
        )
    results = ResultStore(100)
    fire = MockProcessor(TaskName.FIRE_SMOKE)
    plate = MockProcessor(TaskName.PLATE_RECOGNITION)
    workers = {
        TaskName.FIRE_SMOKE: TaskWorker(
            processor=fire,
            result_store=results,
            batch_size=8,
            max_wait_ms=10,
        ),
        TaskName.PLATE_RECOGNITION: TaskWorker(
            processor=plate,
            result_store=results,
            batch_size=8,
            max_wait_ms=10,
        ),
    }
    return TaskRouter(registry=registry, workers=workers, result_store=results), results, registry, fire, plate


def wait_for_results(store: ResultStore, count: int) -> list[dict]:
    deadline = time.time() + 2
    while time.time() < deadline:
        values = store.recent(limit=100)
        if len(values) >= count:
            return values
        time.sleep(0.01)
    return store.recent(limit=100)


def test_routes_one_round_to_expected_tasks(tmp_path: Path) -> None:
    router, results, _, fire, plate = build_router(tmp_path)
    router.start()
    try:
        summary = router.submit_round(
            frames=[np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(14)],
            source_ids=[f"camera-{index:02d}" for index in range(1, 15)],
            round_sequence=1,
        )
        values = wait_for_results(results, 21)
        assert summary == {
            "received_frames": 14,
            "accepted_sources": 13,
            "task_submissions": 21,
        }
        assert len(values) == 21
        assert sum(fire.batch_sizes) == 12
        assert sum(plate.batch_sizes) == 9
        assert max(fire.batch_sizes) <= 8
        assert max(plate.batch_sizes) <= 8
    finally:
        router.close()


def test_dynamic_disable_and_task_change_apply_without_restart(tmp_path: Path) -> None:
    router, results, registry, _, _ = build_router(tmp_path)
    router.start()
    try:
        registry.update("camera-01", enabled=False)
        registry.update("camera-02", tasks={TaskName.PLATE_RECOGNITION})
        summary = router.submit_round(
            frames=[np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)],
            source_ids=["camera-01", "camera-02"],
            round_sequence=2,
        )
        values = wait_for_results(results, 1)
        assert summary["task_submissions"] == 1
        assert values[0]["source_id"] == "camera-02"
        assert values[0]["task"] == "plate_recognition"
    finally:
        router.close()


def test_enabled_camera_with_no_tasks_uses_play_only_callback(tmp_path: Path) -> None:
    router, results, registry, _, _ = build_router(tmp_path)
    packets = []
    router.play_only_callback = packets.append
    registry.update("camera-06", enabled=True, tasks=set())
    router.start()
    try:
        summary = router.submit_round(
            frames=[np.zeros((16, 16, 3), dtype=np.uint8)],
            source_ids=["camera-06"],
            round_sequence=3,
        )

        assert summary["accepted_sources"] == 1
        assert summary["task_submissions"] == 0
        assert len(packets) == 1
        assert packets[0].metadata["assigned_tasks"] == []
        assert results.recent(limit=10) == []
        assert router.status()["play_only_frames"] == 1
    finally:
        router.close()


def test_fifty_sources_can_be_routed_without_global_camera_limit(tmp_path: Path) -> None:
    registry = SourceRegistry(tmp_path / "fifty.json")
    for index in range(1, 51):
        registry.create(
            SourceRecord(
                source_id=f"camera-{index:02d}",
                name=f"Camera {index}",
                tasks={TaskName.FIRE_SMOKE, TaskName.PLATE_RECOGNITION},
            )
        )
    results = ResultStore(200)
    fire = MockProcessor(TaskName.FIRE_SMOKE)
    plate = MockProcessor(TaskName.PLATE_RECOGNITION)
    router = TaskRouter(
        registry=registry,
        result_store=results,
        workers={
            TaskName.FIRE_SMOKE: TaskWorker(
                processor=fire, result_store=results, batch_size=8, max_wait_ms=5
            ),
            TaskName.PLATE_RECOGNITION: TaskWorker(
                processor=plate, result_store=results, batch_size=8, max_wait_ms=5
            ),
        },
    )
    router.start()
    try:
        summary = router.submit_round(
            frames=[np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(50)],
            source_ids=[f"camera-{index:02d}" for index in range(1, 51)],
            round_sequence=1,
        )
        values = wait_for_results(results, 100)
        assert summary["task_submissions"] == 100
        assert len(values) == 100
        assert max(fire.batch_sizes) <= 8
        assert max(plate.batch_sizes) <= 8
    finally:
        router.close()
