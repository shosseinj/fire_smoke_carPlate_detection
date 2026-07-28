from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from app.core.result_store import ResultStore
from app.core.router import TaskRouter
from app.core.source_registry import SourceRecord, SourceRegistry
from app.core.types import TaskName
from app.core.worker import TaskWorker
from app.processors.mock import MockProcessor


def build_router(registry: SourceRegistry) -> tuple[TaskRouter, ResultStore, SourceRegistry, MockProcessor, MockProcessor]:
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
                source_uri=f"camera-{index:02d}",
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


def test_routes_one_round_to_expected_tasks(source_registry: SourceRegistry) -> None:
    router, results, _, fire, plate = build_router(source_registry)
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


def test_dynamic_disable_and_task_change_apply_without_restart(source_registry: SourceRegistry) -> None:
    router, results, registry, _, _ = build_router(source_registry)
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


def test_enabled_camera_with_no_tasks_uses_play_only_callback(source_registry: SourceRegistry) -> None:
    router, results, registry, _, _ = build_router(source_registry)
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


def test_enabled_camera_with_tasks_is_broadcast_before_worker_result() -> None:
    packets = []
    submitted = []
    source = SimpleNamespace(
        enabled=True,
        tasks={TaskName.PLATE_RECOGNITION},
    )
    registry = SimpleNamespace(
        get=lambda source_id: source if source_id == "camera-01" else None,
        enabled_source_ids=lambda: ["camera-01"],
    )
    worker = SimpleNamespace(
        submit=lambda packet: submitted.append(packet) is None,
        status=lambda: {},
    )
    router = TaskRouter(
        registry=registry,
        workers={TaskName.PLATE_RECOGNITION: worker},
        result_store=ResultStore(10),
        play_only_callback=packets.append,
    )

    summary = router.submit_round(
        frames=[np.zeros((16, 16, 3), dtype=np.uint8)],
        source_ids=["camera-01"],
        round_sequence=4,
    )

    assert summary["accepted_sources"] == 1
    assert summary["task_submissions"] == 1
    assert len(packets) == 1
    assert submitted == packets
    assert packets[0].metadata["assigned_tasks"] == ["plate_recognition"]
    assert router.status()["broadcast_frames"] == 1
    assert router.status()["play_only_frames"] == 0


def test_disabled_task_processing_keeps_ai_submissions_at_zero() -> None:
    submitted = []
    source = SimpleNamespace(
        enabled=True,
        tasks={TaskName.FACE_RECOGNITION},
    )
    registry = SimpleNamespace(
        get=lambda source_id: source if source_id == "camera-01" else None,
        enabled_source_ids=lambda: ["camera-01"],
    )
    worker = SimpleNamespace(
        submit=lambda packet: submitted.append(packet) is None,
        status=lambda: {},
    )
    router = TaskRouter(
        registry=registry,
        workers={TaskName.FACE_RECOGNITION: worker},
        result_store=ResultStore(10),
        task_processing_enabled_callback=lambda: False,
    )

    summary = router.submit_round(
        frames=[np.zeros((16, 16, 3), dtype=np.uint8)],
        source_ids=["camera-01"],
        round_sequence=1,
    )

    assert summary["task_submissions"] == 0
    assert submitted == []
    assert router.status()["task_processing_enabled"] is False
    assert router.status()["task_processing_paused_frames"] == 1


def test_fifty_sources_can_be_routed_without_global_camera_limit(source_registry: SourceRegistry) -> None:
    registry = source_registry
    for index in range(1, 51):
        registry.create(
            SourceRecord(
                source_uri=f"camera-{index:02d}",
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


def test_face_recognition_has_an_independent_worker(source_registry: SourceRegistry) -> None:
    registry = source_registry
    registry.create(
        SourceRecord(
            source_uri="face-camera",
            name="Face camera",
            tasks={TaskName.FACE_RECOGNITION},
        )
    )
    results = ResultStore(10)
    face = MockProcessor(TaskName.FACE_RECOGNITION)
    router = TaskRouter(
        registry=registry,
        result_store=results,
        workers={
            TaskName.FACE_RECOGNITION: TaskWorker(
                processor=face,
                result_store=results,
                batch_size=8,
                max_wait_ms=5,
            )
        },
    )
    router.start()
    try:
        summary = router.submit_round(
            frames=[np.zeros((16, 16, 3), dtype=np.uint8)],
            source_ids=["face-camera"],
            round_sequence=1,
        )
        values = wait_for_results(results, 1)
        assert summary["task_submissions"] == 1
        assert values[0]["task"] == "face_recognition"
        assert face.batch_sizes == [1]
    finally:
        router.close()
