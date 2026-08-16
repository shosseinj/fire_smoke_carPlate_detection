from __future__ import annotations

import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from app.core.recording_coordinator import RecordingCoordinator, RecordingCoordinatorShutdownError
from app.core.recording_executor import RecordingResult
from app.core.recording_job_store import RecordingJob
from app.core.recording_scheduler import RecordingScheduler
from app.runtime import Runtime


def _job(status: str = "recording", *, spool_path: str | None = None) -> RecordingJob:
    now = datetime.now(timezone.utc)
    return RecordingJob("11111111-1111-1111-1111-111111111111", "rtsp://camera/1", 1, None,
                        status, now, now + timedelta(minutes=5), spool_path=spool_path)


class _Store:
    def __init__(self, job: RecordingJob) -> None:
        self.job = job

    def get(self, job_id: str) -> RecordingJob | None:
        return self.job if self.job.id == job_id else None

    def require(self, job_id: str) -> RecordingJob:
        assert job_id == self.job.id
        return self.job

    def recoverable(self) -> list[RecordingJob]:
        return [] if self.job.status in {"partial", "completed", "failed", "cancelled"} else [self.job]

    def transition(self, job_id: str, target: str, **fields: object) -> RecordingJob:
        del job_id
        fields.pop("now", None)
        if fields.get("warning") is None:
            fields.pop("warning", None)
        self.job = replace(self.job, status=target, **fields)
        return self.job

    def cancel(self, job_id: str) -> RecordingJob:
        del job_id
        self.job = replace(self.job, cancel_requested_at_utc=datetime.now(timezone.utc), is_partial=True)
        return self.job

    def recover_spool(self, job_id: str, spool_path: str, **_kwargs: object) -> RecordingJob:
        del job_id
        self.job = replace(self.job, status="finalizing", spool_path=spool_path, is_partial=True)
        return self.job


class _Scheduler:
    lock_ttl_seconds = 3
    global_concurrency = 1

    def __init__(self, store: _Store) -> None:
        self.store = store
        self.enqueued: list[tuple[str, str]] = []
        self.reconciles = 0
        self.due_failures = 0

    def reconcile(self) -> list[object]:
        self.reconciles += 1
        return []

    def due(self, *, limit: int) -> list[object]:
        del limit
        if self.due_failures:
            self.due_failures -= 1
            raise ConnectionError("redis unavailable")
        return []

    def enqueue(self, kind: str, job_id: str, _when: datetime) -> object:
        self.enqueued.append((kind, job_id))
        return object()

    def accept_upload_result(self, job_id: str, result: object) -> RecordingJob:
        current = self.store.require(job_id)
        target = "partial" if current.is_partial or current.cancel_requested_at_utc else "completed"
        return self.store.transition(job_id, target, object_key=result.object_key, size_bytes=result.size_bytes)


class _Spool:
    def __init__(self) -> None:
        self.failures = 0

    def cleanup_failed(self, _protected: set[Path]) -> None:
        if self.failures:
            self.failures -= 1
            raise OSError("transient spool failure")

    def status(self) -> object:
        return SimpleNamespace(high_water_exceeded=False, used_percent=1.0)


class _Executor:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.cancel_seen = False

    def execute(self, job: object, cancelled: threading.Event) -> RecordingResult:
        path = self.finalized_path(job.job_id)
        path.write_bytes(b"playable")
        while not cancelled.wait(0.01):
            pass
        self.cancel_seen = True
        return RecordingResult("cancelled_partial", path, False)

    def finalized_path(self, job_id: str) -> Path:
        import hashlib
        return self.root / f"{hashlib.sha256(job_id.encode()).hexdigest()}.mp4"

    def recoverable_path(self, job_id: str) -> Path | None:
        path = self.finalized_path(job_id)
        return path if path.is_file() and path.stat().st_size else None


def test_cancellation_during_recording_persists_and_schedules_partial_upload(tmp_path: Path) -> None:
    store = _Store(_job())
    scheduler = _Scheduler(store)
    executor = _Executor(tmp_path)
    coordinator = RecordingCoordinator(store, scheduler, executor, SimpleNamespace(), _Spool())  # type: ignore[arg-type]
    thread = threading.Thread(target=coordinator._record, args=(store.job,))
    thread.start()
    while coordinator._active_job is None:
        time.sleep(0.005)
    coordinator.cancel(store.job.id)
    thread.join(timeout=1)

    assert executor.cancel_seen is True
    assert store.job.status == "finalizing" and store.job.is_partial is True
    assert scheduler.enqueued == [("upload", store.job.id)]


def test_cancellation_during_upload_records_object_as_partial_without_orphan(tmp_path: Path) -> None:
    path = tmp_path / "recording.mp4"
    path.write_bytes(b"playable")
    store = _Store(_job("uploading", spool_path=str(path)))
    scheduler = _Scheduler(store)
    entered = threading.Event()
    release = threading.Event()

    class Storage:
        def upload_finalized(self, _job_id: str, _path: Path, **_options: object) -> object:
            entered.set()
            release.wait(1)
            return SimpleNamespace(object_key="recordings/job.mp4", size=8)

    coordinator = RecordingCoordinator(store, scheduler, _Executor(tmp_path), Storage(), _Spool())  # type: ignore[arg-type]
    thread = threading.Thread(target=coordinator._upload, args=(store.job,))
    thread.start()
    assert entered.wait(1)
    coordinator.cancel(store.job.id)
    release.set()
    thread.join(timeout=1)

    assert store.job.status == "partial" and store.job.object_key == "recordings/job.mp4"
    assert not path.exists()


def test_loop_survives_redis_and_spool_outages_and_reconciles(tmp_path: Path) -> None:
    store = _Store(_job("scheduled"))
    scheduler = _Scheduler(store)
    scheduler.due_failures = 1
    spool = _Spool()
    spool.failures = 1
    coordinator = RecordingCoordinator(store, scheduler, _Executor(tmp_path), SimpleNamespace(initialize_private_bucket=lambda: None),
                                       spool, poll_seconds=0.01, reconcile_seconds=0.02)  # type: ignore[arg-type]
    coordinator.start()
    time.sleep(0.25)
    assert coordinator._thread is not None and coordinator._thread.is_alive()
    assert scheduler.reconciles >= 2
    assert coordinator.close() is True


def test_restart_discovers_deterministic_finalized_spool_and_enqueues_upload(tmp_path: Path) -> None:
    store = _Store(_job("recording"))
    scheduler = _Scheduler(store)
    executor = _Executor(tmp_path)
    executor.finalized_path(store.job.id).write_bytes(b"playable")
    coordinator = RecordingCoordinator(store, scheduler, executor, SimpleNamespace(), _Spool())  # type: ignore[arg-type]
    coordinator._recover_spool_artifacts()

    assert store.job.status == "finalizing" and store.job.is_partial is True
    assert scheduler.enqueued == [("upload", store.job.id)]


def test_combined_spool_recovery_and_scheduler_reconcile_leave_one_upload_task(tmp_path: Path) -> None:
    class Redis:
        def __init__(self) -> None:
            self.values: dict[str, str] = {}
            self.sorted: dict[str, dict[str, float]] = {}

        def set(self, name: str, value: str, **_kwargs: object) -> bool:
            self.values[name] = value
            return True

        def get(self, name: str) -> str | None:
            return self.values.get(name)

        def delete(self, *names: str) -> int:
            return sum(int(self.values.pop(name, None) is not None) for name in names)

        def zadd(self, name: str, mapping: dict[str, float]) -> int:
            self.sorted.setdefault(name, {}).update(mapping)
            return len(mapping)

        def zrem(self, name: str, *values: str) -> int:
            return sum(int(self.sorted.setdefault(name, {}).pop(value, None) is not None) for value in values)

        def zrangebyscore(self, name: str, minimum: float, maximum: float, **_kwargs: object) -> list[str]:
            return [value for value, score in self.sorted.get(name, {}).items() if minimum <= score <= maximum]

    store = _Store(_job("recording"))
    redis = Redis()
    scheduler = RecordingScheduler(store, redis)  # type: ignore[arg-type]
    executor = _Executor(tmp_path)
    executor.finalized_path(store.job.id).write_bytes(b"playable")
    coordinator = RecordingCoordinator(store, scheduler, executor, SimpleNamespace(), _Spool())  # type: ignore[arg-type]

    coordinator._recover_spool_artifacts()
    scheduler.reconcile()

    members = redis.sorted[scheduler.QUEUE_KEY]
    assert len(members) == 1
    assert next(iter(members)).startswith("recording:upload:")


def _runtime_for_shutdown(coordinator: object, events: list[str]) -> SimpleNamespace:
    def component(name: str) -> object:
        return SimpleNamespace(close=lambda: events.append(name))

    return SimpleNamespace(
        broadcast=component("broadcast"), personnel_zip_imports=component("zip"),
        excel_imports=component("excel"),
        model_conversions=component("models"), recording_coordinator=coordinator,
        recording_redis=component("redis"), media_preview=component("preview"),
        live_branch=component("live"), static_video_ingestor=component("static"),
        video_ingestor=component("video"), static_video_lifecycle=component("lifecycle"),
        router=component("router"), fire_smoke_logs=component("fire"), plate_logs=component("plate"),
        human_logs=component("human"), registry=component("registry"),
        database=SimpleNamespace(dispose=lambda: events.append("database")), recording_error=None,
    )


def test_runtime_shutdown_stops_recorder_before_redis_and_live_branch() -> None:
    events: list[str] = []
    runtime = _runtime_for_shutdown(SimpleNamespace(close=lambda: events.append("recording")), events)
    Runtime.close(runtime)  # type: ignore[arg-type]
    assert events.index("recording") < events.index("redis") < events.index("live")
    assert events[-1] == "database"


def test_coordinator_close_raises_when_worker_remains_alive(tmp_path: Path) -> None:
    coordinator = RecordingCoordinator(_Store(_job()), _Scheduler(_Store(_job())), _Executor(tmp_path),
                                       SimpleNamespace(), _Spool(), shutdown_timeout_seconds=0.01)  # type: ignore[arg-type]
    blocker = threading.Event()
    coordinator._thread = threading.Thread(target=lambda: blocker.wait(1), daemon=True)
    coordinator._thread.start()
    try:
        try:
            coordinator.close()
            assert False, "close must report a live worker"
        except RecordingCoordinatorShutdownError:
            pass
        assert coordinator.status()["last_error"].startswith("shutdown:")
    finally:
        blocker.set()
        coordinator._thread.join(timeout=1)


def test_runtime_preserves_recording_dependencies_when_coordinator_is_stuck() -> None:
    events: list[str] = []

    def fail_close() -> None:
        events.append("recording")
        raise RecordingCoordinatorShutdownError("stuck")

    runtime = _runtime_for_shutdown(SimpleNamespace(close=fail_close), events)
    try:
        Runtime.close(runtime)  # type: ignore[arg-type]
        assert False, "runtime close must surface coordinator shutdown failure"
    except RuntimeError as exc:
        assert "still active" in str(exc)
    assert runtime.recording_error == "RecordingCoordinatorShutdownError"
    assert "redis" not in events and "live" not in events and "database" not in events
    assert "router" in events and "registry" in events


def test_runtime_configures_bounded_redis_network_timeouts() -> None:
    source = Path("app/runtime.py").read_text(encoding="utf-8")
    assert "socket_connect_timeout=3.0" in source
    assert "socket_timeout=5.0" in source
    assert "retry_on_timeout=False" in source

import pytest

pytestmark = pytest.mark.streaming
