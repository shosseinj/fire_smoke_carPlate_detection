from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.core.recording_job_store import RecordingJob
from app.core.recording_scheduler import RecordingScheduler, deterministic_task_id


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.sorted: dict[str, dict[str, float]] = {}
        self.expirations: dict[str, int] = {}

    def set(self, name: str, value: str, *, nx: bool = False, ex: int | None = None):
        del ex
        if nx and name in self.values:
            return False
        self.values[name] = value
        return True

    def get(self, name: str):
        return self.values.get(name)

    def delete(self, *names: str) -> int:
        removed = 0
        for name in names:
            removed += int(self.values.pop(name, None) is not None)
        return removed

    def zadd(self, name: str, mapping: dict[str, float]) -> int:
        self.sorted.setdefault(name, {}).update(mapping)
        return len(mapping)

    def zrem(self, name: str, *values: str) -> int:
        target = self.sorted.setdefault(name, {})
        return sum(int(target.pop(value, None) is not None) for value in values)

    def zrangebyscore(self, name: str, minimum: float, maximum: float, *, start: int = 0, num: int | None = None):
        items = [value for value, score in sorted(self.sorted.get(name, {}).items(), key=lambda item: item[1]) if minimum <= score <= maximum]
        return items[start:] if num is None else items[start : start + num]

    def compare_and_delete(self, key: str, owner: str) -> int:
        if self.values.get(key) != owner:
            return 0
        del self.values[key]
        return 1

    def compare_and_expire(self, key: str, owner: str, seconds: int) -> int:
        if self.values.get(key) != owner:
            return 0
        self.expirations[key] = seconds
        return 1


class FakeStore:
    def recoverable(self):
        return []


def _job(source: str = "rtsp://camera/1") -> RecordingJob:
    start = datetime.now(timezone.utc) + timedelta(minutes=1)
    return RecordingJob("11111111-1111-1111-1111-111111111111", source, 1, None, "scheduled", start, start + timedelta(minutes=5))


def test_scheduled_queue_uses_deterministic_ids_and_due_time() -> None:
    redis = FakeRedis()
    scheduler = RecordingScheduler(FakeStore(), redis)  # type: ignore[arg-type]
    job = _job()

    first = scheduler.schedule_job(job)
    second = scheduler.schedule_job(job)

    assert first.task_id == second.task_id == deterministic_task_id("start", job.id)
    assert scheduler.due(now=job.scheduled_start_utc - timedelta(seconds=1)) == []
    assert [task.task_id for task in scheduler.due(now=job.scheduled_start_utc)] == [first.task_id]
    scheduler.acknowledge(first)
    assert scheduler.due(now=job.scheduled_start_utc) == []


def test_queue_identity_is_idempotent_when_reconcile_timestamp_changes() -> None:
    redis = FakeRedis()
    scheduler = RecordingScheduler(FakeStore(), redis)  # type: ignore[arg-type]
    job = _job()
    first = scheduler.enqueue("upload", job.id, job.scheduled_start_utc)
    second = scheduler.enqueue("upload", job.id, job.scheduled_start_utc + timedelta(seconds=5))

    assert first.task_id == second.task_id
    assert list(redis.sorted[scheduler.QUEUE_KEY]) == [first.task_id]
    assert scheduler.due(now=job.scheduled_start_utc) == []
    due = scheduler.due(now=job.scheduled_start_utc + timedelta(seconds=5))
    assert len(due) == 1 and due[0].task_id == first.task_id


def test_global_and_per_source_locks_are_bounded_and_owner_safe() -> None:
    redis = FakeRedis()
    scheduler = RecordingScheduler(FakeStore(), redis)  # type: ignore[arg-type]
    first = _job()
    second = _job("rtsp://camera/2")

    assert scheduler.acquire_locks(first, "worker-a") is True
    assert scheduler.acquire_locks(second, "worker-b") is False
    scheduler.release_locks(first, "wrong-worker")
    assert scheduler.acquire_locks(second, "worker-b") is False
    scheduler.release_locks(first, "worker-a")
    assert scheduler.acquire_locks(second, "worker-b") is True


def test_lock_lease_renews_both_keys_and_foreign_owner_is_independent() -> None:
    redis = FakeRedis()
    scheduler = RecordingScheduler(FakeStore(), redis, lock_ttl_seconds=9)  # type: ignore[arg-type]
    job = _job()
    assert scheduler.acquire_locks(job, "worker-a") is True
    assert scheduler.renew_locks(job, "worker-a") is True
    assert len(redis.expirations) == 2 and set(redis.expirations.values()) == {9}

    global_key = "recording:lock:global:0"
    redis.values[global_key] = "worker-b"
    assert scheduler.renew_locks(job, "worker-a") is False
    scheduler.release_locks(job, "worker-a")
    assert redis.values[global_key] == "worker-b"


def test_injected_redis_without_atomic_operations_is_rejected_safely() -> None:
    class UnsafeRedis(FakeRedis):
        compare_and_delete = None  # type: ignore[assignment]
        compare_and_expire = None  # type: ignore[assignment]

    scheduler = RecordingScheduler(FakeStore(), UnsafeRedis())  # type: ignore[arg-type]
    job = _job()
    assert scheduler.acquire_locks(job, "worker-a") is True
    import pytest
    with pytest.raises(RuntimeError, match="atomic compare-expire"):
        scheduler.renew_locks(job, "worker-a")


def test_retry_is_bounded_and_exponential() -> None:
    scheduler = RecordingScheduler(FakeStore(), FakeRedis(), max_retries=2)  # type: ignore[arg-type]
    task = scheduler.schedule_job(_job())
    now = datetime.now(timezone.utc)

    retry_one = scheduler.retry(task, now=now)
    assert retry_one is not None and retry_one.attempt == 1
    retry_two = scheduler.retry(retry_one, now=now)
    assert retry_two is not None and retry_two.attempt == 2
    assert scheduler.retry(retry_two, now=now) is None
