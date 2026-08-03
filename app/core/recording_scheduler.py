from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from app.core.recording_job_store import RecordingJob, RecordingJobStore


class RedisCoordinator(Protocol):
    def set(self, name: str, value: str, *, nx: bool = False, ex: int | None = None) -> Any: ...
    def get(self, name: str) -> Any: ...
    def delete(self, *names: str) -> int: ...
    def zadd(self, name: str, mapping: dict[str, float]) -> int: ...
    def zrem(self, name: str, *values: str) -> int: ...
    def zrangebyscore(self, name: str, minimum: float, maximum: float, *, start: int = 0, num: int | None = None) -> list[Any]: ...
    def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> Any: ...


class RecordingWorker(Protocol):
    def record(self, job: RecordingJob) -> "RecordingResult": ...


class UploadWorker(Protocol):
    def upload(self, job: RecordingJob, spool_path: str) -> "UploadResult": ...


@dataclass(frozen=True, slots=True)
class RecordingResult:
    spool_path: str
    playable: bool
    partial: bool = False
    warning: str | None = None
    content_type: str = "video/mp4"


@dataclass(frozen=True, slots=True)
class UploadResult:
    object_key: str
    size_bytes: int
    content_type: str = "video/mp4"


@dataclass(frozen=True, slots=True)
class ScheduledTask:
    task_id: str
    kind: str
    job_id: str
    run_at_utc: datetime
    attempt: int = 0


def deterministic_task_id(kind: str, job_id: str) -> str:
    digest = hashlib.sha256(f"recording:{kind}:{job_id}".encode()).hexdigest()[:24]
    return f"recording:{kind}:{digest}"


class RecordingScheduler:
    QUEUE_KEY = "recording:tasks"
    _COMPARE_DELETE = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"
    _COMPARE_EXPIRE = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('expire', KEYS[1], ARGV[2]) else return 0 end"

    @staticmethod
    def _task_key(task_id: str) -> str:
        return f"recording:task:{task_id}"

    def __init__(
        self,
        store: RecordingJobStore,
        redis: RedisCoordinator | None,
        *,
        lock_ttl_seconds: int = 300,
        global_concurrency: int = 1,
        max_retries: int = 3,
    ) -> None:
        if global_concurrency != 1:
            raise ValueError("initial recording global concurrency must be 1")
        self.store = store
        self.redis = redis
        self.lock_ttl_seconds = max(1, lock_ttl_seconds)
        self.global_concurrency = global_concurrency
        self.max_retries = max(0, max_retries)

    def schedule_job(self, job: RecordingJob) -> ScheduledTask:
        return self.enqueue("start", job.id, job.scheduled_start_utc)

    def enqueue(self, kind: str, job_id: str, run_at_utc: datetime, *, attempt: int = 0) -> ScheduledTask:
        run_at = run_at_utc.astimezone(timezone.utc)
        task = ScheduledTask(deterministic_task_id(kind, job_id), kind, job_id, run_at, attempt)
        if self.redis is not None:
            payload = json.dumps({"task_id": task.task_id, "kind": kind, "job_id": job_id, "run_at": run_at.isoformat(), "attempt": attempt}, separators=(",", ":"), sort_keys=True)
            self.redis.set(self._task_key(task.task_id), payload)
            self.redis.zadd(self.QUEUE_KEY, {task.task_id: run_at.timestamp()})
        return task

    def due(self, *, now: datetime | None = None, limit: int = 100) -> list[ScheduledTask]:
        if self.redis is None:
            return []
        instant = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        raw_items = self.redis.zrangebyscore(self.QUEUE_KEY, float("-inf"), instant.timestamp(), start=0, num=max(1, limit))
        tasks: list[ScheduledTask] = []
        for raw in raw_items:
            task_id = raw.decode() if isinstance(raw, bytes) else str(raw)
            payload = self.redis.get(self._task_key(task_id))
            if payload is None:
                self.redis.zrem(self.QUEUE_KEY, task_id)
                continue
            text = payload.decode() if isinstance(payload, bytes) else str(payload)
            data = json.loads(text)
            tasks.append(ScheduledTask(data["task_id"], data["kind"], data["job_id"], datetime.fromisoformat(data["run_at"]), int(data["attempt"])))
        return tasks

    def acknowledge(self, task: ScheduledTask) -> None:
        if self.redis is None:
            return
        self.redis.zrem(self.QUEUE_KEY, task.task_id)
        self.redis.delete(self._task_key(task.task_id))

    def retry(self, task: ScheduledTask, *, now: datetime | None = None) -> ScheduledTask | None:
        next_attempt = task.attempt + 1
        if next_attempt > self.max_retries:
            return None
        delay = timedelta(seconds=min(300, 2 ** next_attempt))
        return self.enqueue(task.kind, task.job_id, (now or datetime.now(timezone.utc)) + delay, attempt=next_attempt)

    def accept_recording_result(self, job_id: str, result: RecordingResult, *, now: datetime | None = None) -> ScheduledTask:
        if not result.playable:
            self.store.transition(job_id, "failed", error=result.warning or "خروجی ضبط قابل پخش نیست", spool_path=result.spool_path, now=now)
            raise ValueError("recording worker did not produce a playable file")
        job = self.store.transition(
            job_id,
            "finalizing",
            warning=result.warning,
            spool_path=result.spool_path,
            content_type=result.content_type,
            is_partial=result.partial,
            now=now,
        )
        return self.enqueue("upload", job.id, now or datetime.now(timezone.utc))

    def accept_upload_result(self, job_id: str, result: UploadResult, *, now: datetime | None = None) -> RecordingJob:
        job = self.store.require(job_id)
        if job.status == "finalizing":
            job = self.store.transition(job.id, "uploading", now=now)
        target = "partial" if job.is_partial or job.cancel_requested_at_utc is not None else "completed"
        return self.store.transition(
            job.id,
            target,
            object_key=result.object_key,
            size_bytes=result.size_bytes,
            content_type=result.content_type,
            now=now,
        )

    def acquire_locks(self, job: RecordingJob, owner: str) -> bool:
        if self.redis is None:
            return True
        global_key = "recording:lock:global:0"
        source_key = f"recording:lock:source:{hashlib.sha256(job.source_uri.encode()).hexdigest()}"
        if not self.redis.set(global_key, owner, nx=True, ex=self.lock_ttl_seconds):
            return False
        if not self.redis.set(source_key, owner, nx=True, ex=self.lock_ttl_seconds):
            self._release_if_owned(global_key, owner)
            return False
        return True

    def release_locks(self, job: RecordingJob, owner: str) -> None:
        if self.redis is None:
            return
        self._release_if_owned("recording:lock:global:0", owner)
        source_key = f"recording:lock:source:{hashlib.sha256(job.source_uri.encode()).hexdigest()}"
        self._release_if_owned(source_key, owner)

    def renew_locks(self, job: RecordingJob, owner: str) -> bool:
        if self.redis is None:
            return True
        global_key = "recording:lock:global:0"
        source_key = f"recording:lock:source:{hashlib.sha256(job.source_uri.encode()).hexdigest()}"
        return self._compare_expire(global_key, owner) and self._compare_expire(source_key, owner)

    def reconcile(self, *, now: datetime | None = None) -> list[ScheduledTask]:
        instant = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        tasks: list[ScheduledTask] = []
        for job in self.store.recoverable():
            if job.status in {"recording", "finalizing"}:
                if job.spool_path:
                    if job.status == "recording":
                        job = self.store.transition(job.id, "finalizing", warning="فرایند پس از راه‌اندازی مجدد بازیابی شد", now=instant)
                    tasks.append(self.enqueue("upload", job.id, instant))
                else:
                    self.store.transition(job.id, "failed", error="فرایند ضبط با راه‌اندازی مجدد قطع شد", now=instant)
            elif job.status == "uploading":
                tasks.append(self.enqueue("upload", job.id, instant))
            elif job.scheduled_end_utc <= instant:
                self.store.transition(job.id, "failed", error="زمان‌بندی در هنگام توقف سرویس منقضی شد", now=instant)
            else:
                tasks.append(self.enqueue("start", job.id, max(instant, job.scheduled_start_utc)))
        return tasks

    def _release_if_owned(self, key: str, owner: str) -> None:
        if self.redis is None:
            return
        compare_delete = getattr(self.redis, "compare_and_delete", None)
        if compare_delete is not None:
            compare_delete(key, owner)
            return
        evaluator = getattr(self.redis, "eval", None)
        if evaluator is None:
            raise RuntimeError("Redis coordinator lacks atomic compare-delete support")
        evaluator(self._COMPARE_DELETE, 1, key, owner)

    def _compare_expire(self, key: str, owner: str) -> bool:
        if self.redis is None:
            return True
        compare_expire = getattr(self.redis, "compare_and_expire", None)
        if compare_expire is not None:
            return bool(compare_expire(key, owner, self.lock_ttl_seconds))
        evaluator = getattr(self.redis, "eval", None)
        if evaluator is None:
            raise RuntimeError("Redis coordinator lacks atomic compare-expire support")
        return bool(evaluator(self._COMPARE_EXPIRE, 1, key, owner, self.lock_ttl_seconds))
