from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.recording_executor import RecordingExecution, ScheduledRecordingExecutor
from app.core.recording_job_store import RecordingJob, RecordingJobStore
from app.core.recording_scheduler import RecordingScheduler, UploadResult
from app.core.recording_storage import LocalSpoolLifecycle, RecordingStorageService


LOGGER = logging.getLogger(__name__)


class RecordingCoordinatorShutdownError(RuntimeError):
    pass


class RecordingCoordinator:
    """Single bounded recording worker coordinating durable schedule and storage."""

    def __init__(self, store: RecordingJobStore, scheduler: RecordingScheduler, executor: ScheduledRecordingExecutor,
                 storage: RecordingStorageService, spool: LocalSpoolLifecycle, *, poll_seconds: float = 0.5,
                 reconcile_seconds: float = 30.0, shutdown_timeout_seconds: float = 45.0) -> None:
        self.store = store
        self.scheduler = scheduler
        self.executor = executor
        self.storage = storage
        self.spool = spool
        self.poll_seconds = max(0.1, poll_seconds)
        self.reconcile_seconds = max(self.poll_seconds, reconcile_seconds)
        self.shutdown_timeout_seconds = max(0.01, shutdown_timeout_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_job: str | None = None
        self._cancel = threading.Event()
        self._last_error: str | None = None
        self._wake = threading.Event()
        self._last_reconcile = 0.0

    def start(self) -> None:
        try:
            self.storage.initialize_private_bucket()
        except Exception as exc:
            self._record_error("storage_initialize", exc)
        self._thread = threading.Thread(target=self._run, name="recording-coordinator", daemon=True)
        self._thread.start()
        self.request_reconcile()

    def close(self) -> bool:
        self._stop.set()
        self._cancel.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=self.shutdown_timeout_seconds)
            if self._thread.is_alive():
                error = RecordingCoordinatorShutdownError("recording coordinator did not stop within its bounded timeout")
                self._record_error("shutdown", error)
                raise error
        return True

    def cancel(self, job_id: str) -> RecordingJob:
        job = self.store.cancel(job_id)
        if self._active_job == job_id:
            self._cancel.set()
        return job

    def request_reconcile(self) -> None:
        self._last_reconcile = 0.0
        self._wake.set()

    def status(self) -> dict[str, Any]:
        try:
            spool = self.spool.status()
            spool_percent: float | None = spool.used_percent
            high_water: bool | None = spool.high_water_exceeded
        except Exception as exc:
            self._record_error("spool_status", exc)
            spool_percent = None
            high_water = None
        return {"enabled": True, "running": bool(self._thread and self._thread.is_alive()),
                "active_job": self._active_job, "last_error": self._last_error,
                "spool_used_percent": spool_percent, "spool_high_water": high_water,
                "global_concurrency": 1}

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self.poll_seconds)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                self.spool.cleanup_failed(self._protected_paths())
            except Exception as exc:
                self._record_error("spool_cleanup", exc)
            if time.monotonic() - self._last_reconcile >= self.reconcile_seconds:
                try:
                    self._recover_spool_artifacts()
                    self.scheduler.reconcile()
                    self._last_reconcile = time.monotonic()
                except Exception as exc:
                    self._record_error("reconcile", exc)
            try:
                tasks = self.scheduler.due(limit=1)
            except Exception as exc:
                self._record_error("scheduler_due", exc)
                continue
            for task in tasks:
                try:
                    self._process(task)
                except Exception as exc:
                    self._record_error("task_loop", exc)

    def _process(self, task: Any) -> None:
        job = self.store.get(task.job_id)
        if job is None or job.status in {"cancelled", "completed", "partial", "failed"}:
            self.scheduler.acknowledge(task)
            return
        owner = f"coordinator:{task.task_id}"
        try:
            if not self.scheduler.acquire_locks(job, owner):
                return
        except Exception as exc:
            self._record_error("lock_acquire", exc)
            return
        lease_stop = threading.Event()
        lease = threading.Thread(target=self._renew_lease, args=(job, owner, lease_stop), daemon=True)
        lease.start()
        try:
            if task.kind == "upload":
                self._upload(job)
            else:
                self._record(job)
            self.scheduler.acknowledge(task)
            self._last_error = None
        except Exception as exc:
            self._last_error = type(exc).__name__
            LOGGER.warning("RECORDING_TASK_FAILED job=%s kind=%s error=%s", job.id, task.kind, type(exc).__name__)
            self.scheduler.acknowledge(task)
            retry = self.scheduler.retry(task)
            if retry is None and self.store.require(job.id).status not in {"cancelled", "completed", "partial", "failed"}:
                self.store.transition(job.id, "failed", error="سرویس ضبط پس از تلاش‌های محدود ناموفق بود")
        finally:
            lease_stop.set()
            lease.join(timeout=self.poll_seconds + 1.0)
            try:
                self.scheduler.release_locks(job, owner)
            except Exception as exc:
                self._record_error("lock_release", exc)

    def _record(self, job: RecordingJob) -> None:
        if self.spool.status().high_water_exceeded:
            self.store.transition(job.id, "failed", error="فضای موقت ضبط از حد مجاز عبور کرده است")
            return
        if job.status == "scheduled":
            job = self.store.transition(job.id, "queued")
        if job.status == "queued":
            job = self.store.transition(job.id, "recording")
        self._active_job = job.id
        self._cancel = threading.Event()
        if job.cancel_requested_at_utc is not None:
            self._cancel.set()
        try:
            result = self.executor.execute(
                RecordingExecution(job.id, job.source_uri, job.scheduled_start_utc, job.scheduled_end_utc), self._cancel
            )
            if result.local_path is None:
                current = self.store.require(job.id)
                if current.status != "cancelled":
                    self.store.transition(job.id, "cancelled")
                return
            partial = result.status in {"cancelled_partial", "interrupted_partial"}
            current = self.store.require(job.id)
            partial = partial or current.cancel_requested_at_utc is not None
            finalized = self.store.transition(job.id, "finalizing", spool_path=str(result.local_path), is_partial=partial,
                                              warning="ضبط به‌صورت قابل پخش اما ناقص پایان یافت" if partial else None)
            self.scheduler.enqueue("upload", finalized.id, datetime.now(timezone.utc))
        finally:
            self._active_job = None

    def _upload(self, job: RecordingJob) -> None:
        if not job.spool_path:
            raise RuntimeError("recording spool path is missing")
        if job.status == "finalizing":
            job = self.store.transition(job.id, "uploading")
        path = Path(job.spool_path)
        stored = self.storage.upload_finalized(
            job.id, path, retention_days=job.retention_days,
            metadata={
                "quality-preset": job.quality_preset,
                "output-width": str(job.output_width or "source"),
                "output-height": str(job.output_height or "source"),
                "output-fps": str(job.output_fps or "source"),
                "output-bitrate-bps": str(job.output_bitrate_bps or ""),
            },
        )
        self.scheduler.accept_upload_result(job.id, UploadResult(stored.object_key, stored.size))
        path.unlink(missing_ok=True)

    def _protected_paths(self) -> set[Path]:
        return {Path(job.spool_path) for job in self.store.recoverable() if job.spool_path}

    def _recover_spool_artifacts(self) -> None:
        now = datetime.now(timezone.utc)
        for job in self.store.recoverable():
            path = self.executor.recoverable_path(job.id)
            if path is None:
                continue
            recovered = self.store.recover_spool(job.id, str(path), now=now)
            if recovered.status in {"finalizing", "uploading"}:
                self.scheduler.enqueue("upload", recovered.id, now)

    def _renew_lease(self, job: RecordingJob, owner: str, stopped: threading.Event) -> None:
        interval = max(0.2, self.scheduler.lock_ttl_seconds / 3.0)
        while not stopped.wait(interval):
            try:
                if not self.scheduler.renew_locks(job, owner):
                    self._record_error("lock_lease_lost", RuntimeError("recording lock lease lost"))
                    self._cancel.set()
                    return
            except Exception as exc:
                self._record_error("lock_renew", exc)
                self._cancel.set()
                return

    def _record_error(self, operation: str, exc: Exception) -> None:
        self._last_error = f"{operation}:{type(exc).__name__}"
        LOGGER.warning("RECORDING_COORDINATOR_TRANSIENT operation=%s error=%s", operation, type(exc).__name__)
