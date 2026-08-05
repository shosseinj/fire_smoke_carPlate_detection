from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable

from app.core.import_progress_store import ImportProgressRecord, ImportProgressStore


LOGGER = logging.getLogger("uvicorn.error")
ProgressCallback = Callable[[int, int, int, int], None]
ImportProcessor = Callable[[ProgressCallback], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class _ExcelJob:
    progress_id: int
    processor: ImportProcessor


class ExcelImportManager:
    """Bounded single-worker queue shared by observable Excel imports."""

    def __init__(self, progress_store: ImportProgressStore, *, queue_size: int = 4) -> None:
        self._progress_store = progress_store
        self._queue: queue.Queue[_ExcelJob | None] = queue.Queue(maxsize=max(1, queue_size))
        self._closed = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="excel-import", daemon=True)
        self._thread.start()

    def submit(
        self,
        import_type: str,
        filename: str,
        created_by: int | None,
        processor: ImportProcessor,
    ) -> ImportProgressRecord:
        with self._lock:
            if self._closed:
                raise RuntimeError("سرویس ورود اکسل در حال توقف است.")
            if self._queue.full():
                raise queue.Full
            record = self._progress_store.create(
                import_type, filename, created_by=created_by, status="queued"
            )
            try:
                self._queue.put_nowait(_ExcelJob(record.id, processor))
            except queue.Full:
                self._progress_store.delete(record.id)
                raise
            return record

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                self._process(job)
            finally:
                self._queue.task_done()

    def _process(self, job: _ExcelJob) -> None:
        self._progress_store.update(job.progress_id, status="running")

        def report(total: int, imported: int, skipped: int, failed: int) -> None:
            self._progress_store.update(
                job.progress_id,
                total_rows=total,
                imported_rows=imported,
                skipped_rows=skipped,
                failed_rows=failed,
            )

        try:
            result = job.processor(report)
            summary = result.get("summary", {})
            total = int(summary.get("total_rows", summary.get("total_data_rows", 0)))
            imported = int(summary.get("imported", summary.get("successful", summary.get("valid_rows", 0))))
            skipped = int(summary.get("skipped", 0))
            failed = int(summary.get("failed", summary.get("invalid_rows", 0)))
            report(total, imported, skipped, failed)
            terminal = "completed_with_errors" if failed else "completed"
            self._progress_store.update(job.progress_id, status=terminal, result=result)
        except Exception as exc:
            LOGGER.exception("Excel import job %s failed", job.progress_id)
            self._progress_store.update(
                job.progress_id,
                status="failed",
                error_message="پردازش فایل اکسل با خطای داخلی متوقف شد.",
                result={
                    "success": False,
                    "database_changed": False,
                    "message": "پردازش فایل اکسل ناموفق بود.",
                    "errors": [{"code": "internal_error", "type": type(exc).__name__}],
                },
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(None)
        self._thread.join()
