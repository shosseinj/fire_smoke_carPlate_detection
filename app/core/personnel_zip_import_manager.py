from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any

from app.core.import_progress_store import ImportProgressRecord, ImportProgressStore
from app.core.personnel_store import PersonnelStore


LOGGER = logging.getLogger("uvicorn.error")


@dataclass(frozen=True, slots=True)
class _Job:
    progress_id: int
    data: bytes
    enable_cropping: bool


def format_zip_result(filename: str, result: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "total_images_in_zip": result.get("total_images_in_zip", 0),
        "total_persons": result.get("created_personnel", 0),
        "total_images_saved": result.get("created_images", 0),
        "total_errors": len(result.get("errors", [])),
        "total_failed": result.get("total_failed", 0),
        "qdrant_enrolled_count": result.get("qdrant_enrolled", 0),
        "face_stats": result.get("face_counts", {}),
    }
    return {
        "success": not result.get("errors"),
        "filename": filename,
        "message": (
            f"تعداد {summary['total_images_in_zip']} تصویر پردازش شد: "
            f"{summary['total_images_saved']} ذخیره شد، "
            f"{summary['qdrant_enrolled_count']} در پایگاه برداری ثبت شد، "
            f"{summary['total_failed']} ناموفق"
        ),
        "summary": summary,
        "details": result.get("image_details", []),
        "errors": result.get("errors", []),
    }


class PersonnelZipImportManager:
    """Bounded single-worker queue for observable personnel ZIP imports."""

    def __init__(
        self,
        personnel_store: PersonnelStore,
        progress_store: ImportProgressStore,
        face_processor: Any | None,
        *,
        queue_size: int = 4,
    ) -> None:
        self._personnel_store = personnel_store
        self._progress_store = progress_store
        self._face_processor = face_processor
        self._queue: queue.Queue[_Job | None] = queue.Queue(maxsize=max(1, queue_size))
        self._closed = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="personnel-zip-import", daemon=True)
        self._thread.start()

    def submit(self, data: bytes, filename: str, enable_cropping: bool, created_by: int) -> ImportProgressRecord:
        with self._lock:
            if self._closed:
                raise RuntimeError("سرویس ورود فایل ZIP در حال توقف است.")
            if self._queue.full():
                raise queue.Full
            record = self._progress_store.create(
                "personnel_zip", filename, created_by=created_by, status="queued"
            )
            try:
                self._queue.put_nowait(_Job(record.id, data, enable_cropping))
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

    def _process(self, job: _Job) -> None:
        record = self._progress_store.get(job.progress_id)
        if record is None:
            return
        self._progress_store.update(job.progress_id, status="running")

        def report(values: dict[str, int]) -> None:
            self._progress_store.update(
                job.progress_id,
                total_rows=values["total"],
                imported_rows=values["imported"],
                skipped_rows=values["skipped"],
                failed_rows=values["failed"],
            )

        try:
            result = self._personnel_store.upload_personnel_zip(
                job.data,
                self._face_processor,
                job.enable_cropping,
                report,
            )
            response = format_zip_result(record.source_filename, result)
            terminal = "completed" if response["success"] else "completed_with_errors"
            self._progress_store.update(job.progress_id, status=terminal, result=response)
        except Exception as exc:
            LOGGER.exception("Personnel ZIP import job %s failed", job.progress_id)
            self._progress_store.update(
                job.progress_id,
                status="failed",
                error_message="پردازش فایل ZIP با خطای داخلی متوقف شد؛ گزارش سرویس را بررسی و دوباره تلاش کنید.",
                result={"success": False, "filename": record.source_filename, "errors": [{"error_type": type(exc).__name__}]},
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(None)
        self._thread.join()
