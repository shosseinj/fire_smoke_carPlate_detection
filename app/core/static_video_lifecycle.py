from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.core.router import TaskRouter
from app.core.source_registry import SourceRecord, SourceRegistry
from app.core.static_video_store import StaticVideoRecord, StaticVideoStore


LOGGER = logging.getLogger("uvicorn.error")


class StaticVideoLifecycle:
    """Coordinates decoder events with persisted static-video state."""

    def __init__(
        self,
        store: StaticVideoStore,
        registry: SourceRegistry,
        router: TaskRouter,
    ) -> None:
        self.store = store
        self.registry = registry
        self.router = router
        self._terminal_sources: set[str] = set()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="static-video-lifecycle")
        self._closed = False
        self._failure_baselines: dict[str, int] = {}

    def recover(self) -> None:
        self.store.recover_interrupted()
        for record in self.store.list():
            if record.processing_status != "queued":
                if self.registry.get(record.source_uri) is not None:
                    self.registry.delete(record.source_uri)
                continue
            if self.registry.get(record.source_uri) is None:
                self.registry.create(self._source_record(record))

    def on_source_started(self, source_uri: str) -> None:
        record = self.store.mark_processing(source_uri)
        if record is not None and record.processing_status == "processing":
            with self._lock:
                self._failure_baselines.setdefault(
                    source_uri, self.router.source_failed_frames(source_uri)
                )

    def on_source_eos(self, source_uri: str, will_loop: bool) -> None:
        if will_loop:
            return
        self._schedule_terminal(source_uri, error=None)

    def on_source_failed(self, source_uri: str, error: str) -> None:
        self._schedule_terminal(source_uri, error=error)

    def _schedule_terminal(self, source_uri: str, error: str | None) -> None:
        with self._lock:
            if self._closed or source_uri in self._terminal_sources:
                return
            self._terminal_sources.add(source_uri)
        self._executor.submit(self._finish, source_uri, error)

    def _finish(self, source_uri: str, error: str | None) -> None:
        try:
            source = self.registry.get(source_uri)
            source_config = source.to_dict() if source is not None else None
            if error is None:
                if not self.router.wait_for_source_idle(source_uri, timeout_seconds=60.0):
                    error = "زمان انتظار برای پایان پردازش فریم‌های ویدیو به پایان رسید"
                else:
                    with self._lock:
                        baseline = self._failure_baselines.get(source_uri, 0)
                    failed_frames = self.router.source_failed_frames(source_uri) - baseline
                    if failed_frames > 0:
                        error = f"پردازش {failed_frames} فریم ارسال‌شده با خطا مواجه شد"
            if error is None:
                self.store.mark_completed(source_uri, source_config=source_config)
            else:
                self.store.mark_failed(
                    source_uri,
                    self._safe_error(source_uri, error),
                    source_config=source_config,
                )
            if self.registry.get(source_uri) is not None:
                self.registry.delete(source_uri)
        except Exception:
            LOGGER.exception("Could not finalize static video %s", source_uri)
        finally:
            with self._lock:
                self._terminal_sources.discard(source_uri)
                self._failure_baselines.pop(source_uri, None)

    def retry(self, source_uri: str, *, loop: bool | None = None) -> StaticVideoRecord:
        record = self.store.reset_for_retry(source_uri, loop=loop)
        if record is None:
            raise KeyError(source_uri)
        existing = self.registry.get(source_uri)
        source = self._source_record(record)
        if existing is None:
            self.registry.create(source)
        else:
            self.registry.update(
                source_uri,
                enabled=True,
                loop=record.loop,
            )
        return record

    def _source_record(self, record: StaticVideoRecord) -> SourceRecord:
        if record.source_config:
            values: dict[str, Any] = dict(record.source_config)
            values.update({
                "id": None,
                "source_uri": record.source_uri,
                "name": record.name,
                "source_type": "static_video",
                "enabled": True,
                "loop": record.loop,
            })
            return SourceRecord.from_dict(values)
        return SourceRecord(
            source_uri=record.source_uri,
            name=record.name,
            source_type="static_video",
            loop=record.loop,
        )

    @staticmethod
    def _safe_error(source_uri: str, error: str) -> str:
        text = str(error).replace(source_uri, "<static-video>")[:2000]
        if any("\u0600" <= character <= "\u06ff" for character in text):
            return text
        return "پردازش ویدیوی ایستا با خطا مواجه شد؛ گزارش سرویس را بررسی کنید"

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
