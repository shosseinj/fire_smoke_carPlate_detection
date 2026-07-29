from __future__ import annotations

import pytest

from app.core.source_registry import STATIC_VIDEO, SourceRecord, SourceRegistry
from app.core.static_video_lifecycle import StaticVideoLifecycle
from app.core.static_video_store import StaticVideoStore
from app.database import Database


pytestmark = pytest.mark.usefixtures("postgres_database")


class IdleRouter:
    def wait_for_source_idle(self, source_id: str, timeout_seconds: float = 60.0) -> bool:
        return True

    def source_failed_frames(self, source_id: str) -> int:
        return 0


class FailedRouter(IdleRouter):
    def __init__(self) -> None:
        self.calls = 0

    def source_failed_frames(self, source_id: str) -> int:
        self.calls += 1
        return 0 if self.calls == 1 else 2


def test_static_video_store_lifecycle(postgres_database: Database) -> None:
    store = StaticVideoStore(postgres_database)
    record = store.create("clip", "/media/clip.mp4", loop=False)
    assert record.processing_status == "queued"
    assert record.is_processed is False
    assert record.processing_attempts == 0

    record = store.mark_processing(record.source_uri)
    assert record is not None
    assert record.processing_status == "processing"
    assert record.processing_attempts == 1

    record = store.mark_completed(record.source_uri)
    assert record is not None
    assert record.processing_status == "completed"
    assert record.is_processed is True

    record = store.reset_for_retry(record.source_uri, loop=True)
    assert record is not None
    assert record.processing_status == "queued"
    assert record.is_processed is False
    assert record.loop is True


def test_terminal_video_is_removed_from_sources_and_can_retry(
    postgres_database: Database,
) -> None:
    store = StaticVideoStore(postgres_database)
    registry = SourceRegistry(postgres_database)
    lifecycle = StaticVideoLifecycle(store, registry, IdleRouter())  # type: ignore[arg-type]
    uri = "/media/retry.mp4"
    store.create("retry", uri, loop=False)
    registry.create(
        SourceRecord(
            source_uri=uri,
            name="retry",
            source_type=STATIC_VIDEO,
            loop=False,
        )
    )
    lifecycle.on_source_started(uri)
    lifecycle.on_source_eos(uri, False)
    lifecycle.close()

    completed = store.get(uri)
    assert completed is not None and completed.is_processed is True
    assert registry.get(uri) is None

    retried = StaticVideoLifecycle(store, registry, IdleRouter())  # type: ignore[arg-type]
    record = retried.retry(uri, loop=True)
    assert record.processing_status == "queued"
    assert record.loop is True
    assert registry.require(uri).loop is True
    retried.close()
    registry.close()


def test_task_failure_marks_video_failed_and_removes_source(
    postgres_database: Database,
) -> None:
    store = StaticVideoStore(postgres_database)
    registry = SourceRegistry(postgres_database)
    lifecycle = StaticVideoLifecycle(store, registry, FailedRouter())  # type: ignore[arg-type]
    uri = "/media/failed.mp4"
    store.create("failed", uri, loop=False)
    registry.create(
        SourceRecord(source_uri=uri, name="failed", source_type=STATIC_VIDEO, loop=False)
    )

    lifecycle.on_source_started(uri)
    lifecycle.on_source_eos(uri, False)
    lifecycle.close()

    failed = store.get(uri)
    assert failed is not None
    assert failed.processing_status == "failed"
    assert failed.is_processed is False
    assert "پردازش 2 فریم ارسال‌شده با خطا مواجه شد" in (failed.processing_error or "")
    assert registry.get(uri) is None
    registry.close()
