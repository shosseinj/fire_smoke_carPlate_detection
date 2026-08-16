from __future__ import annotations

from dataclasses import replace

from app.config import settings
from app.core.broadcast import BroadcastControlEvent
from app.runtime import build_runtime


def test_saved_detection_is_published_to_dashboard_subscribers(
    postgres_database,
) -> None:
    runtime = build_runtime(
        replace(
            settings,
            database_url=postgres_database.url,
            processor_mode="mock",
            video_ingestion_enabled=False,
            media_preview_enabled=False,
            broadcast_enabled=True,
        )
    )
    subscriber_id, target = runtime.broadcast.subscribe()
    try:
        created = runtime.detection_log_store.create(
            source_system="test",
            person="Unknown",
            confidence=0.9,
            counts_for_attendance=True,
        )

        event = target.get(timeout=2.0)
        assert isinstance(event, BroadcastControlEvent)
        assert event.payload["type"] == "recent_detections"
        assert event.payload["reason"] == "log_created"
        assert event.payload["updated_log_id"] == created.id
        assert event.payload["detections"][0]["id"] == created.id
        assert event.payload["limit"] > 0
    finally:
        runtime.broadcast.unsubscribe(subscriber_id)
        runtime.close()

import pytest

pytestmark = [pytest.mark.postgresql, pytest.mark.streaming]
