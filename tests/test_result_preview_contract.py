from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from app.core.types import FramePacket, TaskName, TaskResult


def test_result_includes_frame_geometry_and_source_timestamp() -> None:
    packet = FramePacket(
        source_id="camera-preview",
        frame=np.zeros((360, 640, 3), dtype=np.uint8),
        round_sequence=2,
        frame_index=41,
        captured_monotonic=time.monotonic(),
        captured_at_utc="2026-07-22T10:00:00+00:00",
        source_time_seconds=12.5,
    )

    payload = TaskResult.success(
        task=TaskName.FIRE_SMOKE,
        packet=packet,
        processing_ms=3.25,
        data={"tracks": [{"bbox": [10, 20, 30, 40]}]},
    ).to_dict()

    assert payload["frame_width"] == 640
    assert payload["frame_height"] == 360
    assert payload["source_time_seconds"] == 12.5
    assert payload["captured_at_utc"] == "2026-07-22T10:00:00+00:00"
    assert payload["processed_at_utc"]


def test_dashboard_uses_public_preview_and_bounded_multi_task_overlays() -> None:
    dashboard = (
        Path(__file__).parents[1] / "app" / "web" / "dashboard.html"
    ).read_text(encoding="utf-8")

    assert 'fetch("/api/v1/sources/preview-config"' in dashboard
    assert "source.preview_path" in dashboard
    assert "requestVideoFrameCallback" in dashboard
    assert "OVERLAY_STALE_MS = 4000" in dashboard
    assert "taskResults.set(result.task, result)" in dashboard
    assert "schedulePreviewRetry(sourceId)" in dashboard
    assert "previewRetryTimers" in dashboard
    assert 'method: "DELETE"' in dashboard
    assert "/api/v1/broadcast/ws" in dashboard
    assert "source.source_id" not in dashboard
    assert "source.source_uri" in dashboard
