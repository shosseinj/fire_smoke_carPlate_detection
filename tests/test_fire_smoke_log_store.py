from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from app.core.fire_smoke_log_store import FireSmokeLogStore
from app.core.types import FramePacket, TaskName, TaskResult
from app.fire_core.policy import FireSmokePolicyConfig


def test_fire_event_snapshot_and_policy_are_persisted_off_worker_path(
    tmp_path: Path,
) -> None:
    database = tmp_path / "logs.sqlite3"
    store = FireSmokeLogStore(database, tmp_path / "media")
    packet = FramePacket(
        source_id="camera-fire",
        frame=np.zeros((64, 64, 3), dtype=np.uint8),
        round_sequence=1,
        frame_index=5,
        captured_monotonic=10.0,
        captured_at_utc="2026-01-01T00:00:00+00:00",
    )
    result = TaskResult.success(
        task=TaskName.FIRE_SMOKE,
        packet=packet,
        processing_ms=1.0,
        data={
            "severity": "medium",
            "previous_severity": "none",
            "severity_changed": True,
            "severity_window_seconds": 3.0,
            "incident_id": "incident-stable-1",
            "fire": {"positive_count": 10, "max_confidence": 0.91},
            "smoke": {"positive_count": 0, "max_confidence": 0.0},
            "tracks": [
                {
                    "label": "fire",
                    "confidence": 0.91,
                    "bbox": [5, 5, 30, 30],
                }
            ],
            "events": [
                {
                    "event_type": "incident_started",
                    "incident_id": "incident-stable-1",
                }
            ],
        },
    )
    store.observe_result(packet, result)
    upgraded = TaskResult.success(
        task=TaskName.FIRE_SMOKE,
        packet=packet,
        processing_ms=1.0,
        data={
            **result.data,
            "severity": "high",
            "previous_severity": "medium",
            "fire": {"positive_count": 20, "max_confidence": 0.95},
            "events": [
                {
                    "event_type": "alert_started",
                    "incident_id": "incident-stable-1",
                }
            ],
        },
    )
    store.observe_result(packet, upgraded)
    downgraded = TaskResult.success(
        task=TaskName.FIRE_SMOKE,
        packet=packet,
        processing_ms=1.0,
        data={
            **result.data,
            "severity": "medium",
            "previous_severity": "high",
            "fire": {"positive_count": 10, "max_confidence": 0.91},
        },
    )
    store.observe_result(packet, downgraded)
    updated = store.update_policy(
        FireSmokePolicyConfig(
            window_seconds=4.0,
            low_count=6,
            medium_count=12,
            high_count=24,
        )
    )
    store.close()

    assert updated["window_seconds"] == 4.0
    rows = store.list(camera="camera-fire")
    assert len(rows) == 1
    assert rows[0]["fire_count"] == 20
    assert rows[0]["severity"] == "high"
    assert rows[0]["snapshot_url"].startswith("/media/fire_smoke_snapshots/")
    snapshot = tmp_path / "media" / rows[0]["snapshot_url"].removeprefix("/media/")
    assert snapshot.is_file()
    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {"fire_smoke_logs", "fire_smoke_settings"}.issubset(tables)
