from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.api.fire_logs import FireLogCreate, FireLogResponse, FireLogUpdate, _date_range_values
from app.core.jalali_utils import utc_iso_to_jalali_datetime


def test_fire_log_create_requires_independent_confidence() -> None:
    with pytest.raises(ValidationError):
        FireLogCreate(
            detection_time=datetime.now(timezone.utc),
            camera_id="camera-1",
            hazard_type="fire",
            severity="high",
        )

    payload = FireLogCreate(
        detection_time=datetime.now(timezone.utc),
        camera_id=" camera-1 ",
        hazard_type="fire_smoke",
        severity="high",
        fire_confidence=0.9,
    )
    assert payload.camera_id == "camera-1"


def test_fire_log_request_enums_and_camera_validation() -> None:
    with pytest.raises(ValidationError):
        FireLogCreate(
            detection_time=datetime.now(timezone.utc),
            camera_id="camera-1",
            hazard_type="steam",
            severity="high",
            fire_confidence=0.5,
        )
    with pytest.raises(ValidationError):
        FireLogUpdate(camera_id="   ")


def test_fire_log_response_contract_and_date_range() -> None:
    response = FireLogResponse(
        id=1,
        detection_time=utc_iso_to_jalali_datetime(
            datetime.now(timezone.utc).isoformat()
        ),
        camera_id="camera-1",
        hazard_type="fire",
        severity="high",
        fire_count=1,
        smoke_count=0,
        fire_confidence=0.9,
        smoke_confidence=0.0,
        window_seconds=3.0,
        snapshot_url="",
        video_url="",
        details={"events": []},
    )
    assert response.details == {"events": []}
    assert isinstance(response.detection_time, str)

    start, end = _date_range_values(datetime(2026, 1, 1), datetime(2026, 1, 2))
    assert start == "2026-01-01T00:00:00+00:00"
    assert end == "2026-01-02T00:00:00+00:00"
    with pytest.raises(ValueError):
        _date_range_values(datetime(2026, 1, 2), datetime(2026, 1, 1))

pytestmark = pytest.mark.unit
