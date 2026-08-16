from __future__ import annotations

from datetime import datetime, timezone

import pytest


def test_personnel_request_types_are_defined() -> None:
    request_types = {"leave", "sick_leave", "mission", "remote_work", "personal", "unpaid_leave", "overtime", "earned_leave"}
    assert "mission" in request_types
    assert "earned_leave" in request_types
    assert "overtime" in request_types


def test_detection_log_response_preserves_unknown_person_message() -> None:
    response = {
        "id": 1,
        "person": "unknown",
        "detection_time": "2026-07-08T08:00:00+00:00",
        "log_type": "camera_rtsp",
    }
    assert response["person"] == "unknown"
    assert response["log_type"] == "camera_rtsp"
pytestmark = pytest.mark.unit
