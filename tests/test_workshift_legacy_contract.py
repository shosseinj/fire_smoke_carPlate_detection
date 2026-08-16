from __future__ import annotations

from app.api.shifts import ShiftCreate, ShiftUpdate, router


def test_shift_request_defaults_match_legacy_schema() -> None:
    create = ShiftCreate(
        shift_name="Morning",
        shift_type="morning",
        start_time="08:00",
        end_time="16:00",
    )
    assert create.max_minutes_delay is None
    assert create.max_minutes_early is None
    assert create.max_overtime_hours == 8.0
    assert create.monday is True
    assert create.tuesday is True
    assert create.wednesday is True
    assert create.thursday is False
    assert create.friday is False
    assert create.saturday is True
    assert create.sunday is True

    update = ShiftUpdate()
    assert update.monday is True
    assert update.thursday is False


def test_shift_legacy_routes_are_registered_in_archive_order() -> None:
    registered = [
        (route.path, tuple(sorted(route.methods or [])))
        for route in router.routes[:8]
    ]
    assert registered == [
        ("/api/v1/shifts/bulk-assignments", ("POST",)),
        ("/api/v1/shifts/", ("GET",)),
        ("/api/v1/shifts/statistics", ("GET",)),
        ("/api/v1/shifts/{shift_id}", ("GET",)),
        ("/api/v1/shifts/", ("POST",)),
        ("/api/v1/shifts/{shift_id}", ("PUT",)),
        ("/api/v1/shifts/{shift_id}", ("DELETE",)),
        ("/api/v1/shifts/{shift_id}/personnel", ("GET",)),
    ]

import pytest

pytestmark = pytest.mark.unit
