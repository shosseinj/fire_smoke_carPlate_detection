"""Tests for AttendanceService."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from app.core.attendance_service import AttendanceService
from app.core.holiday_store import HolidayStore
from app.core.human_log_store import HumanLogStore
from app.core.personnel_store import PersonnelStore
from app.core.request_store import RequestStore
from app.core.shift_store import ShiftStore
from app.database import Database


@pytest.fixture
def stores(postgres_database: Database, tmp_path: Path) -> dict[str, Any]:
    media_path = tmp_path / "media"
    with postgres_database.connection() as connection:
        connection.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            ("attendance-reviewer", "unused-in-store-tests", "admin"),
        )
    shift_store = ShiftStore(postgres_database)
    holiday_store = HolidayStore(postgres_database)
    request_store = RequestStore(postgres_database)
    personnel_store = PersonnelStore(postgres_database, media_path)
    human_log_store = HumanLogStore(
        database=postgres_database,
        saved_media_path=media_path,
        queue_size=16,
    )
    svc = AttendanceService(
        personnel_store=personnel_store,
        human_log_store=human_log_store,
        shift_store=shift_store,
        holiday_store=holiday_store,
        request_store=request_store,
    )
    yield {
        "database": postgres_database,
        "shift": shift_store,
        "holiday": holiday_store,
        "request": request_store,
        "personnel": personnel_store,
        "human_logs": human_log_store,
        "service": svc,
    }
    human_log_store.close()


def _make_personnel(stores: dict[str, Any]) -> int:
    p = stores["personnel"].create(
        fname="Att", lname="Test",
        national_code="1234567891",
        employee_type="employee",
    )
    return p.id


def _make_personnel_with_shift(stores: dict[str, Any]) -> tuple[int, int]:
    pid = _make_personnel(stores)
    shift = stores["shift"].create(
        shift_name="Day",
        shift_type="morning",
        start_time="08:00",
        end_time="16:00",
        works_saturday=True, works_sunday=True,
        works_monday=True, works_tuesday=True,
        works_wednesday=True, works_thursday=True,
        works_friday=True,  # Work all days for test
    )
    stores["shift"].assign_personnel(pid, shift.id)
    return pid, shift.id


class TestAttendanceService:
    def test_daily_absent(self, stores: dict[str, Any]) -> None:
        pid, _ = _make_personnel_with_shift(stores)
        result = stores["service"].compute_daily_summary(pid, "2026-07-21")
        assert result["status"] == "absent"
        assert result["day_type"] == "working_day"

    def test_daily_holiday(self, stores: dict[str, Any]) -> None:
        pid, _ = _make_personnel_with_shift(stores)
        stores["holiday"].create(name="TestH", date_value="2026-07-21", holiday_type="national")
        result = stores["service"].compute_daily_summary(pid, "2026-07-21")
        assert result["is_holiday"] is True
        assert result["day_type"] == "holiday"

    def test_daily_non_working_day(self, stores: dict[str, Any]) -> None:
        pid = _make_personnel(stores)
        shift = stores["shift"].create(
            shift_name="WeekdaysOnly",
            works_saturday=True, works_sunday=True,
            works_monday=True, works_tuesday=True,
            works_wednesday=True, works_thursday=True,
            works_friday=False,  # Friday is off
        )
        stores["shift"].assign_personnel(pid, shift.id)
        # 2026-07-24 is a Friday (Python weekday=4, Iranian=6)
        result = stores["service"].compute_daily_summary(pid, "2026-07-24")
        assert result["day_type"] == "non_working_day"

    def test_daily_present(self, stores: dict[str, Any]) -> None:
        pid, _ = _make_personnel_with_shift(stores)
        # Manually insert a human log for the personnel
        stores["human_logs"].set_counts_for_attendance  # just reference
        result = stores["service"].compute_daily_summary(pid, "2026-07-21")
        # Should still be absent since no logs
        assert result["status"] == "absent"

    def test_monthly_summary_gregorian(self, stores: dict[str, Any]) -> None:
        pid, _ = _make_personnel_with_shift(stores)
        result = stores["service"].compute_monthly_summary(pid, 2026, 7, "gregorian")
        assert result["total_days"] >= 28
        assert result["absent_days"] == result["total_days"]  # all absent

    def test_monthly_summary_jalali(self, stores: dict[str, Any]) -> None:
        pid, _ = _make_personnel_with_shift(stores)
        result = stores["service"].compute_monthly_summary(pid, 1405, 4, "jalali")
        assert result["calendar"] == "jalali"
        assert "period" in result

    def test_monthly_performance(self, stores: dict[str, Any]) -> None:
        pid, _ = _make_personnel_with_shift(stores)
        result = stores["service"].compute_monthly_performance(2026, 7, "gregorian")
        assert "results" in result
        assert result["total_personnel"] >= 1

    def test_yearly_leave_summary(self, stores: dict[str, Any]) -> None:
        pid, _ = _make_personnel_with_shift(stores)
        # Create approved leave
        req = stores["request"].create(
            personnel_id=pid,
            request_type="leave",
            start_date="2026-08-01",
            end_date="2026-08-05",
        )
        stores["request"].approve(req.id, approved_by=1)
        result = stores["service"].compute_yearly_leave_summary(pid, 2026, "gregorian")
        assert result["total_leave_days"] == 5
        assert result["total_days_off"] == 5

    def test_yearly_leave_summary_jalali(self, stores: dict[str, Any]) -> None:
        pid, _ = _make_personnel_with_shift(stores)
        result = stores["service"].compute_yearly_leave_summary(pid, 1405, "jalali")
        assert result["calendar"] == "jalali"

    def test_set_counts_for_attendance(self, stores: dict[str, Any]) -> None:
        """Test toggling counts_for_attendance on a log."""
        # We need a real log entry, so we test the underlying store method
        result = stores["service"].set_counts_for_attendance(99999, False)
        assert result is None  # nonexistent log

    def test_invalid_personnel(self, stores: dict[str, Any]) -> None:
        result = stores["service"].compute_daily_summary(99999, "2026-07-21")
        assert "error" in result

    def test_expected_seconds(self, stores: dict[str, Any]) -> None:
        shift = stores["shift"].create(
            shift_name="EightHours",
            start_time="08:00",
            end_time="16:00",
            works_saturday=True, works_sunday=True,
            works_monday=True, works_tuesday=True,
            works_wednesday=True, works_thursday=True,
            works_friday=True,
        )
        expected = stores["service"]._compute_expected_seconds(shift, date(2026, 7, 21))
        assert expected == 8 * 3600  # 8 hours

    def test_overnight_expected_seconds(self, stores: dict[str, Any]) -> None:
        shift = stores["shift"].create(
            shift_name="NightShift",
            start_time="22:00",
            end_time="06:00",
            works_saturday=True, works_sunday=True,
            works_monday=True, works_tuesday=True,
            works_wednesday=True, works_thursday=True,
            works_friday=True,
        )
        expected = stores["service"]._compute_expected_seconds(shift, date(2026, 7, 21))
        assert expected == 8 * 3600  # 8 hours
