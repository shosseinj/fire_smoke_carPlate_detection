from datetime import date, datetime, time, timezone

from app.core.attendance_summary_service import (
    AttendanceSummaryService,
    SummaryLog,
    SummaryPersonnel,
    SummaryShift,
    TimePeriod,
)


REPORT_DAY = date(2026, 7, 27)


def _shift() -> SummaryShift:
    return SummaryShift(
        id=1,
        shift_name="day",
        start_time=time(7),
        end_time=time(18),
        timezone_name="UTC",
        max_minutes_delay=0,
        max_minutes_early=0,
        max_overtime_hours=0,
        works_monday=True,
        works_tuesday=False,
        works_wednesday=False,
        works_thursday=False,
        works_friday=False,
        works_saturday=False,
        works_sunday=False,
    )


def _daily_summary(monkeypatch, detection_hours: list[int]) -> dict:
    shift = _shift()
    person = SummaryPersonnel(
        id=1,
        fname="Test",
        lname="Person",
        national_code="123",
        department_id=None,
        section_name=None,
        shift_id=shift.id,
        shift=shift,
    )
    logs = [
        SummaryLog(
            id=index,
            person=person.national_code,
            personnel_id=person.id,
            detection_time=datetime(
                REPORT_DAY.year,
                REPORT_DAY.month,
                REPORT_DAY.day,
                hour,
                tzinfo=timezone.utc,
            ),
        )
        for index, hour in enumerate(detection_hours, start=1)
    ]
    service = AttendanceSummaryService(database=None)  # type: ignore[arg-type]
    monkeypatch.setattr(
        service,
        "_resolve_daily_date_range",
        lambda *_: (
            datetime(2026, 7, 27, tzinfo=timezone.utc),
            datetime(2026, 7, 28, tzinfo=timezone.utc),
            REPORT_DAY,
            REPORT_DAY,
        ),
    )
    monkeypatch.setattr(service, "_personnel", lambda: [person])
    monkeypatch.setattr(service, "_logs", lambda *_: {person.national_code: logs})
    monkeypatch.setattr(service, "_holiday_dates", lambda *_: set())
    monkeypatch.setattr(service, "_accepted_requests_by_person_day", lambda *_: {})

    rows = service.daily_summary(
        period=TimePeriod.CUSTOM,
        from_date_jalali=None,
        to_date_jalali=None,
        include_non_workdays=False,
        include_absent=True,
    )
    assert len(rows) == 1
    return rows[0]


def test_daily_summary_single_detection_hides_first_and_keeps_last(monkeypatch) -> None:
    row = _daily_summary(monkeypatch, [9])

    assert row["first_detection"] is None
    assert row["last_detection"] == "09:00"


def test_daily_summary_two_detections_keeps_first_and_last(monkeypatch) -> None:
    row = _daily_summary(monkeypatch, [9, 17])

    assert row["first_detection"] == "09:00"
    assert row["last_detection"] == "17:00"
