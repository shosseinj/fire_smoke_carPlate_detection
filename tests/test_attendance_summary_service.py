from dataclasses import replace
from datetime import date, datetime, time, timezone

from app.core.attendance_summary_service import (
    AttendanceSummaryService,
    SummaryLog,
    SummaryPersonnel,
    SummaryShift,
    TimePeriod,
    _compute_shift_day_stats,
    _daily_summary_stats,
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


def test_daily_summary_single_detection_keeps_first_and_hides_last(monkeypatch) -> None:
    row = _daily_summary(monkeypatch, [9])

    assert row["first_detection"] == "09:00"
    assert row["last_detection"] is None


def test_daily_summary_two_detections_keeps_first_and_last(monkeypatch) -> None:
    row = _daily_summary(monkeypatch, [9, 17])

    assert row["first_detection"] == "09:00"
    assert row["last_detection"] == "17:00"


def test_daily_summary_first_last_span_is_not_clipped_to_shift(monkeypatch) -> None:
    row = _daily_summary(monkeypatch, [6, 20])

    assert row["first_last_span_time"] == "14:00"
    assert row["raw_worked_time"] == "11:00"


def test_daily_summary_first_last_span_is_null_for_odd_or_zero_detections(
    monkeypatch,
) -> None:
    odd_row = _daily_summary(monkeypatch, [6, 12, 20])
    empty_row = _daily_summary(monkeypatch, [])

    assert odd_row["first_last_span_time"] is None
    assert empty_row["first_last_span_time"] is None


def _log_at(hour: int, minute: int, *, log_id: int = 1) -> SummaryLog:
    return SummaryLog(
        id=log_id,
        person="123",
        personnel_id=1,
        detection_time=datetime(
            REPORT_DAY.year,
            REPORT_DAY.month,
            REPORT_DAY.day,
            hour,
            minute,
            tzinfo=timezone.utc,
        ),
    )


def _shift_with_delay_grace() -> SummaryShift:
    return replace(_shift(), max_minutes_delay=15)


def test_daily_delay_grace_preserves_arrival_and_forgives_up_to_limit() -> None:
    shift = _shift_with_delay_grace()

    for arrival_minute in (14, 15):
        arrival = _log_at(7, arrival_minute)
        stats = _daily_summary_stats(
            REPORT_DAY,
            [arrival],
            shift,
            requests=[],
            is_workday=True,
        )

        assert stats["first_detection"] == arrival.detection_time
        assert stats["delay_minutes"] == 0


def test_daily_delay_above_grace_reports_full_delay() -> None:
    shift = _shift_with_delay_grace()
    arrival = _log_at(7, 18)

    stats = _daily_summary_stats(
        REPORT_DAY,
        [arrival],
        shift,
        requests=[],
        is_workday=True,
    )

    assert stats["first_detection"] == arrival.detection_time
    assert stats["delay_minutes"] == 18


def test_monthly_delay_grace_uses_same_all_or_nothing_rule() -> None:
    shift = _shift_with_delay_grace()

    for arrival_minute, expected_delay in ((14, 0), (15, 0), (18, 18)):
        arrival = _log_at(7, arrival_minute)
        departure = _log_at(18, 0, log_id=2)
        stats = _compute_shift_day_stats(REPORT_DAY, [arrival, departure], shift)

        assert stats["first_detection"] == arrival.detection_time
        assert stats["delay_minutes"] == expected_delay


def test_non_working_day_counts_only_paired_presence_as_holiday_overtime() -> None:
    stats = _daily_summary_stats(
        REPORT_DAY,
        [
            _log_at(6, 0, log_id=1),
            _log_at(9, 0, log_id=2),
            _log_at(14, 0, log_id=3),
            _log_at(18, 0, log_id=4),
        ],
        _shift(),
        requests=[],
        is_workday=False,
    )

    assert stats["holiday_overtime_minutes"] == 7 * 60
    assert stats["overtime_minutes"] == 0
    assert stats["illegal_presence_minutes"] == 0


def test_non_working_day_holiday_overtime_excludes_between_pair_gaps() -> None:
    stats = _daily_summary_stats(
        REPORT_DAY,
        [
            _log_at(7, 0, log_id=1),
            _log_at(10, 0, log_id=2),
            _log_at(16, 0, log_id=3),
            _log_at(18, 0, log_id=4),
        ],
        _shift(),
        requests=[],
        is_workday=False,
    )

    assert stats["first_last_span_time"] == 11 * 60
    assert stats["holiday_overtime_minutes"] == 5 * 60


def test_non_working_day_odd_detection_ignores_only_unmatched_final_entry() -> None:
    stats = _daily_summary_stats(
        REPORT_DAY,
        [
            _log_at(7, 0, log_id=1),
            _log_at(10, 0, log_id=2),
            _log_at(16, 0, log_id=3),
        ],
        _shift(),
        requests=[],
        is_workday=False,
    )

    assert stats["holiday_overtime_minutes"] == 3 * 60
    assert stats["first_last_span_time"] is None
    assert stats["raw_worked_time"] is None
    assert stats["net_worked_time"] is None
    assert stats["message"] is not None


def _monthly_summary_date_range(monkeypatch, *, move_days: int) -> dict:
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
    service = AttendanceSummaryService(database=None)  # type: ignore[arg-type]
    monkeypatch.setattr(service, "_personnel", lambda *_: [person])
    monkeypatch.setattr(service, "_logs", lambda *_: {person.national_code: []})
    monkeypatch.setattr(service, "_holiday_dates", lambda *_: set())
    monkeypatch.setattr(service, "_accepted_requests_by_person_day", lambda *_: {})

    rows = service.monthly_summary(
        jalali_year=1405,
        jalali_month=6,
        personnel_id=None,
        section_id=None,
        shift_id=None,
        include_daily_rows=False,
        move_days=move_days,
    )
    assert len(rows) == 1
    return rows[0]


def test_monthly_summary_returns_jalali_month_boundaries(monkeypatch) -> None:
    row = _monthly_summary_date_range(monkeypatch, move_days=0)

    assert row["month_start"] == "1405-06-01"
    assert row["month_end"] == "1405-06-31"


def test_monthly_summary_returns_jalali_shifted_boundaries(monkeypatch) -> None:
    row = _monthly_summary_date_range(monkeypatch, move_days=15)

    assert row["month_start"] == "1405-05-16"
    assert row["month_end"] == "1405-06-15"
