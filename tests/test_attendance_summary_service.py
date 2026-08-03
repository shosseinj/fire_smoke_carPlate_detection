from dataclasses import replace
from types import SimpleNamespace
from datetime import date, datetime, time, timedelta, timezone

import app.core.attendance_summary_service as attendance_module
from app.core.attendance_summary_service import (
    AttendanceSummaryService,
    SummaryLog,
    SummaryPersonnel,
    SummaryShift,
    TimePeriod,
    _calculate_monthly_earned_leave_days,
    _compute_shift_day_stats,
    _daily_summary_stats,
    _yearly_leave_month_attendance,
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
    store = _FakeShiftStore(
        [
            SimpleNamespace(
                personnel_id=person.id,
                shift_id=shift.id,
                start_date=REPORT_DAY,
                end_date=REPORT_DAY,
            )
        ],
        {shift.id: _shift_record(shift.id, shift.shift_name)},
    )
    service = AttendanceSummaryService(database=None, shift_store=store)  # type: ignore[arg-type]
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


def test_daily_summary_emits_no_shift_row_when_no_assignment_covers_day(
    monkeypatch,
) -> None:
    person = SummaryPersonnel(
        id=1,
        fname="Test",
        lname="Person",
        national_code="123",
        department_id=None,
        section_name=None,
        shift_id=None,
        shift=None,
    )
    service = AttendanceSummaryService(
        database=None,  # type: ignore[arg-type]
        shift_store=_FakeShiftStore([], {}),
    )
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
    monkeypatch.setattr(service, "_logs", lambda *_: {})
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
    row = rows[0]
    assert row["status"] == "no_shift"
    assert row["message"] == "برای این کارمند شیفت کاری تعریف نشده است"
    assert row["shift_id"] is None
    assert row["first_detection"] is None
    assert row["delay_minutes"] is None
    assert row["overtime_minutes"] is None
    assert row["raw_worked_time"] is None


def test_daily_summary_no_shift_row_shows_detections_but_null_shift_calculations(
    monkeypatch,
) -> None:
    person = SummaryPersonnel(
        id=1,
        fname="Test",
        lname="Person",
        national_code="123",
        department_id=None,
        section_name=None,
        shift_id=None,
        shift=None,
    )
    logs = [
        SummaryLog(
            id=1,
            person=person.national_code,
            personnel_id=person.id,
            detection_time=datetime(2026, 7, 27, 8, 5, tzinfo=timezone.utc),
        ),
        SummaryLog(
            id=2,
            person=person.national_code,
            personnel_id=person.id,
            detection_time=datetime(2026, 7, 27, 12, 30, tzinfo=timezone.utc),
        ),
        SummaryLog(
            id=3,
            person=person.national_code,
            personnel_id=person.id,
            detection_time=datetime(2026, 7, 27, 17, 45, tzinfo=timezone.utc),
        ),
    ]
    service = AttendanceSummaryService(
        database=None,  # type: ignore[arg-type]
        shift_store=_FakeShiftStore([], {}),
    )
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
    row = rows[0]
    assert row["status"] == "no_shift"
    assert row["message"] == "برای این کارمند شیفت کاری تعریف نشده است"
    assert row["first_detection"] == "11:35"
    assert row["last_detection"] == "21:15"
    assert row["middle_detections"] == ["16:00"]
    assert row["raw_worked_time"] is None
    assert row["net_worked_time"] is None
    assert row["delay_minutes"] is None
    assert row["early_leave_minutes"] is None
    assert row["overtime_minutes"] is None
    assert row["holiday_overtime_minutes"] is None
    assert row["illegal_presence_minutes"] is None
    assert row["total_absence_time"] is None


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


def _monthly_summary_date_range(
    monkeypatch, *, move_days: int, holiday_on_shift_off_day: bool = False,
    include_daily_rows: bool = False,
) -> dict:
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
    store = _FakeShiftStore(
        [
            SimpleNamespace(
                personnel_id=person.id,
                shift_id=shift.id,
                start_date=date(2026, 3, 20),
                end_date=date(2027, 3, 21),
            )
        ],
        {shift.id: _shift_record(shift.id, shift.shift_name)},
    )
    service = AttendanceSummaryService(database=None, shift_store=store)  # type: ignore[arg-type]
    monkeypatch.setattr(service, "_personnel", lambda *_: [person])
    monkeypatch.setattr(service, "_logs", lambda *_: {person.national_code: []})
    holiday_date: dict[str, date] = {}

    def holiday_dates(start_day: date, end_day: date) -> set[date]:
        if not holiday_on_shift_off_day:
            return set()
        day = next(
            candidate
            for candidate in (
                start_day + timedelta(days=offset)
                for offset in range((end_day - start_day).days + 1)
            )
            if candidate.weekday() != 0
        )
        holiday_date["value"] = day
        return {day}

    monkeypatch.setattr(service, "_holiday_dates", holiday_dates)
    monkeypatch.setattr(service, "_accepted_requests_by_person_day", lambda *_: {})

    rows = service.monthly_summary(
        jalali_year=1405,
        jalali_month=6,
        personnel_id=None,
        section_id=None,
        shift_id=None,
        include_daily_rows=include_daily_rows,
        move_days=move_days,
    )
    assert len(rows) == 1
    if holiday_date:
        rows[0]["_test_holiday_date"] = holiday_date["value"]
    return rows[0]


def test_monthly_summary_returns_jalali_month_boundaries(monkeypatch) -> None:
    row = _monthly_summary_date_range(monkeypatch, move_days=0)

    assert row["month_start"] == "1405-06-01"
    assert row["month_end"] == "1405-06-31"


def test_monthly_summary_returns_jalali_shifted_boundaries(monkeypatch) -> None:
    row = _monthly_summary_date_range(monkeypatch, move_days=15)

    assert row["month_start"] == "1405-05-16"
    assert row["month_end"] == "1405-06-15"


def test_monthly_summary_reports_overlapping_shift_off_and_custom_holiday_independently(
    monkeypatch,
) -> None:
    row = _monthly_summary_date_range(
        monkeypatch,
        move_days=0,
        holiday_on_shift_off_day=True,
        include_daily_rows=True,
    )
    holiday_day = row.pop("_test_holiday_date")
    holiday_row = next(
        item for item in row["days"] if item["date"] == holiday_day.isoformat()
    )

    assert holiday_row["is_shift_day"] is False
    assert holiday_row["is_custom_holiday"] is True
    assert holiday_row["status"] == "custom_holiday"
    assert row["custom_holidays"] == 1
    assert row["total_holidays"] == row["off_days"]


def test_yearly_month_attendance_deduplicates_overlapping_holiday_total() -> None:
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
    month_start = date(2026, 8, 23)
    month_end = date(2026, 9, 22)
    holiday_day = month_start
    shifts_by_day = {
        month_start + timedelta(days=offset): shift
        for offset in range((month_end - month_start).days + 1)
    }

    stats = _yearly_leave_month_attendance(
        person,
        person_logs=[],
        month_start=month_start,
        month_end=month_end,
        holiday_days={holiday_day},
        requests_by_day={},
        shifts_by_day=shifts_by_day,
    )

    assert stats["off_days"] == 26
    assert stats["custom_holidays"] == 1
    assert stats["total_holidays"] == 26
    assert stats["month_working_days"] == 5


def test_monthly_earned_leave_is_full_when_all_expected_days_are_eligible() -> None:
    assert _calculate_monthly_earned_leave_days(22, 22) == 2.5


def test_monthly_earned_leave_reduces_for_absence_and_unpaid_leave() -> None:
    # Twenty expected days, with one absence and one unpaid-leave day.
    assert _calculate_monthly_earned_leave_days(18, 20) == 2.25


def test_monthly_earned_leave_is_zero_without_expected_workdays() -> None:
    assert _calculate_monthly_earned_leave_days(0, 0) == 0.0


def test_yearly_leave_uses_previous_month_21_to_current_month_20(
    monkeypatch,
) -> None:
    person = SummaryPersonnel(
        id=1,
        fname="Test",
        lname="Person",
        national_code="123",
        department_id=None,
        section_name=None,
        shift_id=None,
        shift=None,
    )
    service = AttendanceSummaryService(
        database=None,  # type: ignore[arg-type]
        shift_store=_FakeShiftStore([], {}),
    )
    actual_range = attendance_module._jalali_moving_month_range
    move_values: list[int] = []

    def tracked_range(year: int, month: int, move_days: int):
        move_values.append(move_days)
        return actual_range(year, month, move_days)

    monkeypatch.setattr(attendance_module, "_jalali_moving_month_range", tracked_range)
    monkeypatch.setattr(service, "_personnel", lambda *_: [person])
    monkeypatch.setattr(service, "_logs", lambda *_: {})
    monkeypatch.setattr(service, "_holiday_dates", lambda *_: set())
    monkeypatch.setattr(service, "_accepted_requests_by_person_day", lambda *_: {})
    monkeypatch.setattr(service, "_accepted_requests", lambda *_args, **_kwargs: [])

    row = service.yearly_leave_summary(1405)[0]

    assert move_values == [10] * 12
    assert row["months"][0]["period_start"] == "1404-12-21"
    assert row["months"][0]["period_end"] == "1405-01-20"
    assert row["months"][11]["period_start"] == "1405-11-21"
    assert row["months"][11]["period_end"] == "1405-12-20"


def test_monthly_summary_uses_dated_shift_assignments_and_preserves_gaps(
    monkeypatch,
) -> None:
    first_day = date(2026, 7, 1)
    last_day = date(2026, 7, 5)
    person = SummaryPersonnel(
        id=1,
        fname="Test",
        lname="Person",
        national_code="123",
        department_id=None,
        section_name=None,
        shift_id=2,
        shift=None,
    )
    assignments = [
        SimpleNamespace(
            personnel_id=person.id,
            shift_id=1,
            start_date=first_day,
            end_date=date(2026, 7, 2),
        ),
        SimpleNamespace(
            personnel_id=person.id,
            shift_id=2,
            start_date=date(2026, 7, 4),
            end_date=last_day,
        ),
    ]
    records = {1: _shift_record(1, "first"), 2: _shift_record(2, "second")}
    for record in records.values():
        for weekday in (
            "monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday",
        ):
            setattr(record, f"works_{weekday}", True)
    service = AttendanceSummaryService(
        database=None,  # type: ignore[arg-type]
        shift_store=_FakeShiftStore(assignments, records),
    )
    logs = [
        SummaryLog(1, person.national_code, person.id, datetime(2026, 7, 1, 7, tzinfo=timezone.utc)),
        SummaryLog(2, person.national_code, person.id, datetime(2026, 7, 1, 18, tzinfo=timezone.utc)),
    ]
    monkeypatch.setattr(
        attendance_module,
        "_jalali_moving_month_range",
        lambda *_: (first_day, last_day),
    )
    monkeypatch.setattr(service, "_personnel", lambda *_: [person])
    monkeypatch.setattr(service, "_logs", lambda *_: {person.national_code: logs})
    monkeypatch.setattr(service, "_holiday_dates", lambda *_: set())
    monkeypatch.setattr(service, "_accepted_requests_by_person_day", lambda *_: {})

    rows = service.monthly_summary(
        jalali_year=1405,
        jalali_month=4,
        personnel_id=None,
        section_id=None,
        shift_id=2,
        include_daily_rows=True,
        move_days=0,
    )

    assert len(rows) == 1
    row = rows[0]
    assert [item["date"] for item in row["days"]] == [
        (first_day + timedelta(days=offset)).isoformat() for offset in range(5)
    ]
    assert row["days"][0]["status"] == "present"
    assert row["days"][0]["detection_count"] == 2
    assert row["days"][0]["work_time"] == "11:00"
    assert row["days"][2]["status"] == "no_shift"
    assert row["days"][2]["shift_id"] is None
    assert row["no_shift_days"] == 1
    assert row["off_days"] == 0
    assert row["custom_holidays"] == 0
    assert row["total_holidays"] == 0
    assert row["month_working_days"] == 4
    assert row["present_days"] == 1
    assert row["absent_days"] == 3
    assert row["shift_id"] is None
    assert row["shift_assignments"] == [
        {
            "shift_id": 1,
            "shift_name": "first",
            "start_date": "2026-07-01",
            "end_date": "2026-07-02",
        },
        {
            "shift_id": 2,
            "shift_name": "second",
            "start_date": "2026-07-04",
            "end_date": "2026-07-05",
        },
    ]


class _FakeShiftStore:
    def __init__(self, assignments, shifts):
        self.assignments = assignments
        self.shifts = shifts

    def list_assignments(self, personnel_id, start_date=None, end_date=None):
        return [
            item for item in self.assignments
            if item.personnel_id == personnel_id
            and (start_date is None or item.end_date >= start_date)
            and (end_date is None or item.start_date <= end_date)
        ]

    def get(self, shift_id):
        return self.shifts.get(shift_id)


def _shift_record(shift_id: int, name: str, start: str = "07:00", end: str = "18:00"):
    shift = _shift()
    return SimpleNamespace(
        id=shift_id, shift_name=name, start_time=start, end_time=end,
        timezone_name="UTC", max_minutes_delay=0, max_minutes_early=0,
        max_overtime_hours=0, works_monday=shift.works_monday,
        works_tuesday=shift.works_tuesday, works_wednesday=shift.works_wednesday,
        works_thursday=shift.works_thursday, works_friday=shift.works_friday,
        works_saturday=shift.works_saturday, works_sunday=shift.works_sunday,
    )


def test_effective_shifts_switch_at_inclusive_assignment_boundary() -> None:
    assignments = [
        SimpleNamespace(personnel_id=1, shift_id=1, start_date=date(2026, 7, 1), end_date=date(2026, 7, 15)),
        SimpleNamespace(personnel_id=1, shift_id=2, start_date=date(2026, 7, 16), end_date=date(2026, 7, 31)),
    ]
    store = _FakeShiftStore(assignments, {1: _shift_record(1, "first"), 2: _shift_record(2, "second")})
    service = AttendanceSummaryService(database=None, shift_store=store)  # type: ignore[arg-type]

    shifts = service._effective_shifts(1, date(2026, 7, 15), date(2026, 7, 16))

    assert shifts[date(2026, 7, 15)].id == 1
    assert shifts[date(2026, 7, 16)].id == 2


def test_overnight_logs_remain_owned_by_assignment_start_day() -> None:
    overnight = AttendanceSummaryService._record_to_shift(_shift_record(3, "night", "22:00", "06:00"))
    logs = [SummaryLog(1, "123", 1, datetime(2026, 7, 28, 2, tzinfo=timezone.utc))]

    from app.core.attendance_summary_service import _logs_for_report_day

    assert _logs_for_report_day(logs, date(2026, 7, 27), overnight, use_shift_window=True) == logs
    assert _logs_for_report_day(logs, date(2026, 7, 28), overnight, use_shift_window=True) == []
