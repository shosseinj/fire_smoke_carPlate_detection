"""Legacy-compatible attendance summaries for detection-log reporting.

This module reproduces the calculations and response contracts of the old
``daily-summary``, ``monthly-summary`` and ``yearly-leave-summary`` endpoints
while reading the current PostgreSQL schema directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_FLOOR
from enum import Enum
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import jdatetime
from fastapi import HTTPException

from app.core.jalali_utils import parse_jalali_date
from app.database import Database, Row
from app.core.shift_store import ShiftStore
from app.time_utils import ensure_aware_utc, utc_now

DEFAULT_LOCAL_TZ_NAME = "Asia/Tehran"
WEEKDAY_FIELD = {
    0: "works_monday",
    1: "works_tuesday",
    2: "works_wednesday",
    3: "works_thursday",
    4: "works_friday",
    5: "works_saturday",
    6: "works_sunday",
}

REQUEST_EARNED_LEAVE = "earned_leave"
REQUEST_SICK_LEAVE = "sick_leave"
REQUEST_UNPAID_LEAVE = "unpaid_leave"
REQUEST_MISSION = "mission"
REQUEST_OVERTIME = "overtime"
ACCEPTED_REQUEST_STATUSES = ("accepted", "approved")
ANNUAL_LEAVE_DAYS = 30
MONTHLY_EARNED_LEAVE_DAYS = ANNUAL_LEAVE_DAYS / 12


class TimePeriod(str, Enum):
    TODAY = "today"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class SummaryShift:
    id: int
    shift_name: str
    start_time: time | None
    end_time: time | None
    timezone_name: str
    max_minutes_delay: int
    max_minutes_early: int
    max_overtime_hours: float
    works_monday: bool
    works_tuesday: bool
    works_wednesday: bool
    works_thursday: bool
    works_friday: bool
    works_saturday: bool
    works_sunday: bool


@dataclass(frozen=True, slots=True)
class SummaryPersonnel:
    id: int
    fname: str
    lname: str
    national_code: str
    department_id: int | None
    section_name: str | None
    shift_id: int | None
    shift: SummaryShift | None


@dataclass(frozen=True, slots=True)
class SummaryLog:
    id: int
    person: str
    personnel_id: int | None
    detection_time: datetime


@dataclass(frozen=True, slots=True)
class SummaryRequest:
    id: int
    personnel_id: int
    request_type: str
    duration_type: str | None
    start_date: date
    end_date: date | None
    start_time: time | None
    end_time: time | None
    duration_days: float | None
    duration_minutes: int | None
    status: str


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _as_time(value: Any) -> time | None:
    if value is None:
        return None
    if isinstance(value, time):
        return value.replace(tzinfo=None)
    text = str(value).strip()
    if not text:
        return None
    return time.fromisoformat(text)


def _as_utc_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return ensure_aware_utc(value)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return ensure_aware_utc(parsed)


def _daterange(start: date, end: date) -> Iterable[date]:
    for index in range((end - start).days + 1):
        yield start + timedelta(days=index)


def _shift_assignment_segments(
    shifts_by_day: dict[date, SummaryShift],
    start_day: date,
    end_day: date,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    segment_start: date | None = None
    segment_shift: SummaryShift | None = None
    previous_day: date | None = None

    for day in _daterange(start_day, end_day):
        shift = shifts_by_day.get(day)
        if shift is not None and segment_shift is not None and shift.id == segment_shift.id:
            previous_day = day
            continue
        if segment_shift is not None and segment_start is not None and previous_day is not None:
            segments.append(
                {
                    "shift_id": segment_shift.id,
                    "shift_name": segment_shift.shift_name,
                    "start_date": segment_start.isoformat(),
                    "end_date": previous_day.isoformat(),
                }
            )
        segment_shift = shift
        segment_start = day if shift is not None else None
        previous_day = day

    if segment_shift is not None and segment_start is not None and previous_day is not None:
        segments.append(
            {
                "shift_id": segment_shift.id,
                "shift_name": segment_shift.shift_name,
                "start_date": segment_start.isoformat(),
                "end_date": previous_day.isoformat(),
            }
        )
    return segments


def _local_day_bounds_utc(day: date, timezone_name: str) -> tuple[datetime, datetime]:
    tz = ZoneInfo(timezone_name)
    local_start = datetime.combine(day, time.min, tzinfo=tz)
    return local_start.astimezone(timezone.utc), (local_start + timedelta(days=1)).astimezone(timezone.utc)


def _local_date_range_bounds_utc(
    start_day: date,
    end_day: date,
    timezone_name: str,
) -> tuple[datetime, datetime]:
    if end_day < start_day:
        raise ValueError("تاریخ پایان نباید قبل از تاریخ شروع باشد")
    start_utc, _ = _local_day_bounds_utc(start_day, timezone_name)
    _, end_utc = _local_day_bounds_utc(end_day, timezone_name)
    return start_utc, end_utc


def _utc_to_local(value: datetime, timezone_name: str = DEFAULT_LOCAL_TZ_NAME) -> datetime:
    return ensure_aware_utc(value).astimezone(ZoneInfo(timezone_name))


def _jalali_date_string(day: date) -> str:
    jday = jdatetime.date.fromgregorian(date=day)
    return f"{jday.year:04d}-{jday.month:02d}-{jday.day:02d}"


def _shift_timezone_name(shift: SummaryShift | None) -> str:
    return getattr(shift, "timezone_name", None) or DEFAULT_LOCAL_TZ_NAME


def _to_shift_local(value: datetime, shift: SummaryShift | None = None) -> datetime:
    return _utc_to_local(value, _shift_timezone_name(shift))


def _jalali_month_range(jalali_year: int, jalali_month: int) -> tuple[date, date]:
    if not 1 <= jalali_month <= 12:
        raise HTTPException(status_code=400, detail="ماه شمسی باید بین ۱ تا ۱۲ باشد")
    j_start = jdatetime.date(jalali_year, jalali_month, 1)
    if jalali_month == 12:
        j_end = jdatetime.date(jalali_year + 1, 1, 1) - jdatetime.timedelta(days=1)
    else:
        j_end = jdatetime.date(jalali_year, jalali_month + 1, 1) - jdatetime.timedelta(days=1)
    return j_start.togregorian(), j_end.togregorian()


def _jalali_moving_month_range(
    jalali_year: int,
    jalali_month: int,
    move_days: int,
) -> tuple[date, date]:
    if not 1 <= jalali_month <= 12:
        raise HTTPException(status_code=400, detail="ماه شمسی باید بین ۱ تا ۱۲ باشد")
    if move_days == 0:
        return _jalali_month_range(jalali_year, jalali_month)
    if not 2 <= move_days <= 29:
        raise HTTPException(
            status_code=400,
            detail="move_days باید صفر یا عددی بین ۲ تا ۲۹ باشد",
        )

    end_day = 30 - move_days
    start_day = end_day + 1
    if jalali_month == 1:
        previous_year, previous_month = jalali_year - 1, 12
    else:
        previous_year, previous_month = jalali_year, jalali_month - 1
    try:
        j_start = jdatetime.date(previous_year, previous_month, start_day)
        j_end = jdatetime.date(jalali_year, jalali_month, end_day)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"بازه ماهانه شمسی نامعتبر است: {exc}",
        ) from exc
    return j_start.togregorian(), j_end.togregorian()


def _shift_window(day: date, shift: SummaryShift | None) -> tuple[datetime, datetime] | None:
    if not shift or not shift.start_time or not shift.end_time:
        return None
    tz = ZoneInfo(_shift_timezone_name(shift))
    start_dt = datetime.combine(day, shift.start_time, tzinfo=tz)
    end_dt = datetime.combine(day, shift.end_time, tzinfo=tz)
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)
    return start_dt, end_dt


def _shift_missing_message(shift: SummaryShift | None) -> str | None:
    if not shift:
        return "برای این کارمند شیفت کاری تعریف نشده است"
    if not shift.start_time or not shift.end_time:
        return "زمان شروع یا پایان شیفت برای این کارمند کامل نیست"
    return None


def _shift_expected_on_day(day: date, shift: SummaryShift | None) -> bool:
    if not shift:
        return False
    return bool(getattr(shift, WEEKDAY_FIELD[day.weekday()], False))


def _holiday_to_gregorian(holiday_date: date, jalali_year: int | None = None) -> date:
    if holiday_date.year < 1700:
        year = jalali_year if jalali_year is not None else holiday_date.year
        return jdatetime.date(year, holiday_date.month, holiday_date.day).togregorian()
    return holiday_date


def _request_overlaps_day(request_item: SummaryRequest, day: date) -> bool:
    request_end = request_item.end_date or request_item.start_date
    return request_item.start_date <= day <= request_end


def _hourly_request_intervals_for_day(
    requests: list[SummaryRequest],
    day: date,
    shift_start: datetime,
    shift_end: datetime,
) -> dict[str, list[tuple[datetime, datetime]]]:
    intervals: dict[str, list[tuple[datetime, datetime]]] = {
        "earned_leave_minutes": [],
        "sick_leave_minutes": [],
        "unpaid_leave_minutes": [],
        "mission_minutes": [],
    }
    key_by_type = {
        REQUEST_EARNED_LEAVE: "earned_leave_minutes",
        REQUEST_SICK_LEAVE: "sick_leave_minutes",
        REQUEST_UNPAID_LEAVE: "unpaid_leave_minutes",
        REQUEST_MISSION: "mission_minutes",
    }
    for request_item in requests:
        if (
            request_item.duration_type != "hourly"
            or request_item.start_time is None
            or request_item.end_time is None
        ):
            continue
        key = key_by_type.get(request_item.request_type)
        if key is None:
            continue
        request_start = datetime.combine(day, request_item.start_time, tzinfo=shift_start.tzinfo)
        request_end = datetime.combine(day, request_item.end_time, tzinfo=shift_start.tzinfo)
        if request_end <= request_start:
            request_end += timedelta(days=1)
        clipped_start = max(request_start, shift_start)
        clipped_end = min(request_end, shift_end)
        if clipped_end > clipped_start:
            intervals[key].append((clipped_start, clipped_end))
    return intervals


def _merge_intervals(intervals: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: item[0])
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _interval_minutes(intervals: list[tuple[datetime, datetime]]) -> int:
    return sum(
        max(0, int((end - start).total_seconds() // 60))
        for start, end in _merge_intervals(intervals)
    )


def _overlap_minutes(
    target_intervals: list[tuple[datetime, datetime]],
    approved_intervals: list[tuple[datetime, datetime]],
) -> int:
    overlaps: list[tuple[datetime, datetime]] = []
    for target_start, target_end in target_intervals:
        for approved_start, approved_end in approved_intervals:
            start = max(target_start, approved_start)
            end = min(target_end, approved_end)
            if end > start:
                overlaps.append((start, end))
    return _interval_minutes(overlaps)


def _time_to_hhmm(value: datetime | None) -> str | None:
    return value.strftime("%H:%M") if value is not None else None


def _daily_summary_stats(
    day: date,
    day_logs: list[SummaryLog],
    shift: SummaryShift | None,
    requests: list[SummaryRequest],
    is_workday: bool,
) -> dict[str, Any]:
    window = _shift_window(day, shift)
    if window is None:
        return {
            "message": "برای این کارمند شیفت کاری تعریف نشده است",
            "first_detection": None,
            "last_detection": None,
            "middle_detections": [],
            "first_last_span_time": None,
            "raw_worked_time": None,
            "net_worked_time": None,
            "net_worked_time_with_overtime": None,
            "delay_minutes": 0,
            "early_leave_minutes": 0,
            "overtime_minutes": 0,
            "holiday_overtime_minutes": 0,
            "illegal_presence_minutes": 0,
            "in_between_absence_minutes": 0,
            "total_absence_time": 0,
            "request_minutes": {
                "earned_leave_minutes": 0,
                "sick_leave_minutes": 0,
                "unpaid_leave_minutes": 0,
                "mission_minutes": 0,
            },
        }

    shift_start, shift_end = window
    local_logs = sorted(day_logs, key=lambda log: _to_shift_local(log.detection_time, shift))
    local_times = [_to_shift_local(log.detection_time, shift) for log in local_logs]
    first_dt = local_times[0] if local_times else None
    last_dt = local_times[-1] if local_times else None
    middle = local_times[1:-1] if len(local_times) > 2 else []

    request_intervals = _hourly_request_intervals_for_day(
        requests,
        day,
        shift_start,
        shift_end,
    )
    all_request_intervals = [
        interval for values in request_intervals.values() for interval in values
    ]
    request_minutes = {
        key: _interval_minutes(values) for key, values in request_intervals.items()
    }

    max_delay = int(getattr(shift, "max_minutes_delay", 0) or 0)
    max_early = int(getattr(shift, "max_minutes_early", 0) or 0)

    delay_interval: list[tuple[datetime, datetime]] = []
    early_interval: list[tuple[datetime, datetime]] = []
    if is_workday:
        effective_arrival = first_dt or shift_end
        delay_end = min(max(effective_arrival, shift_start), shift_end)
        actual_delay_minutes = max(
            0,
            int((delay_end - shift_start).total_seconds() // 60),
        )
        if actual_delay_minutes > max_delay:
            delay_interval = [(shift_start, delay_end)]

        if last_dt is not None:
            early_start = max(min(last_dt, shift_end), shift_start)
            early_end = shift_end - timedelta(minutes=max_early)
            if early_end > early_start:
                early_interval = [(early_start, early_end)]

    internal_absence_intervals: list[tuple[datetime, datetime]] = []
    for index in range(1, len(local_times) - 1, 2):
        gap_start = max(local_times[index], shift_start)
        gap_end = min(local_times[index + 1], shift_end)
        if gap_end > gap_start:
            internal_absence_intervals.append((gap_start, gap_end))

    delay_minutes = max(
        0,
        _interval_minutes(delay_interval)
        - _overlap_minutes(delay_interval, all_request_intervals),
    )
    early_leave_minutes = max(
        0,
        _interval_minutes(early_interval)
        - _overlap_minutes(early_interval, all_request_intervals),
    )
    in_between_absence_minutes = max(
        0,
        _interval_minutes(internal_absence_intervals)
        - _overlap_minutes(internal_absence_intervals, all_request_intervals),
    )

    even_detection_count = len(local_times) > 0 and len(local_times) % 2 == 0
    paired_presence_intervals: list[tuple[datetime, datetime]] = []
    for index in range(0, len(local_times) - 1, 2):
        presence_start = local_times[index]
        presence_end = local_times[index + 1]
        if presence_end > presence_start:
            paired_presence_intervals.append((presence_start, presence_end))

    illegal_presence_intervals: list[tuple[datetime, datetime]] = []
    overtime_intervals: list[tuple[datetime, datetime]] = []
    holiday_overtime_minutes = 0
    if is_workday:
        for presence_start, presence_end in paired_presence_intervals:
            illegal_end = min(presence_end, shift_start)
            if illegal_end > presence_start:
                illegal_presence_intervals.append((presence_start, illegal_end))
            overtime_start = max(presence_start, shift_end)
            if presence_end > overtime_start:
                overtime_intervals.append((overtime_start, presence_end))
    else:
        holiday_overtime_minutes = _interval_minutes(paired_presence_intervals)

    illegal_presence_minutes = _interval_minutes(illegal_presence_intervals)
    overtime_minutes = _interval_minutes(overtime_intervals)

    raw_worked_time: int | None = None
    first_last_span_time: int | None = None
    net_worked_time: int | None = None
    if even_detection_count and first_dt and last_dt:
        first_last_span_time = max(
            0,
            int((last_dt - first_dt).total_seconds() // 60),
        )
        raw_worked_time = _interval_minutes(paired_presence_intervals)
        if is_workday:
            regular_presence_intervals: list[tuple[datetime, datetime]] = []
            for presence_start, presence_end in paired_presence_intervals:
                regular_start = max(presence_start, shift_start)
                regular_end = min(presence_end, shift_end)
                if regular_end > regular_start:
                    regular_presence_intervals.append((regular_start, regular_end))
            net_worked_time = _interval_minutes(regular_presence_intervals)
        else:
            net_worked_time = raw_worked_time

    net_worked_time_with_overtime = (
        net_worked_time + overtime_minutes if net_worked_time is not None else None
    )
    total_absence_time = delay_minutes + early_leave_minutes + in_between_absence_minutes
    message = None
    if local_times and not even_detection_count:
        message = "تعداد ترددهای این روز فرد است؛ زمان کار خام و خالص قابل محاسبه نیست"

    return {
        "message": message,
        "first_detection": first_dt,
        "last_detection": last_dt,
        "middle_detections": middle,
        "first_last_span_time": first_last_span_time,
        "raw_worked_time": raw_worked_time,
        "net_worked_time": net_worked_time,
        "delay_minutes": delay_minutes,
        "early_leave_minutes": early_leave_minutes,
        "overtime_minutes": overtime_minutes,
        "holiday_overtime_minutes": holiday_overtime_minutes,
        "illegal_presence_minutes": illegal_presence_minutes,
        "net_worked_time_with_overtime": net_worked_time_with_overtime,
        "in_between_absence_minutes": in_between_absence_minutes,
        "total_absence_time": total_absence_time,
        "request_minutes": request_minutes,
    }


def _request_minutes_for_day(
    requests: list[SummaryRequest],
    day: date,
    shift_start: datetime,
    shift_end: datetime,
) -> dict[str, float]:
    totals = {
        "earned_leave_minutes": 0.0,
        "sick_leave_minutes": 0.0,
        "unpaid_leave_minutes": 0.0,
        "mission_minutes": 0.0,
        "approved_overtime_minutes": 0.0,
        "hourly_leave_minutes": 0.0,
    }
    shift_minutes = max(0, int((shift_end - shift_start).total_seconds() // 60))
    key_by_type = {
        REQUEST_EARNED_LEAVE: "earned_leave_minutes",
        REQUEST_SICK_LEAVE: "sick_leave_minutes",
        REQUEST_UNPAID_LEAVE: "unpaid_leave_minutes",
        REQUEST_MISSION: "mission_minutes",
        REQUEST_OVERTIME: "approved_overtime_minutes",
    }

    for request_item in requests:
        key = key_by_type.get(request_item.request_type)
        if (
            request_item.request_type == REQUEST_EARNED_LEAVE
            and request_item.duration_type == "hourly"
        ):
            key = "hourly_leave_minutes"
        if key is None:
            continue

        if request_item.duration_type == "daily":
            requested_minutes = shift_minutes
        elif request_item.start_time is not None and request_item.end_time is not None:
            request_start = datetime.combine(day, request_item.start_time, tzinfo=shift_start.tzinfo)
            request_end = datetime.combine(day, request_item.end_time, tzinfo=shift_start.tzinfo)
            if request_end <= request_start:
                request_end += timedelta(days=1)
            overlap_start = max(request_start, shift_start)
            overlap_end = min(request_end, shift_end)
            requested_minutes = max(
                0,
                int((overlap_end - overlap_start).total_seconds() // 60),
            )
        else:
            requested_minutes = float(request_item.duration_minutes or 0)
        totals[key] += requested_minutes
    return totals


def _minutes_to_hhmm(total_minutes: float | int | None) -> str | None:
    if total_minutes is None:
        return None
    minutes = max(0, int(round(float(total_minutes))))
    hours, remainder = divmod(minutes, 60)
    return f"{hours:02d}:{remainder:02d}"


def _shift_crosses_midnight(shift: SummaryShift | None) -> bool:
    return bool(
        shift
        and shift.start_time
        and shift.end_time
        and shift.end_time <= shift.start_time
    )


def _build_monthly_logs_by_day(
    person_logs: list[SummaryLog],
    start_day: date,
    end_day: date,
    shift: SummaryShift | None,
) -> dict[date, list[SummaryLog]]:
    mapped: dict[date, list[SummaryLog]] = {}
    if not shift:
        return mapped

    crosses_midnight = _shift_crosses_midnight(shift)
    scheduled_windows: dict[date, tuple[datetime, datetime, datetime, datetime]] = {}
    if crosses_midnight:
        for report_day in _daterange(start_day - timedelta(days=1), end_day):
            if not _shift_expected_on_day(report_day, shift):
                continue
            window = _shift_window(report_day, shift)
            if window is None:
                continue
            shift_start, shift_end = window
            shift_duration_minutes = int((shift_end - shift_start).total_seconds() // 60)
            gap_minutes = max(0, (24 * 60) - shift_duration_minutes)
            configured_overtime = max(
                0,
                int(float(shift.max_overtime_hours or 0) * 60),
            )
            buffer_minutes = min(configured_overtime, gap_minutes // 2)
            scheduled_windows[report_day] = (
                shift_start - timedelta(minutes=buffer_minutes),
                shift_end + timedelta(minutes=buffer_minutes),
                shift_start,
                shift_end,
            )

    for log in person_logs:
        local_dt = _to_shift_local(log.detection_time, shift)
        local_day = local_dt.date()
        assigned_day: date | None = None
        if crosses_midnight:
            candidates: list[tuple[float, date]] = []
            for candidate_day in (local_day - timedelta(days=1), local_day):
                window = scheduled_windows.get(candidate_day)
                if window is None:
                    continue
                window_start, window_end, shift_start, shift_end = window
                if not (window_start <= local_dt < window_end):
                    continue
                if shift_start <= local_dt <= shift_end:
                    distance = 0.0
                elif local_dt < shift_start:
                    distance = (shift_start - local_dt).total_seconds()
                else:
                    distance = (local_dt - shift_end).total_seconds()
                candidates.append((distance, candidate_day))
            if candidates:
                candidates.sort(key=lambda item: (item[0], item[1]))
                assigned_day = candidates[0][1]
        if assigned_day is None:
            assigned_day = local_day
        if start_day <= assigned_day <= end_day:
            mapped.setdefault(assigned_day, []).append(log)

    for report_day, day_logs in list(mapped.items()):
        day_logs.sort(key=lambda item: _to_shift_local(item.detection_time, shift))
        if len(day_logs) > 2:
            mapped[report_day] = [day_logs[0], day_logs[-1]]
    return mapped


def _monthly_presence_minutes(
    day: date,
    day_logs: list[SummaryLog],
    shift: SummaryShift | None,
    is_workday: bool,
) -> dict[str, int]:
    if not day_logs or not shift:
        return {"regular": 0, "overtime": 0, "holiday_overtime": 0}
    local_logs = sorted(day_logs, key=lambda log: _to_shift_local(log.detection_time, shift))
    first_dt = _to_shift_local(local_logs[0].detection_time, shift)
    last_dt = _to_shift_local(local_logs[-1].detection_time, shift)
    total_span = max(0, int((last_dt - first_dt).total_seconds() // 60))
    if not is_workday:
        return {"regular": 0, "overtime": 0, "holiday_overtime": total_span}
    window = _shift_window(day, shift)
    if window is None:
        return {"regular": 0, "overtime": 0, "holiday_overtime": 0}
    shift_start, shift_end = window
    overlap_start = max(first_dt, shift_start)
    overlap_end = min(last_dt, shift_end)
    regular = max(0, int((overlap_end - overlap_start).total_seconds() // 60))
    overtime = max(0, total_span - regular)
    return {"regular": regular, "overtime": overtime, "holiday_overtime": 0}


def _full_day_request_category(requests: list[SummaryRequest]) -> str | None:
    available: set[str] = set()
    for request_item in requests:
        if request_item.duration_type != "daily":
            continue
        if request_item.request_type in {
            REQUEST_MISSION,
            REQUEST_SICK_LEAVE,
            REQUEST_EARNED_LEAVE,
            REQUEST_UNPAID_LEAVE,
        }:
            available.add(request_item.request_type)
    for request_type in (
        REQUEST_MISSION,
        REQUEST_SICK_LEAVE,
        REQUEST_EARNED_LEAVE,
        REQUEST_UNPAID_LEAVE,
    ):
        if request_type in available:
            return request_type
    return None


def _daily_request_status(requests: list[SummaryRequest]) -> str | None:
    category = _full_day_request_category(requests)
    if category is not None:
        return category
    daily_requests = sorted(
        (request for request in requests if request.duration_type == "daily"),
        key=lambda request: request.id,
    )
    return daily_requests[0].request_type if daily_requests else None


def _empty_shift_stats(message: str | None = None) -> dict[str, Any]:
    return {
        "first_detection": None,
        "last_detection": None,
        "delay_minutes": 0,
        "early_leave_minutes": 0,
        "overtime_minutes": 0,
        "interval_absence_minutes": 0,
        "absence_minutes": 0,
        "total_span_minutes": 0,
        "regular_work_minutes": 0,
        "expected_work_minutes": 0,
        "message": message,
    }


def _compute_shift_day_stats(
    day: date,
    day_logs: list[SummaryLog],
    shift: SummaryShift | None,
) -> dict[str, Any]:
    window = _shift_window(day, shift)
    if window is None:
        return _empty_shift_stats("برای این کارمند شیفت کاری تعریف نشده است")
    shift_start, shift_end = window
    max_delay = shift.max_minutes_delay if shift and shift.max_minutes_delay is not None else 0
    max_early = shift.max_minutes_early if shift and shift.max_minutes_early is not None else 0
    expected_minutes = max(0, int((shift_end - shift_start).total_seconds() // 60))
    if not day_logs:
        return {
            "first_detection": None,
            "last_detection": None,
            "delay_minutes": 0,
            "early_leave_minutes": 0,
            "overtime_minutes": 0,
            "interval_absence_minutes": 0,
            "absence_minutes": expected_minutes,
            "total_span_minutes": 0,
            "regular_work_minutes": 0,
            "expected_work_minutes": expected_minutes,
            "message": None,
        }
    local_logs = sorted(day_logs, key=lambda log: _to_shift_local(log.detection_time, shift))
    first_dt = _to_shift_local(local_logs[0].detection_time, shift)
    last_dt = _to_shift_local(local_logs[-1].detection_time, shift)
    actual_delay_minutes = max(
        0,
        int((first_dt - shift_start).total_seconds() // 60),
    )
    delay_minutes = (
        actual_delay_minutes
        if actual_delay_minutes > int(max_delay or 0)
        else 0
    )
    early_leave_minutes = max(
        0,
        int((shift_end - last_dt).total_seconds() // 60) - int(max_early or 0),
    )
    overtime_minutes = max(0, int((last_dt - shift_end).total_seconds() // 60))
    interval_absence_minutes = 0
    if len(local_logs) >= 2:
        for previous, following in zip(local_logs, local_logs[1:]):
            gap_start = max(_to_shift_local(previous.detection_time, shift), shift_start)
            gap_end = min(_to_shift_local(following.detection_time, shift), shift_end)
            if gap_end > gap_start:
                interval_absence_minutes += int((gap_end - gap_start).total_seconds() // 60)
    total_span_minutes = max(0, int((last_dt - first_dt).total_seconds() // 60))
    regular_work_minutes = max(
        0,
        min(total_span_minutes, expected_minutes) - interval_absence_minutes,
    )
    absence_minutes = max(
        0,
        delay_minutes + early_leave_minutes + interval_absence_minutes,
    )
    return {
        "first_detection": first_dt,
        "last_detection": last_dt,
        "delay_minutes": delay_minutes,
        "early_leave_minutes": early_leave_minutes,
        "overtime_minutes": overtime_minutes,
        "interval_absence_minutes": interval_absence_minutes,
        "absence_minutes": absence_minutes,
        "total_span_minutes": total_span_minutes,
        "regular_work_minutes": regular_work_minutes,
        "expected_work_minutes": expected_minutes,
        "message": None,
    }


def _logs_for_report_day(
    person_logs: list[SummaryLog],
    day: date,
    shift: SummaryShift | None,
    use_shift_window: bool = True,
) -> list[SummaryLog]:
    tz = ZoneInfo(_shift_timezone_name(shift))
    if use_shift_window and _shift_crosses_midnight(shift):
        window = _shift_window(day, shift)
        if window is None:
            return []
        window_start, window_end = window
    else:
        window_start = datetime.combine(day, time.min, tzinfo=tz)
        window_end = window_start + timedelta(days=1)
    selected = []
    for log in person_logs:
        local_dt = _to_shift_local(log.detection_time, shift)
        if window_start <= local_dt < window_end:
            selected.append(log)
    return sorted(selected, key=lambda item: _to_shift_local(item.detection_time, shift))


def _calculate_monthly_earned_leave_days(
    absent_days: int,
    unpaid_leave_days: int,
    sick_leave_days: int,
) -> float:
    excluded_days = max(
        0.0,
        min(30.0, float(absent_days + unpaid_leave_days + sick_leave_days)),
    )
    eligible_ratio = (30.0 - excluded_days) / 30.0
    return round(MONTHLY_EARNED_LEAVE_DAYS * eligible_ratio, 4)


def _floor_to_two_decimal_places(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_FLOOR))


def _earned_leave_request_days_in_range(
    request_item: SummaryRequest,
    range_start: date,
    range_end: date,
    shift: SummaryShift | None,
    holiday_days: set[date],
) -> float:
    if request_item.duration_type != "daily" or shift is None:
        return 0.0
    request_start = request_item.start_date
    request_end = request_item.end_date or request_item.start_date
    overlap_start = max(request_start, range_start)
    overlap_end = min(request_end, range_end)
    if overlap_start > overlap_end:
        return 0.0
    return float(
        sum(
            1
            for day in _daterange(overlap_start, overlap_end)
            if _shift_expected_on_day(day, shift) and day not in holiday_days
        )
    )


def _yearly_leave_month_attendance(
    personnel: SummaryPersonnel,
    person_logs: list[SummaryLog],
    month_start: date,
    month_end: date,
    holiday_days: set[date],
    requests_by_day: dict[date, list[SummaryRequest]],
    shifts_by_day: dict[date, SummaryShift] | None = None,
) -> dict[str, Any]:
    month_days = (month_end - month_start).days + 1
    shifts_by_day = shifts_by_day or {}
    off_days = 0
    no_shift_days = 0
    custom_holidays = 0
    total_holidays = 0
    month_working_days = 0
    present_days = 0
    mission_days = 0
    earned_leave_days = 0
    sick_leave_days = 0
    unpaid_leave_days = 0

    for day in _daterange(month_start, month_end):
        shift = shifts_by_day.get(day)
        is_custom_holiday = day in holiday_days
        if shift is None:
            no_shift_days += 1
            if is_custom_holiday:
                custom_holidays += 1
                total_holidays += 1
            continue
        is_shift_day = _shift_expected_on_day(day, shift)
        is_workday = is_shift_day and not is_custom_holiday
        if not is_shift_day:
            off_days += 1
        if is_custom_holiday:
            custom_holidays += 1
        if not is_shift_day or is_custom_holiday:
            total_holidays += 1
        if not is_workday:
            continue
        month_working_days += 1
        if _logs_for_report_day(person_logs, day, shift, use_shift_window=True):
            present_days += 1
            continue
        request_category = _full_day_request_category(requests_by_day.get(day, []))
        if request_category == REQUEST_MISSION:
            mission_days += 1
        elif request_category == REQUEST_SICK_LEAVE:
            sick_leave_days += 1
        elif request_category == REQUEST_EARNED_LEAVE:
            earned_leave_days += 1
        elif request_category == REQUEST_UNPAID_LEAVE:
            unpaid_leave_days += 1

    final_working_days = present_days + mission_days + earned_leave_days + sick_leave_days
    absent_days = max(
        0,
        month_working_days - final_working_days - unpaid_leave_days,
    )
    return {
        "month_days": month_days,
        "off_days": off_days,
        "no_shift_days": no_shift_days,
        "custom_holidays": custom_holidays,
        "total_holidays": total_holidays,
        "month_working_days": month_working_days,
        "present_days": present_days,
        "mission_days": mission_days,
        "earned_leave_days": earned_leave_days,
        "sick_leave_days": sick_leave_days,
        "unpaid_leave_days": unpaid_leave_days,
        "final_working_days": final_working_days,
        "absent_days": absent_days,
        "message": (
            "برای بخشی از بازه گزارش، شیفت کاری تعریف نشده است"
            if len(shifts_by_day) < month_days else None
        ),
    }


class AttendanceSummaryService:
    """Read current PostgreSQL data and reproduce legacy report behavior."""

    def __init__(self, database: Database, shift_store: ShiftStore | None = None) -> None:
        self.database = database
        self.shift_store = shift_store or (ShiftStore(database) if database is not None else None)

    @staticmethod
    def _record_to_shift(record: Any | None) -> SummaryShift | None:
        if record is None:
            return None
        return SummaryShift(
            id=int(record.id), shift_name=str(record.shift_name),
            start_time=_as_time(record.start_time), end_time=_as_time(record.end_time),
            timezone_name=str(record.timezone_name or DEFAULT_LOCAL_TZ_NAME),
            max_minutes_delay=int(record.max_minutes_delay or 0),
            max_minutes_early=int(record.max_minutes_early or 0),
            max_overtime_hours=float(record.max_overtime_hours or 0),
            works_monday=bool(record.works_monday), works_tuesday=bool(record.works_tuesday),
            works_wednesday=bool(record.works_wednesday), works_thursday=bool(record.works_thursday),
            works_friday=bool(record.works_friday), works_saturday=bool(record.works_saturday),
            works_sunday=bool(record.works_sunday),
        )

    def _effective_shifts(self, personnel_id: int, start_day: date, end_day: date) -> dict[date, SummaryShift]:
        if self.shift_store is None:
            return {}
        assignments = self.shift_store.list_assignments(personnel_id, start_day, end_day)
        shifts: dict[date, SummaryShift] = {}
        records: dict[int, SummaryShift | None] = {}
        for assignment in assignments:
            shift_id = int(assignment.shift_id)
            if shift_id not in records:
                records[shift_id] = self._record_to_shift(self.shift_store.get(shift_id))
            shift = records[shift_id]
            if shift is None:
                continue
            first = max(start_day, _as_date(assignment.start_date))
            last = min(end_day, _as_date(assignment.end_date))
            for day in _daterange(first, last):
                shifts[day] = shift
        return shifts

    def _person_shifts(self, person: SummaryPersonnel, start_day: date, end_day: date) -> dict[date, SummaryShift]:
        return self._effective_shifts(person.id, start_day, end_day)

    @staticmethod
    def _row_to_shift(row: Row) -> SummaryShift | None:
        if row.get("shift_id") is None:
            return None
        return SummaryShift(
            id=int(row["shift_id"]),
            shift_name=str(row.get("shift_name") or ""),
            start_time=_as_time(row.get("shift_start_time")),
            end_time=_as_time(row.get("shift_end_time")),
            timezone_name=str(row.get("shift_timezone_name") or DEFAULT_LOCAL_TZ_NAME),
            max_minutes_delay=int(row.get("shift_max_minutes_delay") or 0),
            max_minutes_early=int(row.get("shift_max_minutes_early") or 0),
            max_overtime_hours=float(row.get("shift_max_overtime_hours") or 0),
            works_monday=bool(row.get("shift_works_monday")),
            works_tuesday=bool(row.get("shift_works_tuesday")),
            works_wednesday=bool(row.get("shift_works_wednesday")),
            works_thursday=bool(row.get("shift_works_thursday")),
            works_friday=bool(row.get("shift_works_friday")),
            works_saturday=bool(row.get("shift_works_saturday")),
            works_sunday=bool(row.get("shift_works_sunday")),
        )

    def _personnel(
        self,
        personnel_id: str | None = None,
        section_id: int | None = None,
        shift_id: int | None = None,
    ) -> list[SummaryPersonnel]:
        where: list[str] = []
        params: list[Any] = []
        if personnel_id:
            personnel_text = str(personnel_id)
            if personnel_text.isdigit():
                where.append("(p.id = ? OR p.national_code = ?)")
                params.extend([int(personnel_text), personnel_text])
            else:
                where.append("p.national_code = ?")
                params.append(personnel_text)
        if section_id is not None:
            where.append("p.department_id = ?")
            params.append(section_id)
        where_sql = f" WHERE {' AND '.join(where)}" if where else ""
        sql = (
            "SELECT p.id, p.fname, p.lname, p.national_code, p.department_id, "
            "p.shift_id, s.name AS section_name, "
            "ws.shift_name, ws.start_time AS shift_start_time, "
            "ws.end_time AS shift_end_time, ws.timezone_name AS shift_timezone_name, "
            "ws.max_minutes_delay AS shift_max_minutes_delay, "
            "ws.max_minutes_early AS shift_max_minutes_early, "
            "ws.max_overtime_hours AS shift_max_overtime_hours, "
            "ws.works_monday AS shift_works_monday, "
            "ws.works_tuesday AS shift_works_tuesday, "
            "ws.works_wednesday AS shift_works_wednesday, "
            "ws.works_thursday AS shift_works_thursday, "
            "ws.works_friday AS shift_works_friday, "
            "ws.works_saturday AS shift_works_saturday, "
            "ws.works_sunday AS shift_works_sunday "
            "FROM personnel p "
            "LEFT JOIN sections s ON s.id = p.department_id "
            "LEFT JOIN work_shifts ws ON ws.id = p.shift_id"
            f"{where_sql} ORDER BY p.lname, p.fname"
        )
        with self.database.connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [
            SummaryPersonnel(
                id=int(row["id"]),
                fname=str(row.get("fname") or ""),
                lname=str(row.get("lname") or ""),
                national_code=str(row.get("national_code") or ""),
                department_id=(
                    int(row["department_id"])
                    if row.get("department_id") is not None
                    else None
                ),
                section_name=(
                    str(row["section_name"])
                    if row.get("section_name") is not None
                    else None
                ),
                shift_id=(int(row["shift_id"]) if row.get("shift_id") is not None else None),
                shift=self._row_to_shift(row),
            )
            for row in rows
        ]

    def _logs(
        self,
        personnel: list[SummaryPersonnel],
        from_dt: datetime,
        to_dt: datetime,
    ) -> dict[str, list[SummaryLog]]:
        codes = [item.national_code for item in personnel if item.national_code]
        ids = [item.id for item in personnel]
        if not codes and not ids:
            return {}
        code_by_id = {item.id: item.national_code for item in personnel}
        id_by_code = {item.national_code: item.id for item in personnel if item.national_code}

        rows_by_id: dict[int, Row] = {}
        with self.database.connection() as connection:
            if ids:
                placeholders = ", ".join("?" for _ in ids)
                rows = connection.execute(
                    "SELECT d.id, d.person, d.personnel_id, d.detection_time "
                    "FROM detection_logs d "
                    "WHERE d.counts_for_attendance = 1 "
                    "AND d.detection_time >= ? AND d.detection_time < ? "
                    f"AND d.personnel_id IN ({placeholders}) "
                    "ORDER BY d.detection_time ASC",
                    [from_dt, to_dt, *ids],
                ).fetchall()
                rows_by_id.update({int(row["id"]): row for row in rows})
            if codes:
                placeholders = ", ".join("?" for _ in codes)
                rows = connection.execute(
                    "SELECT d.id, d.person, d.personnel_id, d.detection_time "
                    "FROM detection_logs d "
                    "WHERE d.counts_for_attendance = 1 "
                    "AND d.detection_time >= ? AND d.detection_time < ? "
                    f"AND d.person IN ({placeholders}) "
                    "ORDER BY d.detection_time ASC",
                    [from_dt, to_dt, *codes],
                ).fetchall()
                rows_by_id.update({int(row["id"]): row for row in rows})

        rows = sorted(
            rows_by_id.values(),
            key=lambda row: (_as_utc_datetime(row["detection_time"]), int(row["id"])),
        )

        grouped: dict[str, list[SummaryLog]] = {}
        for row in rows:
            raw_person = str(row.get("person") or "")
            raw_personnel_id = row.get("personnel_id")
            code: str | None = raw_person if raw_person in id_by_code else None
            if code is None and raw_personnel_id is not None:
                code = code_by_id.get(int(raw_personnel_id))
            if code is None:
                continue
            grouped.setdefault(code, []).append(
                SummaryLog(
                    id=int(row["id"]),
                    person=raw_person,
                    personnel_id=(
                        int(raw_personnel_id) if raw_personnel_id is not None else None
                    ),
                    detection_time=_as_utc_datetime(row["detection_time"]),
                )
            )
        return grouped

    def _holiday_dates(self, start_day: date, end_day: date) -> set[date]:
        with self.database.connection() as connection:
            rows = connection.execute(
                "SELECT date_value, every_year FROM holidays WHERE is_active = 1"
            ).fetchall()
        days: set[date] = set()
        jalali_years = {
            jdatetime.date.fromgregorian(date=day).year
            for day in (start_day, end_day)
        }
        gregorian_years = {start_day.year, end_day.year}
        for row in rows:
            raw_date = row.get("date_value")
            if raw_date is None:
                continue
            holiday_date = _as_date(raw_date)
            if bool(row.get("every_year")):
                candidates: list[date] = []
                if holiday_date.year < 1700:
                    for jalali_year in jalali_years:
                        try:
                            candidates.append(
                                _holiday_to_gregorian(holiday_date, jalali_year)
                            )
                        except ValueError:
                            continue
                else:
                    for gregorian_year in gregorian_years:
                        try:
                            candidates.append(
                                date(
                                    gregorian_year,
                                    holiday_date.month,
                                    holiday_date.day,
                                )
                            )
                        except ValueError:
                            continue
                days.update(
                    candidate
                    for candidate in candidates
                    if start_day <= candidate <= end_day
                )
            else:
                gregorian_date = _holiday_to_gregorian(holiday_date)
                if start_day <= gregorian_date <= end_day:
                    days.add(gregorian_date)
        return days

    def _accepted_requests(
        self,
        personnel_ids: list[int],
        start_day: date,
        end_day: date,
        *,
        request_type: str | None = None,
    ) -> list[SummaryRequest]:
        if not personnel_ids:
            return []
        id_placeholders = ", ".join("?" for _ in personnel_ids)
        status_placeholders = ", ".join("?" for _ in ACCEPTED_REQUEST_STATUSES)
        params: list[Any] = [
            *ACCEPTED_REQUEST_STATUSES,
            *personnel_ids,
            end_day,
            start_day,
        ]
        type_filter = ""
        if request_type is not None:
            type_filter = " AND request_type = ?"
            params.append(request_type)
        sql = (
            "SELECT id, personnel_id, request_type, duration_type, start_date, "
            "end_date, start_time, end_time, duration_days, duration_minutes, status "
            "FROM personnel_requests "
            f"WHERE status IN ({status_placeholders}) "
            f"AND personnel_id IN ({id_placeholders}) "
            "AND start_date <= ? AND (end_date >= ? OR end_date IS NULL)"
            f"{type_filter}"
        )
        with self.database.connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [
            SummaryRequest(
                id=int(row["id"]),
                personnel_id=int(row["personnel_id"]),
                request_type=str(row.get("request_type") or ""),
                duration_type=(
                    str(row["duration_type"])
                    if row.get("duration_type") is not None
                    else None
                ),
                start_date=_as_date(row["start_date"]),
                end_date=(
                    _as_date(row["end_date"])
                    if row.get("end_date") is not None
                    else None
                ),
                start_time=_as_time(row.get("start_time")),
                end_time=_as_time(row.get("end_time")),
                duration_days=(
                    float(row["duration_days"])
                    if row.get("duration_days") is not None
                    else None
                ),
                duration_minutes=(
                    int(row["duration_minutes"])
                    if row.get("duration_minutes") is not None
                    else None
                ),
                status=str(row.get("status") or ""),
            )
            for row in rows
        ]

    def _accepted_requests_by_person_day(
        self,
        personnel_ids: list[int],
        start_day: date,
        end_day: date,
    ) -> dict[int, dict[date, list[SummaryRequest]]]:
        requests = self._accepted_requests(personnel_ids, start_day, end_day)
        mapped: dict[int, dict[date, list[SummaryRequest]]] = {}
        for request_item in requests:
            request_end = request_item.end_date or request_item.start_date
            effective_start = max(request_item.start_date, start_day)
            effective_end = min(request_end, end_day)
            for day in _daterange(effective_start, effective_end):
                mapped.setdefault(request_item.personnel_id, {}).setdefault(day, []).append(
                    request_item
                )
        return mapped

    @staticmethod
    def _resolve_daily_date_range(
        period: TimePeriod | None,
        from_date_jalali: str | None,
        to_date_jalali: str | None,
    ) -> tuple[datetime, datetime, date, date]:
        now_utc = utc_now()
        today_local = _utc_to_local(now_utc, DEFAULT_LOCAL_TZ_NAME).date()
        try:
            if period == TimePeriod.TODAY:
                start_utc, end_utc = _local_day_bounds_utc(
                    today_local,
                    DEFAULT_LOCAL_TZ_NAME,
                )
                return start_utc, end_utc, today_local, today_local
            if period == TimePeriod.LAST_WEEK:
                start_utc = now_utc - timedelta(days=7)
                return (
                    start_utc,
                    now_utc,
                    _utc_to_local(start_utc).date(),
                    today_local,
                )
            if period == TimePeriod.LAST_MONTH:
                start_utc = now_utc - timedelta(days=30)
                return (
                    start_utc,
                    now_utc,
                    _utc_to_local(start_utc).date(),
                    today_local,
                )
            if period == TimePeriod.CUSTOM or from_date_jalali or to_date_jalali:
                if not from_date_jalali or not to_date_jalali:
                    raise HTTPException(
                        status_code=400,
                        detail="برای فیلتر تاریخ، تاریخ شروع و پایان شمسی الزامی است",
                    )
                start_day = parse_jalali_date(from_date_jalali)
                end_day = parse_jalali_date(to_date_jalali)
                start_utc, end_utc = _local_date_range_bounds_utc(
                    start_day,
                    end_day,
                    DEFAULT_LOCAL_TZ_NAME,
                )
                return start_utc, end_utc, start_day, end_day
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"بازه تاریخ شمسی نامعتبر است: {exc}",
            ) from exc
        start_utc, end_utc = _local_day_bounds_utc(today_local, DEFAULT_LOCAL_TZ_NAME)
        return start_utc, end_utc, today_local, today_local

    def daily_summary(
        self,
        period: TimePeriod | None,
        from_date_jalali: str | None,
        to_date_jalali: str | None,
        include_non_workdays: bool,
        include_absent: bool,
    ) -> list[dict[str, Any]]:
        from_dt, to_dt, start_day, end_day = self._resolve_daily_date_range(
            period,
            from_date_jalali,
            to_date_jalali,
        )
        personnel_list = self._personnel()
        if not personnel_list:
            return []
        logs_by_person = self._logs(personnel_list, from_dt, to_dt)
        holiday_days = self._holiday_dates(start_day, end_day)
        requests_map = self._accepted_requests_by_person_day(
            [person.id for person in personnel_list],
            start_day,
            end_day,
        )

        result: list[dict[str, Any]] = []
        for person in personnel_list:
            shifts_by_day = self._person_shifts(person, start_day, end_day)
            for day in _daterange(start_day, end_day):
                shift = shifts_by_day.get(day)
                if not shift:
                    day_logs = _logs_for_report_day(
                        logs_by_person.get(person.national_code, []),
                        day,
                        None,
                        use_shift_window=False,
                    )
                    local_times = [
                        _to_shift_local(log.detection_time) for log in day_logs
                    ]
                    result.append(
                        {
                            "person": person.national_code,
                            "full_name": f"{person.fname} {person.lname}".strip(),
                            "section_name": person.section_name,
                            "shift_id": None,
                            "shift_name": None,
                            "date_jalali": _jalali_date_string(day),
                            "is_workday": False,
                            "status": "no_shift",
                            "message": "برای این کارمند شیفت کاری تعریف نشده است",
                            "first_detection": (
                                _time_to_hhmm(local_times[0]) if local_times else None
                            ),
                            "last_detection": (
                                None
                                if len(local_times) <= 1
                                else _time_to_hhmm(local_times[-1])
                            ),
                            "middle_detections": (
                                [_time_to_hhmm(item) for item in local_times[1:-1]]
                                if len(local_times) > 2
                                else []
                            ),
                            "first_last_span_time": None,
                            "raw_worked_time": None,
                            "net_worked_time": None,
                            "net_worked_time_with_overtime": None,
                            "delay_minutes": None,
                            "early_leave_minutes": None,
                            "overtime_minutes": None,
                            "holiday_overtime_minutes": None,
                            "illegal_presence_minutes": None,
                            "in_between_absence_minutes": None,
                            "total_absence_time": None,
                            "earned_leave_minutes": None,
                            "sick_leave_minutes": None,
                            "unpaid_leave_minutes": None,
                            "mission_minutes": None,
                        }
                    )
                    continue

                is_shift_day = _shift_expected_on_day(day, shift)
                is_custom_holiday = day in holiday_days
                is_workday = is_shift_day and not is_custom_holiday
                day_logs = _logs_for_report_day(
                    logs_by_person.get(person.national_code, []),
                    day,
                    shift,
                    use_shift_window=True,
                )
                if not include_non_workdays and not is_workday and not day_logs:
                    continue
                if not include_absent and not day_logs:
                    continue

                requests = requests_map.get(person.id, {}).get(day, [])
                stats = _daily_summary_stats(day, day_logs, shift, requests, is_workday)
                request_minutes = stats["request_minutes"]
                daily_request_category = _daily_request_status(requests)
                if is_custom_holiday:
                    status_value = "holiday"
                elif day_logs and len(day_logs) % 2 != 0:
                    status_value = "absence"
                elif day_logs:
                    status_value = "present"
                elif is_workday and daily_request_category is not None:
                    status_value = daily_request_category
                elif sum(request_minutes.values()) > 0:
                    status_value = "approved_request"
                elif is_workday:
                    status_value = "absent"
                else:
                    status_value = "non_workday"

                result.append(
                    {
                        "person": person.national_code,
                        "full_name": f"{person.fname} {person.lname}".strip(),
                        "section_name": person.section_name,
                        "shift_id": shift.id,
                        "shift_name": shift.shift_name,
                        "date_jalali": _jalali_date_string(day),
                        "is_workday": is_workday,
                        "status": status_value,
                        "message": stats["message"],
                        "first_detection": _time_to_hhmm(stats["first_detection"]),
                        "last_detection": (
                            None
                            if len(day_logs) == 1
                            else _time_to_hhmm(stats["last_detection"])
                        ),
                        "middle_detections": [
                            _time_to_hhmm(item) for item in stats["middle_detections"]
                        ],
                        "first_last_span_time": _minutes_to_hhmm(
                            stats["first_last_span_time"]
                        ),
                        "raw_worked_time": _minutes_to_hhmm(stats["raw_worked_time"]),
                        "net_worked_time": _minutes_to_hhmm(stats["net_worked_time"]),
                        "net_worked_time_with_overtime": _minutes_to_hhmm(
                            stats["net_worked_time_with_overtime"]
                        ),
                        "delay_minutes": _minutes_to_hhmm(
                            stats["delay_minutes"] if is_workday else 0
                        ),
                        "early_leave_minutes": _minutes_to_hhmm(
                            stats["early_leave_minutes"] if is_workday else 0
                        ),
                        "overtime_minutes": _minutes_to_hhmm(stats["overtime_minutes"]),
                        "holiday_overtime_minutes": _minutes_to_hhmm(
                            stats["holiday_overtime_minutes"]
                        ),
                        "illegal_presence_minutes": _minutes_to_hhmm(
                            stats["illegal_presence_minutes"]
                        ),
                        "in_between_absence_minutes": _minutes_to_hhmm(
                            stats["in_between_absence_minutes"] if is_workday else 0
                        ),
                        "total_absence_time": _minutes_to_hhmm(
                            stats["total_absence_time"] if is_workday else 0
                        ),
                        "earned_leave_minutes": _minutes_to_hhmm(
                            request_minutes["earned_leave_minutes"]
                        ),
                        "sick_leave_minutes": _minutes_to_hhmm(
                            request_minutes["sick_leave_minutes"]
                        ),
                        "unpaid_leave_minutes": _minutes_to_hhmm(
                            request_minutes["unpaid_leave_minutes"]
                        ),
                        "mission_minutes": _minutes_to_hhmm(
                            request_minutes["mission_minutes"]
                        ),
                    }
                )
        return result

    def monthly_summary(
        self,
        jalali_year: int,
        jalali_month: int,
        personnel_id: str | None,
        section_id: int | None,
        shift_id: int | None,
        include_daily_rows: bool,
        move_days: int,
    ) -> list[dict[str, Any]]:
        g_start, g_end = _jalali_moving_month_range(
            jalali_year,
            jalali_month,
            move_days,
        )
        month_days = (g_end - g_start).days + 1
        query_start = g_start - timedelta(days=1)
        query_end = g_end + timedelta(days=1)
        from_dt, to_dt = _local_date_range_bounds_utc(
            query_start,
            query_end,
            DEFAULT_LOCAL_TZ_NAME,
        )
        personnel_list = self._personnel(personnel_id, section_id, shift_id)
        if not personnel_list:
            return []
        logs_by_person = self._logs(personnel_list, from_dt, to_dt)
        holiday_days = self._holiday_dates(g_start, g_end)
        requests_map = self._accepted_requests_by_person_day(
            [person.id for person in personnel_list],
            g_start,
            g_end,
        )

        result: list[dict[str, Any]] = []
        for person in personnel_list:
            shifts_by_day = self._person_shifts(person, g_start, g_end)
            # The filter selects personnel who used this shift during the report
            # period; their totals still cover the complete requested month.
            if shift_id is not None and not any(s.id == shift_id for s in shifts_by_day.values()):
                continue
            effective = {shift.id: shift for shift in shifts_by_day.values()}
            scalar_shift = next(iter(effective.values())) if len(effective) == 1 else None
            missing_shift_message = (
                "برای بخشی از بازه گزارش، شیفت کاری تعریف نشده است"
                if len(shifts_by_day) < month_days else None
            )
            person_logs = logs_by_person.get(person.national_code, [])

            off_days = 0
            no_shift_days = 0
            custom_holidays = 0
            total_holidays = 0
            month_working_days = 0
            present_days = 0
            sick_leave_days = 0
            mission_days = 0
            earned_leave_days = 0
            unpaid_leave_days = 0
            early_minutes = 0
            delay_minutes = 0
            hourly_leave_minutes = 0.0
            work_time_minutes = 0
            overtime_minutes = 0
            holiday_overtime_minutes = 0
            daily_rows: list[dict[str, Any]] = []

            for day in _daterange(g_start, g_end):
                shift = shifts_by_day.get(day)
                is_custom_holiday = day in holiday_days
                if shift is None:
                    no_shift_days += 1
                    if is_custom_holiday:
                        custom_holidays += 1
                        total_holidays += 1
                    day_logs = _logs_for_report_day(
                        person_logs, day, None, use_shift_window=False
                    )
                    if include_daily_rows:
                        local_times = [
                            _to_shift_local(log.detection_time) for log in day_logs
                        ]
                        daily_rows.append({
                            "date": day.isoformat(), "date_jalali": _jalali_date_string(day),
                            "shift_id": None, "shift_name": None, "is_shift_day": False,
                            "is_custom_holiday": is_custom_holiday, "is_workday": False,
                            "status": "no_shift", "detection_count": len(day_logs),
                            "first_detection": (
                                local_times[0].time() if local_times else None
                            ),
                            "last_detection": (
                                local_times[-1].time() if len(local_times) > 1 else None
                            ),
                            "hourly_leave_minutes": 0, "work_time": "00:00",
                            "overtime": "00:00", "holiday_overtime": "00:00",
                        })
                    continue
                is_shift_day = _shift_expected_on_day(day, shift)
                is_workday = is_shift_day and not is_custom_holiday
                if not is_shift_day:
                    off_days += 1
                if is_custom_holiday:
                    custom_holidays += 1
                if not is_shift_day or is_custom_holiday:
                    total_holidays += 1
                if is_workday:
                    month_working_days += 1

                day_logs = _logs_for_report_day(
                    person_logs, day, shift, use_shift_window=True
                )
                shift_window = _shift_window(day, shift)
                assert shift_window is not None
                shift_start, shift_end = shift_window
                requests = requests_map.get(person.id, {}).get(day, [])
                request_minutes = _request_minutes_for_day(
                    requests,
                    day,
                    shift_start,
                    shift_end,
                )
                if is_workday:
                    hourly_leave_minutes += request_minutes["hourly_leave_minutes"]

                presence = _monthly_presence_minutes(day, day_logs, shift, is_workday)
                work_time_minutes += presence["regular"]
                overtime_minutes += presence["overtime"]
                holiday_overtime_minutes += presence["holiday_overtime"]

                day_category: str | None = None
                if is_workday:
                    request_category = _full_day_request_category(requests)
                    if request_category == REQUEST_MISSION:
                        mission_days += 1
                        day_category = "mission"
                    elif request_category == REQUEST_SICK_LEAVE:
                        sick_leave_days += 1
                        day_category = "sick_leave"
                    elif request_category == REQUEST_EARNED_LEAVE:
                        earned_leave_days += 1
                        day_category = "earned_leave"
                    elif request_category == REQUEST_UNPAID_LEAVE:
                        unpaid_leave_days += 1
                        day_category = "unpaid_leave"
                    elif day_logs:
                        present_days += 1
                        day_category = "present"
                        stats = _compute_shift_day_stats(day, day_logs, shift)
                        early_minutes += stats["early_leave_minutes"]
                        delay_minutes += stats["delay_minutes"]
                    else:
                        day_category = "absent"
                elif day_logs:
                    day_category = "holiday_overtime"
                elif is_custom_holiday:
                    day_category = "custom_holiday"
                else:
                    day_category = "off_day"

                if include_daily_rows:
                    first_detection = (
                        _to_shift_local(day_logs[0].detection_time, shift)
                        if day_logs
                        else None
                    )
                    last_detection = (
                        _to_shift_local(day_logs[-1].detection_time, shift)
                        if day_logs
                        else None
                    )
                    daily_rows.append(
                        {
                            "date": day.isoformat(),
                            "date_jalali": _jalali_date_string(day),
                            "shift_id": shift.id,
                            "shift_name": shift.shift_name,
                            "is_shift_day": is_shift_day,
                            "is_custom_holiday": is_custom_holiday,
                            "is_workday": is_workday,
                            "status": day_category,
                            "detection_count": len(day_logs),
                            "first_detection": (
                                first_detection.time() if first_detection else None
                            ),
                            "last_detection": (
                                last_detection.time() if last_detection else None
                            ),
                            "hourly_leave_minutes": int(
                                round(request_minutes["hourly_leave_minutes"])
                            ),
                            "work_time": _minutes_to_hhmm(presence["regular"]),
                            "overtime": _minutes_to_hhmm(presence["overtime"]),
                            "holiday_overtime": _minutes_to_hhmm(
                                presence["holiday_overtime"]
                            ),
                        }
                    )

            final_working_days = (
                present_days + mission_days + earned_leave_days + sick_leave_days
            )
            absent_days = max(
                0,
                month_working_days - final_working_days - unpaid_leave_days,
            )
            total_hourly_absent_minutes = early_minutes + delay_minutes
            result_item: dict[str, Any] = {
                "personnel_id": person.id,
                "person": person.national_code,
                "full_name": f"{person.fname} {person.lname}".strip(),
                "section_id": person.department_id,
                "section_name": person.section_name,
                "shift_id": scalar_shift.id if scalar_shift else None,
                "shift_name": scalar_shift.shift_name if scalar_shift else None,
                "timezone_name": _shift_timezone_name(scalar_shift) if scalar_shift else None,
                "shift_assignments": _shift_assignment_segments(
                    shifts_by_day, g_start, g_end
                ),
                "jalali_year": jalali_year,
                "jalali_month": jalali_month,
                "move_days": move_days,
                "month_start": _jalali_date_string(g_start),
                "month_end": _jalali_date_string(g_end),
                "month_days": month_days,
                "off_days": off_days,
                "no_shift_days": no_shift_days,
                "custom_holidays": custom_holidays,
                "total_holidays": total_holidays,
                "month_working_days": month_working_days,
                "present_days": present_days,
                "sick_leave_days": sick_leave_days,
                "mission_days": mission_days,
                "earned_leave_days": earned_leave_days,
                "unpaid_leave_days": unpaid_leave_days,
                "final_working_days": final_working_days,
                "absent_days": absent_days,
                "early_minutes": early_minutes,
                "delay_minutes": delay_minutes,
                "total_hourly_absent_minutes": total_hourly_absent_minutes,
                "hourly_leave_minutes": int(round(hourly_leave_minutes)),
                "work_time_hours": _minutes_to_hhmm(work_time_minutes),
                "overtime": _minutes_to_hhmm(overtime_minutes),
                "holiday_overtime": _minutes_to_hhmm(holiday_overtime_minutes),
                "message": missing_shift_message,
            }
            if include_daily_rows:
                result_item["days"] = daily_rows
            result.append(result_item)
        return result

    def yearly_leave_summary(self, jalali_year: int) -> list[dict[str, Any]]:
        personnel_list = self._personnel()
        if not personnel_list:
            return []
        month_ranges: list[dict[str, Any]] = []
        for month in range(1, 13):
            month_start, month_end = _jalali_moving_month_range(
                jalali_year,
                month,
                10,
            )
            month_ranges.append(
                {"month": month, "g_start": month_start, "g_end": month_end}
            )
        report_start = month_ranges[0]["g_start"]
        report_end = month_ranges[-1]["g_end"]
        query_start = report_start - timedelta(days=1)
        query_end = report_end + timedelta(days=1)
        from_dt, to_dt = _local_date_range_bounds_utc(
            query_start,
            query_end,
            DEFAULT_LOCAL_TZ_NAME,
        )
        logs_by_person = self._logs(personnel_list, from_dt, to_dt)
        holiday_days = self._holiday_dates(report_start, report_end)
        requests_map = self._accepted_requests_by_person_day(
            [person.id for person in personnel_list],
            report_start,
            report_end,
        )
        earned_requests = self._accepted_requests(
            [person.id for person in personnel_list],
            report_start,
            report_end,
            request_type=REQUEST_EARNED_LEAVE,
        )
        earned_requests_by_person: dict[int, list[SummaryRequest]] = {}
        for request_item in earned_requests:
            earned_requests_by_person.setdefault(request_item.personnel_id, []).append(
                request_item
            )

        result: list[dict[str, Any]] = []
        for person in personnel_list:
            yearly_shifts = self._person_shifts(person, report_start, report_end)
            distinct_shifts = {shift.id: shift for shift in yearly_shifts.values()}
            person_logs = logs_by_person.get(person.national_code, [])
            person_requests_by_day = requests_map.get(person.id, {})
            person_earned_requests = earned_requests_by_person.get(person.id, [])
            months: list[dict[str, Any]] = []
            cumulative_earned_days = 0.0
            cumulative_used_days = 0.0
            for month_range in month_ranges:
                month = month_range["month"]
                month_start = month_range["g_start"]
                month_end = month_range["g_end"]
                attendance = _yearly_leave_month_attendance(
                    person,
                    person_logs,
                    month_start,
                    month_end,
                    holiday_days,
                    person_requests_by_day,
                    {day: shift for day, shift in yearly_shifts.items() if month_start <= day <= month_end},
                )
                earned_leave_days = _calculate_monthly_earned_leave_days(
                    attendance["absent_days"],
                    attendance["unpaid_leave_days"],
                    attendance["sick_leave_days"],
                )
                used_leave_days = round(sum(
                    1.0
                    for request_item in person_earned_requests
                    if request_item.duration_type == "daily"
                    for day in _daterange(max(request_item.start_date, month_start), min(request_item.end_date or request_item.start_date, month_end))
                    if day in yearly_shifts
                    and _shift_expected_on_day(day, yearly_shifts[day])
                    and day not in holiday_days
                ), 4)
                cumulative_earned_days += earned_leave_days
                cumulative_used_days += used_leave_days
                remaining_leave_days = _floor_to_two_decimal_places(
                    cumulative_earned_days - cumulative_used_days
                )
                months.append(
                    {
                        "month": month,
                        "period_start": _jalali_date_string(month_start),
                        "period_end": _jalali_date_string(month_end),
                        "earned_leave_days": earned_leave_days,
                        "used_leave_days": used_leave_days,
                        "remaining_leave_days": remaining_leave_days,
                        "expected_working_days": attendance["month_working_days"],
                        "eligible_days": attendance["final_working_days"],
                        "absence_days": attendance["absent_days"],
                        "unpaid_leave_days": attendance["unpaid_leave_days"],
                    }
                )
            result.append(
                {
                    "personnel_id": person.id,
                    "full_name": f"{person.fname} {person.lname}".strip(),
                    "national_code": person.national_code,
                    "section_name": person.section_name,
                    "shift_name": (
                        next(iter(distinct_shifts.values())).shift_name
                        if len(distinct_shifts) == 1 else None
                    ),
                    "jalali_year": jalali_year,
                    "months": months,
                }
            )
        return result
