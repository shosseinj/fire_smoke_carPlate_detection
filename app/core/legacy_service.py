"""Shared service for Step 5 legacy contract: status/type mapping, weekday mapping,
timezone validation, Jalali conversion, and request-duration calculation."""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

import jdatetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.core.jalali_utils import (
    gregorian_to_jalali_str,
    normalize_digits,
    parse_jalali_date,
)
from app.core.personnel_store import PersonnelRecord
from app.core.shift_store import (
    WEEKDAY_COLS,
    WEEKDAY_NAMES,
    WorkShiftRecord,
    _get_weekday_flag,
    _is_overnight,
    _weekday_from_local,
)
from app.core.holiday_store import HolidayStore
from app.core.request_store import RequestStore

# ── Status mapping ─────────────────────────────────────────────────────
# Internal (current)  <->  Legacy (public)
STATUS_MAP_INTERNAL_TO_LEGACY = {
    "pending": "waiting",
    "approved": "accepted",
    "rejected": "rejected",
    "cancelled": "cancelled",
}
STATUS_MAP_LEGACY_TO_INTERNAL = {v: k for k, v in STATUS_MAP_INTERNAL_TO_LEGACY.items()}

# ── Request type mapping ───────────────────────────────────────────────
LEGACY_REQUEST_TYPES = frozenset({
    "earned_leave",
    "sick_leave",
    "unpaid_leave",
    "mission",
    "overtime",
})

LEGACY_REQUEST_TYPE_MAP = {
    "earned_leave": "earned_leave",
    "sick_leave": "sick_leave",
    "unpaid_leave": "unpaid_leave",
    "mission": "mission",
    "overtime": "overtime",
}

LEGACY_DURATION_TYPES = frozenset({"daily", "hourly"})

# ── Status conversions ─────────────────────────────────────────────────


def legacy_status(status: str) -> str:
    return STATUS_MAP_INTERNAL_TO_LEGACY.get(status, status)


def internal_status(status: str) -> str:
    return STATUS_MAP_LEGACY_TO_INTERNAL.get(status, status)


# ── Weekday mapping ────────────────────────────────────────────────────

WEEKDAY_LEGACY_NAMES = [
    "saturday", "sunday", "monday", "tuesday",
    "wednesday", "thursday", "friday",
]

WEEKDAY_INTERNAL_TO_LEGACY = dict(zip(WEEKDAY_COLS, WEEKDAY_LEGACY_NAMES))
WEEKDAY_LEGACY_TO_INTERNAL = dict(zip(WEEKDAY_LEGACY_NAMES, WEEKDAY_COLS))


def shift_to_legacy_weekdays(record: WorkShiftRecord) -> dict[str, bool]:
    return {WEEKDAY_LEGACY_NAMES[i]: _get_weekday_flag(record, i) for i in range(7)}


def legacy_weekdays_to_internal(weekdays: dict[str, bool]) -> dict[str, bool]:
    return {WEEKDAY_LEGACY_TO_INTERNAL[k]: v for k, v in weekdays.items() if k in WEEKDAY_LEGACY_TO_INTERNAL}


# ── Timezone validation ────────────────────────────────────────────────


def validate_timezone(tz_name: str) -> str:
    """Validate a timezone name. Returns the normalized name."""
    try:
        ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, TypeError, ValueError):
        raise ValueError(f"Invalid timezone: {tz_name!r}")
    return tz_name


def validate_clock_time(t: str) -> str:
    """Validate HH:MM format. Reject 24:00 on legacy routes."""
    if t == "24:00":
        raise ValueError("فرمت زمان نامعتبر. از HH:MM استفاده کنید")
    parts = t.split(":")
    if len(parts) != 2:
        raise ValueError("فرمت زمان نامعتبر. از HH:MM استفاده کنید")
    try:
        h, m = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError("فرمت زمان نامعتبر. از HH:MM استفاده کنید")
    if not (0 <= h <= 23) or not (0 <= m <= 59):
        raise ValueError("فرمت زمان نامعتبر. از HH:MM استفاده کنید")
    return t


# ── Jalali helpers ────────────────────────────────────────────────────


def validate_jalali_date(s: str) -> date:
    """Parse and return Gregorian date from Jalali or Gregorian input."""
    s = normalize_digits(s.strip())
    # Try Jalali first (years 1200-1500)
    parts = s.split("-")
    if len(parts) == 3:
        try:
            y = int(parts[0])
            m = int(parts[1])
            d_val = int(parts[2])
        except (ValueError, IndexError):
            raise ValueError(f"Invalid date: {s!r}")
        if 1200 <= y <= 1500:
            try:
                g = parse_jalali_date(s)
                return g
            except ValueError:
                pass
        # Try Gregorian
        try:
            return date(y, m, d_val)
        except (ValueError, IndexError):
            raise ValueError(f"Invalid date: {s!r}")
    try:
        return parse_jalali_date(s)
    except ValueError:
        raise ValueError(f"Invalid date: {s!r}. Use YYYY-MM-DD.")


def format_jalali(g: date) -> str:
    """Convert Gregorian date to Jalali string YYYY-MM-DD."""
    return gregorian_to_jalali_str(g)


def format_jalali_datetime_to_minute(utc_iso: str | None) -> str | None:
    """Convert UTC ISO timestamp to Jalali local datetime to minute precision."""
    if not utc_iso:
        return None
    try:
        dt = datetime.fromisoformat(utc_iso.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    tehran = timezone(timedelta(hours=3, minutes=30))
    local = dt.astimezone(tehran)
    j_date = jdatetime.date.fromgregorian(date=local.date())
    return f"{j_date.year:04d}-{j_date.month:02d}-{j_date.day:02d} {local.hour:02d}:{local.minute:02d}"


# ── Legacy shift response serializer ──────────────────────────────────


def legacy_shift_response(record: WorkShiftRecord, personnel_count: int = 0) -> dict[str, Any]:
    weekdays = shift_to_legacy_weekdays(record)
    return {
        "id": record.id,
        "shift_name": record.shift_name,
        "shift_type": record.shift_type,
        "start_time": record.start_time,
        "end_time": record.end_time,
        "timezone_name": getattr(record, "timezone_name", "Asia/Tehran"),
        "max_minutes_delay": record.max_minutes_delay,
        "max_minutes_early": record.max_minutes_early,
        "max_overtime_hours": record.max_overtime_hours,
        "monday": weekdays["monday"],
        "tuesday": weekdays["tuesday"],
        "wednesday": weekdays["wednesday"],
        "thursday": weekdays["thursday"],
        "friday": weekdays["friday"],
        "saturday": weekdays["saturday"],
        "sunday": weekdays["sunday"],
        "personnel_count": personnel_count,
    }


# ── Legacy holiday response serializer ────────────────────────────────


def legacy_holiday_response(
    record: Any,
    created_by_username: str | None = None,
    updated_by_username: str | None = None,
) -> dict[str, Any]:
    g = date.fromisoformat(record.date_value) if isinstance(record.date_value, str) else record.date_value
    return {
        "id": record.id,
        "name": record.name,
        "date": format_jalali(g),
        "description": record.description,
        "holiday_type": record.holiday_type,
        "every_year": record.every_year,
        "is_active": record.is_active,
        "created_at": format_jalali_datetime_to_minute(getattr(record, "created_at_utc", None)),
        "updated_at": format_jalali_datetime_to_minute(getattr(record, "updated_at_utc", None)),
        "created_by": getattr(record, "created_by", None),
        "updated_by": getattr(record, "updated_by", None),
        "created_by_username": created_by_username,
        "updated_by_username": updated_by_username,
    }


# ── Request duration calculation ──────────────────────────────────────


def calculate_request_duration(
    personnel: PersonnelRecord,
    shift: WorkShiftRecord | None,
    holiday_store: HolidayStore,
    start_date: date,
    end_date: date,
    duration_type: str,
    start_time: time | None = None,
    end_time: time | None = None,
) -> dict[str, Any]:
    """Calculate the duration of a personnel request.

    Returns dict with working_dates, excluded_non_working_dates,
    excluded_holiday_dates, duration_days, duration_minutes, duration_hours.
    """
    working_dates: list[str] = []
    excluded_non_working: list[str] = []
    excluded_holidays: list[str] = []
    total_minutes = 0

    current = start_date
    while current <= end_date:
        j_str = format_jalali(current)

        # Determine weekday
        weekday_idx = _weekday_from_local(datetime.combine(current, time.min).replace(tzinfo=timezone.utc))
        is_working = shift is not None and _get_weekday_flag(shift, weekday_idx)
        is_holiday = holiday_store.is_holiday(current)

        if not is_working:
            excluded_non_working.append(j_str)
        elif is_holiday:
            excluded_holidays.append(j_str)
        else:
            working_dates.append(j_str)

        if duration_type == "hourly" and is_working and not is_holiday and start_time is not None and end_time is not None:
            # Calculate intersection with shift
            sh, sm = (int(x) for x in shift.start_time.split(":")) if shift else (0, 0)
            eh, em = (int(x) for x in shift.end_time.split(":")) if shift else (0, 0)
            shift_start_min = sh * 60 + sm
            shift_end_min = eh * 60 + em
            if _is_overnight(shift.start_time, shift.end_time) if shift else False:
                shift_end_min += 24 * 60

            req_start_min = start_time.hour * 60 + start_time.minute
            req_end_min = end_time.hour * 60 + end_time.minute
            if req_end_min <= req_start_min:
                req_end_min += 24 * 60

            overlap_start = max(shift_start_min, req_start_min)
            overlap_end = min(shift_end_min, req_end_min)
            if overlap_end > overlap_start:
                total_minutes += overlap_end - overlap_start

        current += timedelta(days=1)

    duration_days = float(len(working_dates))
    duration_minutes = total_minutes if duration_type == "hourly" else None
    duration_hours = round(total_minutes / 60, 4) if duration_type == "hourly" else None

    return {
        "working_dates": working_dates,
        "excluded_non_working_dates": excluded_non_working,
        "excluded_holiday_dates": excluded_holidays,
        "duration_days": duration_days,
        "duration_minutes": duration_minutes,
        "duration_hours": duration_hours,
    }


# ── Legacy personnel request response serializer ──────────────────────


def legacy_request_response(
    record: Any,
    full_name: str | None = None,
    duration_minutes: int | None = None,
) -> dict[str, Any]:
    """Serialize a request record to the legacy GET representation."""
    s = date.fromisoformat(record.start_date) if isinstance(record.start_date, str) else record.start_date
    e = date.fromisoformat(record.end_date) if isinstance(record.end_date, str) else record.end_date

    resp: dict[str, Any] = {
        "id": record.id,
        "personnel_id": record.personnel_id,
        "full_name": full_name,
        "request_type": record.request_type,
        "duration_type": getattr(record, "duration_type", None),
        "start_date": format_jalali(s),
        "end_date": format_jalali(e),
        "start_time": _format_time(getattr(record, "start_time", None)),
        "end_time": _format_time(getattr(record, "end_time", None)),
        "description": getattr(record, "reason", getattr(record, "description", None)),
        "duration_days": getattr(record, "duration_days", None),
        "duration_minutes": getattr(record, "duration_minutes", None),
        "duration_time": _format_duration_time(getattr(record, "duration_minutes", duration_minutes)),
        "status": legacy_status(getattr(record, "status", "")),
        "admin_notes": getattr(record, "admin_notes", None),
        "rejection_reason": getattr(record, "rejection_reason", None),
        "created_at": format_jalali_datetime_to_minute(getattr(record, "created_at_utc", None)),
        "updated_at": format_jalali_datetime_to_minute(getattr(record, "updated_at_utc", None)),
        "reviewed_by": getattr(record, "reviewed_by", getattr(record, "approved_by", None)),
        "reviewed_at": format_jalali_datetime_to_minute(
            getattr(record, "reviewed_at", None)
        ),
    }
    return resp


def _format_time(t: Any) -> str | None:
    if t is None:
        return None
    if isinstance(t, time):
        return t.strftime("%H:%M")
    if isinstance(t, str) and t:
        return t[:5]
    return None


def _format_duration_time(minutes: int | None) -> str | None:
    if minutes is None:
        return None
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"
