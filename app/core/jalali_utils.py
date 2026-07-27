"""Jalali (Shamsi) date conversion and timezone utilities."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Literal

import jdatetime

# ── Global business timezone ──────────────────────────────────────────
# The project operates in Asia/Tehran for business date interpretation.
BUSINESS_TZ_PYTZ: str = "Asia/Tehran"
# Fallback UTC offset for Asia/Tehran (+3:30)
TEHRAN_UTC_OFFSET = timedelta(hours=3, minutes=30)


def tehran_now() -> datetime:
    """Return current time in Asia/Tehran timezone-aware datetime."""
    utc_dt = datetime.now(timezone.utc)
    # Approximate Tehran offset (DST-aware through fixed +3:30 for simplicity)
    # In production, use pytz or zoneinfo for DST accuracy.
    return utc_dt.replace(tzinfo=timezone.utc).astimezone(_get_tehran_tz())


def _get_tehran_tz() -> timezone:
    """Return a timezone object for Asia/Tehran (+3:30, no DST)."""
    return timezone(TEHRAN_UTC_OFFSET)


def utc_to_tehran(utc_dt: datetime) -> datetime:
    """Convert a UTC-aware datetime to Asia/Tehran."""
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    return utc_dt.astimezone(_get_tehran_tz())


def tehran_to_utc(tehran_dt: datetime) -> datetime:
    """Convert an Asia/Tehran-aware datetime to UTC."""
    if tehran_dt.tzinfo is None:
        tehran_dt = tehran_dt.replace(tzinfo=_get_tehran_tz())
    return tehran_dt.astimezone(timezone.utc)


# ── Jalali / Gregorian conversion ─────────────────────────────────────

# Arabic and Persian digit normalization
_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def normalize_digits(s: str) -> str:
    """Convert Persian/Arabic digits to ASCII digits."""
    return s.translate(_PERSIAN_DIGITS)


def parse_jalali_date(s: str) -> date:
    """Parse a Jalali date string (YYYY-MM-DD or YYYY/M/D) to Gregorian date.

    Handles Persian/Arabic digits automatically.
    """
    s = normalize_digits(s.strip())
    # Support YYYY-MM-DD and YYYY/M/D formats
    match = re.match(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", s)
    if not match:
        raise ValueError(f"Invalid Jalali date format: {s!r}")
    j_year = int(match.group(1))
    j_month = int(match.group(2))
    j_day = int(match.group(3))
    # Validate Jalali date range
    if not (1 <= j_month <= 12):
        raise ValueError(f"Jalali month out of range: {j_month}")
    if not (1 <= j_day <= 31):
        raise ValueError(f"Jalali day out of range: {j_day}")
    try:
        j_date = jdatetime.date(j_year, j_month, j_day)
        return j_date.togregorian()
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid Jalali date {s!r}: {exc}")


def gregorian_to_jalali(g_date: date) -> tuple[int, int, int]:
    """Convert a Gregorian date to (jalali_year, jalali_month, jalali_day)."""
    j_date = jdatetime.date.fromgregorian(date=g_date)
    return j_date.year, j_date.month, j_date.day


def gregorian_to_jalali_str(g_date: date, sep: str = "-") -> str:
    """Convert a Gregorian date to a Jalali date string (YYYY-MM-DD)."""
    j_date = jdatetime.date.fromgregorian(date=g_date)
    return f"{j_date.year:04d}{sep}{j_date.month:02d}{sep}{j_date.day:02d}"


# ── Local day boundaries ──────────────────────────────────────────────


def local_date_range_bounds_utc(
    start_date: date, end_date: date, tz: timezone | None = None
) -> tuple[datetime, datetime]:
    if tz is None:
        tz = _get_tehran_tz()
    if end_date < start_date:
        raise ValueError("end_date must not be before start_date")
    utc_start, _ = local_day_utc_range(start_date, tz)
    _, utc_end = local_day_utc_range(end_date, tz)
    return utc_start, utc_end


def utc_iso_to_jalali_datetime(utc_iso: str | None) -> str | None:
    """Convert a UTC ISO string (e.g. '2026-07-27T12:00:00Z') to a Jalali datetime string.

    Returns ``None`` when the input is ``None`` or unparseable.
    """
    if not utc_iso:
        return None
    try:
        dt = datetime.fromisoformat(utc_iso.replace("Z", "+00:00"))
        return jalali_datetime_string(dt)
    except (ValueError, TypeError):
        return None


def jalali_datetime_string(utc_dt: datetime, tz_name: str = "Asia/Tehran") -> str:
    from zoneinfo import ZoneInfo
    local_dt = utc_dt.astimezone(ZoneInfo(tz_name))
    j_date = jdatetime.date.fromgregorian(date=local_dt.date())
    return f"{j_date.year:04d}-{j_date.month:02d}-{j_date.day:02d} {local_dt.hour:02d}:{local_dt.minute:02d}:{local_dt.second:02d}"


def local_day_utc_range(local_date: date, tz: timezone | None = None) -> tuple[datetime, datetime]:
    """Return (utc_start, utc_end) half-open range for a local business date.

    The local date is interpreted in Asia/Tehran.
    Returns UTC-aware datetimes: [utc_start, utc_end).
    """
    if tz is None:
        tz = _get_tehran_tz()
    local_start = datetime(local_date.year, local_date.month, local_date.day, 0, 0, 0, tzinfo=tz)
    local_end = local_start + timedelta(days=1)
    utc_start = local_start.astimezone(timezone.utc)
    utc_end = local_end.astimezone(timezone.utc)
    return utc_start, utc_end


def local_month_utc_range(
    year: int, month: int, tz: timezone | None = None
) -> tuple[datetime, datetime]:
    """Return (utc_start, utc_end) for a local Gregorian month range."""
    if tz is None:
        tz = _get_tehran_tz()
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    start = datetime(year, month, 1, 0, 0, 0, tzinfo=tz)
    end = datetime(year, month, last_day, 23, 59, 59, 999999, tzinfo=tz) + timedelta.resolution
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def jalali_month_utc_range(
    j_year: int, j_month: int, tz: timezone | None = None
) -> tuple[datetime, datetime]:
    """Return (utc_start, utc_end) for a Jalali year/month."""
    # Approximate: first day of Jalali month to Gregorian
    j_first = jdatetime.date(j_year, j_month, 1)
    g_first = j_first.togregorian()
    if j_month < 12:
        j_last = jdatetime.date(j_year, j_month + 1, 1) - timedelta(days=1)
    else:
        j_last = jdatetime.date(j_year + 1, 1, 1) - timedelta(days=1)
    g_last = j_last.togregorian()
    return local_day_utc_range(g_first, tz)[0], local_day_utc_range(g_last, tz)[1] + timedelta(days=1)
