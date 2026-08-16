from __future__ import annotations

from datetime import datetime, timezone

from app.core.jalali_utils import parse_jalali_date, utc_iso_to_jalali_datetime
from app.time_utils import ensure_aware_utc, utc_now


def test_utc_helpers_return_timezone_aware_utc_values() -> None:
    now = utc_now()
    assert now.tzinfo is timezone.utc

    naive = datetime(2026, 7, 8, 12, 30)
    converted = ensure_aware_utc(naive)
    assert converted == datetime(2026, 7, 8, 12, 30, tzinfo=timezone.utc)


def test_jalali_date_parsing_converts_to_gregorian() -> None:
    assert parse_jalali_date("1403/01/01") == datetime(2024, 3, 20, tzinfo=timezone.utc).date()


def test_utc_iso_to_jalali_datetime_converts_string() -> None:
    """2024-03-20T05:00:00Z (Tehran 08:30) → 1403-01-01 08:30:00."""
    result = utc_iso_to_jalali_datetime("2024-03-20T05:00:00Z")
    assert result == "1403-01-01 08:30:00"


def test_utc_iso_to_jalali_datetime_none_returns_none() -> None:
    assert utc_iso_to_jalali_datetime(None) is None


def test_utc_iso_to_jalali_datetime_empty_returns_none() -> None:
    assert utc_iso_to_jalali_datetime("") is None


def test_utc_iso_to_jalali_datetime_invalid_returns_none() -> None:
    assert utc_iso_to_jalali_datetime("not-a-date") is None

import pytest

pytestmark = pytest.mark.unit
