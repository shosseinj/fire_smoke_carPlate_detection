from __future__ import annotations

from datetime import datetime, timezone

from app.core.jalali_utils import parse_jalali_date
from app.time_utils import ensure_aware_utc, utc_now


def test_utc_helpers_return_timezone_aware_utc_values() -> None:
    now = utc_now()
    assert now.tzinfo is timezone.utc

    naive = datetime(2026, 7, 8, 12, 30)
    converted = ensure_aware_utc(naive)
    assert converted == datetime(2026, 7, 8, 12, 30, tzinfo=timezone.utc)


def test_jalali_date_parsing_converts_to_gregorian() -> None:
    assert parse_jalali_date("1403/01/01") == datetime(2024, 3, 20, tzinfo=timezone.utc).date()