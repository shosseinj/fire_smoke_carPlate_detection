from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from app.core.jalali_utils import (
    jalali_datetime_string,
    local_date_range_bounds_utc,
    local_day_utc_range,
)


def test_local_day_bounds_use_half_open_utc_range_for_tehran_day() -> None:
    start_utc, end_utc = local_day_utc_range(date(2024, 3, 20))
    assert start_utc == datetime(2024, 3, 19, 20, 30, tzinfo=timezone.utc)
    assert end_utc == datetime(2024, 3, 20, 20, 30, tzinfo=timezone.utc)
    assert (end_utc - start_utc).total_seconds() == 24 * 60 * 60


def test_local_date_range_bounds_rejects_reverse_ranges() -> None:
    with pytest.raises(ValueError, match="تاریخ پایان"):
        local_date_range_bounds_utc(date(2024, 3, 21), date(2024, 3, 20))


def test_jalali_datetime_string_outputs_local_jalali_date_and_time() -> None:
    utc_value = datetime(2024, 3, 20, 5, 0, tzinfo=timezone.utc)
    result = jalali_datetime_string(utc_value)
    assert "1403" in result
    assert "08:30:00" in result

pytestmark = pytest.mark.unit
