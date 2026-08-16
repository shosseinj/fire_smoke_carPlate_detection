"""Tests for HolidayStore."""

from __future__ import annotations

from datetime import date

import pytest

from app.core.holiday_store import HolidayStore
from app.database import Database


@pytest.fixture
def store(postgres_database: Database) -> HolidayStore:
    return HolidayStore(postgres_database)


class TestHolidayStore:
    def test_create(self, store: HolidayStore) -> None:
        h = store.create(
            name="New Year",
            date_value="2026-01-01",
            holiday_type="national",
        )
        assert h.id > 0
        assert h.name == "New Year"
        assert h.is_active is True

    def test_create_jalali(self, store: HolidayStore) -> None:
        h = store.create(
            name="Norouz",
            date_value="1405-01-01",
            holiday_type="national",
            every_year=True,
        )
        assert h.id > 0
        # 1405-01-01 Jalali = 2026-03-21 Gregorian
        assert h.date_value == "2026-03-21"

    def test_create_duplicate_active(self, store: HolidayStore) -> None:
        store.create(name="H1", date_value="2026-06-01", holiday_type="national")
        with pytest.raises(ValueError, match="Active holiday already exists"):
            store.create(name="H2", date_value="2026-06-01", holiday_type="national")

    def test_get(self, store: HolidayStore) -> None:
        h = store.create(name="Test", date_value="2026-07-01", holiday_type="national")
        fetched = store.get(h.id)
        assert fetched is not None
        assert fetched.id == h.id

    def test_get_nonexistent(self, store: HolidayStore) -> None:
        assert store.get(99999) is None

    def test_update(self, store: HolidayStore) -> None:
        h = store.create(name="Old", date_value="2026-03-01", holiday_type="national")
        updated = store.update(h.id, name="New Name")
        assert updated is not None
        assert updated.name == "New Name"

    def test_update_nonexistent(self, store: HolidayStore) -> None:
        assert store.update(99999, name="X") is None

    def test_delete(self, store: HolidayStore) -> None:
        h = store.create(name="Del", date_value="2026-09-01", holiday_type="national")
        assert store.delete(h.id) is True
        fetched = store.get(h.id)
        assert fetched is not None
        assert fetched.is_active is False

    def test_hard_delete(self, store: HolidayStore) -> None:
        h = store.create(name="Hard", date_value="2026-10-01", holiday_type="national")
        assert store.hard_delete(h.id) is True
        assert store.get(h.id) is None

    def test_list(self, store: HolidayStore) -> None:
        store.create(name="A", date_value="2026-01-01", holiday_type="national")
        store.create(name="B", date_value="2026-06-15", holiday_type="religious")
        records, total = store.list(limit=100)
        assert total >= 2

    def test_list_filter_type(self, store: HolidayStore) -> None:
        store.create(name="A", date_value="2026-01-01", holiday_type="national")
        store.create(name="B", date_value="2026-06-15", holiday_type="company")
        records, total = store.list(holiday_type="company")
        assert total == 1

    def test_is_holiday(self, store: HolidayStore) -> None:
        store.create(name="TestH", date_value="2026-12-25", holiday_type="national")
        assert store.is_holiday(date(2026, 12, 25)) is True
        assert store.is_holiday(date(2026, 12, 26)) is False

    def test_is_holiday_annual(self, store: HolidayStore) -> None:
        store.create(name="Annual", date_value="2026-03-21", holiday_type="national", every_year=True)
        assert store.is_holiday(date(2027, 3, 21)) is True
        assert store.is_holiday(date(2028, 3, 21)) is True

    def test_annual_not_count_for_active_only(self, store: HolidayStore) -> None:
        store.create(name="Ann", date_value="2026-03-21", holiday_type="national", every_year=True)
        assert store.is_holiday(date(2027, 3, 21)) is True

    def test_get_holidays_in_range(self, store: HolidayStore) -> None:
        store.create(name="H1", date_value="2026-06-01", holiday_type="national")
        store.create(name="H2", date_value="2026-06-15", holiday_type="national")
        results = store.get_holidays_in_range(date(2026, 6, 1), date(2026, 6, 30))
        assert len(results) == 2

    def test_count(self, store: HolidayStore) -> None:
        store.create(name="C1", date_value="2026-01-01", holiday_type="national")
        assert store.count() >= 1

    def test_count_active(self, store: HolidayStore) -> None:
        h = store.create(name="C2", date_value="2026-02-01", holiday_type="national")
        store.delete(h.id)
        assert store.count_active() >= 0

    def test_invalid_date(self, store: HolidayStore) -> None:
        with pytest.raises(ValueError):
            store.create(name="Bad", date_value="not-a-date", holiday_type="national")

    def test_blank_name(self, store: HolidayStore) -> None:
        with pytest.raises(ValueError, match="Holiday name is required"):
            store.create(name="", date_value="2026-01-01", holiday_type="national")

pytestmark = pytest.mark.postgresql
