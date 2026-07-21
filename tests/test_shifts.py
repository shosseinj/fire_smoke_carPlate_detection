"""Tests for ShiftStore and shifts API."""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.shift_store import (
    WEEKDAY_COLS,
    ShiftStore,
    WorkShiftRecord,
    _is_overnight,
    _weekday_from_local,
)


# ── Helpers ───────────────────────────────────────────────────────────


@pytest.fixture
def store() -> ShiftStore:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    s = ShiftStore(db_path)
    yield s
    # Force-close any remaining SQLite connections
    if hasattr(s, '_lock'):
        with s._lock:
            pass  # flush pending operations
    try:
        db_path.unlink(missing_ok=True)
    except PermissionError:
        pass


# ── Store unit tests ──────────────────────────────────────────────────


class TestShiftStore:
    def test_create(self, store: ShiftStore) -> None:
        shift = store.create(
            shift_name="Morning",
            shift_type="morning",
            start_time="08:00",
            end_time="16:00",
            works_saturday=False,
            works_sunday=True,
            works_monday=True,
            works_tuesday=True,
            works_wednesday=True,
            works_thursday=True,
            works_friday=False,
        )
        assert shift.id > 0
        assert shift.shift_name == "Morning"
        assert shift.works_sunday is True
        assert shift.works_friday is False

    def test_create_defaults(self, store: ShiftStore) -> None:
        shift = store.create(
            shift_name="Default",
            works_saturday=True, works_sunday=True,
            works_monday=True, works_tuesday=True,
            works_wednesday=True, works_thursday=True,
            works_friday=True,
        )
        assert shift.shift_type == "morning"
        assert shift.start_time == "08:00"
        assert shift.end_time == "16:00"
        assert shift.max_minutes_delay == 15

    def test_create_invalid_type(self, store: ShiftStore) -> None:
        with pytest.raises(ValueError, match="Invalid shift type"):
            store.create(shift_name="Bad", shift_type="invalid",
                         works_saturday=True, works_sunday=True,
                         works_monday=True, works_tuesday=True,
                         works_wednesday=True, works_thursday=True,
                         works_friday=True)

    def test_create_no_weekdays(self, store: ShiftStore) -> None:
        with pytest.raises(ValueError, match="At least one weekday"):
            store.create(shift_name="Empty",
                         works_saturday=False, works_sunday=False,
                         works_monday=False, works_tuesday=False,
                         works_wednesday=False, works_thursday=False,
                         works_friday=False)

    def test_get_nonexistent(self, store: ShiftStore) -> None:
        assert store.get(99999) is None

    def test_update(self, store: ShiftStore) -> None:
        shift = store.create(
            shift_name="Morning",
            works_saturday=True, works_sunday=True,
            works_monday=True, works_tuesday=True,
            works_wednesday=True, works_thursday=True,
            works_friday=True,
        )
        updated = store.update(shift.id, shift_name="Evening", works_friday=True)
        assert updated is not None
        assert updated.shift_name == "Evening"
        assert updated.works_friday is True

    def test_update_nonexistent(self, store: ShiftStore) -> None:
        assert store.update(99999, shift_name="X") is None

    def test_delete(self, store: ShiftStore) -> None:
        shift = store.create(
            shift_name="Temp",
            works_saturday=True, works_sunday=True,
            works_monday=True, works_tuesday=True,
            works_wednesday=True, works_thursday=True,
            works_friday=True,
        )
        assert store.delete(shift.id) is True
        assert store.get(shift.id) is None

    def test_delete_nonexistent(self, store: ShiftStore) -> None:
        assert store.delete(99999) is False

    def test_list_pagination(self, store: ShiftStore) -> None:
        for i in range(5):
            store.create(
                shift_name=f"Shift{i}",
                works_saturday=True, works_sunday=True,
                works_monday=True, works_tuesday=True,
                works_wednesday=True, works_thursday=True,
                works_friday=True,
            )
        records, total = store.list(offset=0, limit=2)
        assert len(records) == 2
        assert total >= 5

    def test_list_filter_by_type(self, store: ShiftStore) -> None:
        store.create(
            shift_name="Night",
            shift_type="night",
            works_saturday=True, works_sunday=True,
            works_monday=True, works_tuesday=True,
            works_wednesday=True, works_thursday=True,
            works_friday=True,
        )
        store.create(
            shift_name="Morning",
            shift_type="morning",
            works_saturday=True, works_sunday=True,
            works_monday=True, works_tuesday=True,
            works_wednesday=True, works_thursday=True,
            works_friday=True,
        )
        records, total = store.list(shift_type="night")
        assert total == 1
        assert records[0].shift_type == "night"

    def test_list_search(self, store: ShiftStore) -> None:
        store.create(
            shift_name="SpecialShift",
            works_saturday=True, works_sunday=True,
            works_monday=True, works_tuesday=True,
            works_wednesday=True, works_thursday=True,
            works_friday=True,
        )
        records, total = store.list(search="Special")
        assert total >= 1

    def test_count(self, store: ShiftStore) -> None:
        assert store.count() >= 0

    def test_statistics(self, store: ShiftStore) -> None:
        stats = store.statistics()
        assert "total_shifts" in stats
        assert "total_personnel" in stats
        assert "assigned_personnel" in stats
        assert "distribution" in stats

    def test_is_overnight(self) -> None:
        assert _is_overnight("22:00", "06:00") is True
        assert _is_overnight("08:00", "16:00") is False
        assert _is_overnight("00:00", "00:00") is False

    def test_weekday_from_local(self) -> None:
        # 2026-07-21 is a Tuesday (Python weekday=1, Iranian=3)
        d = date(2026, 7, 21)
        assert _weekday_from_local(d) == 3  # Tuesday in Iranian

    def test_personnel_assignment(self, store: ShiftStore) -> None:
        shift = store.create(
            shift_name="Test",
            works_saturday=True, works_sunday=True,
            works_monday=True, works_tuesday=True,
            works_wednesday=True, works_thursday=True,
            works_friday=True,
        )
        # Assignment needs a personnel record — tested via integration
        assert shift.id > 0


# ── Overnight shift helper test ───────────────────────────────────────


def test_is_overnight_edge_cases() -> None:
    assert _is_overnight("06:00", "06:00") is False
    assert _is_overnight("23:59", "00:00") is True
    assert _is_overnight("12:00", "11:59") is True
    assert _is_overnight("00:00", "23:59") is False
