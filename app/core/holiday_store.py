"""Holiday store — PostgreSQL with thread-safe RLock pattern."""

from __future__ import annotations

from app.database import Connection, Database, IntegrityError, OperationalError, Row, ensure_database
from app.time_utils import utc_now_text

import logging
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


VALID_HOLIDAY_TYPES = frozenset({"national", "religious", "company", "other"})


@dataclass(frozen=True, slots=True)
class HolidayRecord:
    id: int
    name: str
    date_value: str  # ISO date YYYY-MM-DD
    description: str | None
    holiday_type: str
    every_year: bool
    is_active: bool
    created_by: int | None
    updated_by: int | None
    created_at_utc: str
    updated_at_utc: str


def _now() -> str:
    return utc_now_text()


class HolidayStore:
    """PostgreSQL-backed store for Holidays with annual recurrence support."""

    def __init__(self, database: Database | str) -> None:
        self.database = ensure_database(database)
        self._lock = threading.RLock()
        self._init_db()

    def _connection(self) -> Connection:
        return self.database.connection()

    def _init_db(self) -> None:
        # Alembic owns the PostgreSQL schema; runtime startup validates it.
        return None

    @staticmethod
    def _row_to_holiday(row: Row) -> HolidayRecord:
        return HolidayRecord(
            id=row["id"],
            name=row["name"],
            date_value=row["date_value"],
            description=row["description"],
            holiday_type=row["holiday_type"],
            every_year=bool(row["every_year"]),
            is_active=bool(row["is_active"]),
            created_by=row.get("created_by"),
            updated_by=row.get("updated_by"),
            created_at_utc=row["created_at_utc"],
            updated_at_utc=row["updated_at_utc"],
        )

    def _validate_date_str(self, s: str) -> str:
        """Validate and return normalized ISO date string.

        Supports Gregorian (YYYY-MM-DD) and Jalali (YYYY-MM-DD with year 1200-1500).
        """
        from app.core.jalali_utils import normalize_digits, parse_jalali_date

        s = normalize_digits(s.strip())
        parts = s.split("-")
        if len(parts) == 3:
            try:
                y = int(parts[0])
                m = int(parts[1])
                d_val = int(parts[2])
            except (ValueError, IndexError):
                raise ValueError(f"Invalid date format: {s!r}")
            # Jalali years are typically 1200-1500; Gregorian years outside that range
            if 1200 <= y <= 1500:
                # Try Jalali first
                try:
                    g = parse_jalali_date(s)
                    return g.isoformat()
                except ValueError:
                    pass
            # Try Gregorian ISO
            try:
                g = date(y, m, d_val)
                return g.isoformat()
            except (ValueError, IndexError):
                raise ValueError(f"Invalid Gregorian date: {s!r}")
        # Try Jalali as last resort
        try:
            g = parse_jalali_date(s)
            return g.isoformat()
        except ValueError:
            raise ValueError(f"Invalid date: {s!r}. Use YYYY-MM-DD Gregorian or Jalali date.")

    def create(
        self,
        name: str,
        date_value: str,
        description: str | None = None,
        holiday_type: str = "national",
        every_year: bool = False,
        created_by: int | None = None,
    ) -> HolidayRecord:
        name = name.strip()
        if not name:
            raise ValueError("Holiday name is required")
        if holiday_type not in VALID_HOLIDAY_TYPES:
            raise ValueError(f"Invalid holiday type: {holiday_type!r}")
        normalized_date = self._validate_date_str(date_value)
        now = _now()
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                "SELECT id FROM holidays WHERE date_value = ? AND every_year = ? AND is_active = 1",
                (normalized_date, 1 if every_year else 0),
            ).fetchone()
            if existing is not None:
                raise ValueError(
                    f"Active holiday already exists for date {normalized_date}"
                )
            cursor = conn.execute(
                "INSERT INTO holidays (name, date_value, description, holiday_type, every_year, "
                "is_active, created_at_utc, updated_at_utc, created_by) "
                "VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)",
                (name, normalized_date, description, holiday_type, 1 if every_year else 0,
                 now, now, created_by),
            )
            row = conn.execute(
                "SELECT * FROM holidays WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            if row is None:
                raise RuntimeError("Failed to retrieve created holiday")
            return self._row_to_holiday(row)

    def get(self, holiday_id: int) -> HolidayRecord | None:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM holidays WHERE id = ?", (holiday_id,)
            ).fetchone()
            return self._row_to_holiday(row) if row is not None else None

    def update(
        self,
        holiday_id: int,
        name: str | None = None,
        date_value: str | None = None,
        description: str | None = None,
        holiday_type: str | None = None,
        every_year: bool | None = None,
        is_active: bool | None = None,
        updated_by: int | None = None,
    ) -> HolidayRecord | None:
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                "SELECT * FROM holidays WHERE id = ?", (holiday_id,)
            ).fetchone()
            if existing is None:
                return None
            new_name = name.strip() if name else existing["name"]
            if name is not None and not new_name:
                raise ValueError("Holiday name cannot be blank")
            new_date = self._validate_date_str(date_value) if date_value is not None else existing["date_value"]
            new_desc = description if description is not None else existing["description"]
            new_type = holiday_type if holiday_type is not None else existing["holiday_type"]
            if holiday_type is not None and new_type not in VALID_HOLIDAY_TYPES:
                raise ValueError(f"Invalid holiday type: {new_type!r}")
            new_every = 1 if (every_year if every_year is not None else existing["every_year"]) else 0
            new_active = is_active if is_active is not None else bool(existing["is_active"])

            # Check duplicate on reactivation or date change
            check_date = new_date
            check_every = bool(new_every)
            conn.execute(
                "SELECT id FROM holidays WHERE date_value = ? AND every_year = ? "
                "AND is_active = 1 AND id != ?",
                (check_date, new_every, holiday_id),
            )
            dup = conn.execute(
                "SELECT id FROM holidays WHERE date_value = ? AND every_year = ? "
                "AND is_active = 1 AND id != ?",
                (check_date, new_every, holiday_id),
            ).fetchone()
            if dup is not None:
                raise ValueError(
                    f"Active holiday already exists for date {new_date}"
                )

            now = _now()
            conn.execute(
                "UPDATE holidays SET name=?, date_value=?, description=?, holiday_type=?, "
                "every_year=?, is_active=?, updated_at_utc=?, updated_by=? WHERE id=?",
                (new_name, new_date, new_desc, new_type, new_every, 1 if new_active else 0,
                 now, updated_by, holiday_id),
            )
            row = conn.execute(
                "SELECT * FROM holidays WHERE id = ?", (holiday_id,)
            ).fetchone()
            return self._row_to_holiday(row)

    def delete(self, holiday_id: int, updated_by: int | None = None) -> bool:
        """Deactivate a holiday rather than physically removing it."""
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "UPDATE holidays SET is_active = 0, updated_at_utc = ? WHERE id = ?",
                (_now(), holiday_id),
            )
            return cursor.rowcount > 0

    def hard_delete(self, holiday_id: int) -> bool:
        """Physically remove a holiday. Use only for test cleanup."""
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM holidays WHERE id = ?", (holiday_id,)
            )
            return cursor.rowcount > 0

    def list(
        self,
        offset: int = 0,
        limit: int = 50,
        holiday_type: str | None = None,
        is_active: bool | None = None,
        search: str | None = None,
    ) -> tuple[list[HolidayRecord], int]:
        where_clauses: list[str] = []
        params: list[Any] = []
        if holiday_type is not None:
            where_clauses.append("holiday_type = ?")
            params.append(holiday_type)
        if is_active is not None:
            where_clauses.append("is_active = ?")
            params.append(1 if is_active else 0)
        if search is not None:
            where_clauses.append("(name LIKE ? OR description LIKE ?)")
            pattern = f"%{search}%"
            params.extend([pattern, pattern])
        where = ""
        if where_clauses:
            where = " WHERE " + " AND ".join(where_clauses)
        with self._lock, self._connection() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM holidays{where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM holidays{where} ORDER BY date_value DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            return [self._row_to_holiday(r) for r in rows], int(total)

    def is_holiday(self, d: date) -> bool:
        """Check if a given date is an active holiday (including annual recurrence)."""
        d_str = d.isoformat()
        month_day = f"{d.month:02d}-{d.day:02d}"
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT id FROM holidays WHERE is_active = 1 AND "
                "(date_value = ? OR (every_year = 1 AND to_char(date_value, 'MM-DD') = ?))",
                (d_str, month_day),
            ).fetchone()
            return row is not None

    def get_holidays_in_range(self, start: date, end: date) -> list[HolidayRecord]:
        """Return active holidays overlapping [start, end], including annual recurrences."""
        start_str = start.isoformat()
        end_str = end.isoformat()
        with self._lock, self._connection() as conn:
            # One-time holidays within range
            rows = conn.execute(
                "SELECT * FROM holidays WHERE is_active = 1 AND "
                "date_value >= ? AND date_value <= ?",
                (start_str, end_str),
            ).fetchall()
            results = [self._row_to_holiday(r) for r in rows]
            # Annual holidays — check if month-day falls within range
            annual_rows = conn.execute(
                "SELECT * FROM holidays WHERE is_active = 1 AND every_year = 1"
            ).fetchall()
            for row in annual_rows:
                h = self._row_to_holiday(row)
                h_date = date.fromisoformat(h.date_value)
                # For annual holidays, we check if the month-day occurs within the range
                # by constructing dates for the current year
                for yr in range(start.year, end.year + 1):
                    try:
                        candidate = date(yr, h_date.month, h_date.day)
                    except ValueError:
                        continue
                    if start <= candidate <= end:
                        # Check if not already included as a one-time
                        if not any(r.date_value == candidate.isoformat() for r in results):
                            # Create a synthetic record with the correct year
                            from dataclasses import replace
                            results.append(replace(h, date_value=candidate.isoformat()))
                        break
            return results

    def count(self) -> int:
        with self._lock, self._connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM holidays").fetchone()[0])

    def count_active(self) -> int:
        with self._lock, self._connection() as conn:
            return int(
                conn.execute("SELECT COUNT(*) FROM holidays WHERE is_active = 1").fetchone()[0]
            )
