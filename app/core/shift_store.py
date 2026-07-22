"""WorkShift store — PostgreSQL with thread-safe RLock pattern."""

from __future__ import annotations

from app.database import Connection, Database, IntegrityError, OperationalError, Row, ensure_database
from app.time_utils import utc_now_text

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


# Supported shift types
VALID_SHIFT_TYPES = frozenset({
    "morning",
    "evening",
    "night",
    "remote",
    "flexible",
    "rotating",
})

# Weekday columns (Monday=0 ... Sunday=6, but we use Persian/Iranian convention:
# Saturday=0, Sunday=1, Monday=2, Tuesday=3, Wednesday=4, Thursday=5, Friday=6)
WEEKDAY_COLS = [
    "works_saturday",
    "works_sunday",
    "works_monday",
    "works_tuesday",
    "works_wednesday",
    "works_thursday",
    "works_friday",
]

WEEKDAY_NAMES = [
    "saturday", "sunday", "monday", "tuesday",
    "wednesday", "thursday", "friday",
]


@dataclass(frozen=True, slots=True)
class WorkShiftRecord:
    id: int
    shift_name: str
    shift_type: str
    start_time: str  # HH:MM local
    end_time: str  # HH:MM local
    max_minutes_delay: int
    max_minutes_early: int
    max_overtime_hours: float
    works_saturday: bool
    works_sunday: bool
    works_monday: bool
    works_tuesday: bool
    works_wednesday: bool
    works_thursday: bool
    works_friday: bool
    created_at_utc: str
    updated_at_utc: str


# ── Helpers ───────────────────────────────────────────────────────────


def _now() -> str:
    return utc_now_text()


def _validate_time(t: str) -> None:
    """Validate HH:MM format, allow 24:00 for midnight."""
    if t == "24:00":
        return
    parts = t.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format: {t!r}, expected HH:MM")
    try:
        h, m = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid time values: {t!r}")
    if not (0 <= h <= 23) or not (0 <= m <= 59):
        raise ValueError(f"Time out of range: {t!r}")


def _is_overnight(start_time: str, end_time: str) -> bool:
    """Return True if the shift crosses midnight (end_time <= start_time)."""
    if start_time == end_time:
        return False
    # Compare as (hours, minutes)
    sh, sm = (int(x) for x in start_time.split(":"))
    eh, em = (int(x) for x in end_time.split(":"))
    start_minutes = sh * 60 + sm
    end_minutes = eh * 60 + em
    return end_minutes <= start_minutes


def _weekday_from_local(local_date: datetime) -> int:
    """Return weekday index in Iranian convention: Sat=0, Sun=1, ..., Fri=6.

    Python's weekday(): Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
    Mapping: sat=5, sun=6, mon=0, tue=1, wed=2, thu=3, fri=4
    """
    py_weekday = local_date.weekday()  # Mon=0 … Sun=6
    mapping = {5: 0, 6: 1, 0: 2, 1: 3, 2: 4, 3: 5, 4: 6}
    return mapping[py_weekday]


def _get_weekday_flag(record: WorkShiftRecord, weekday_idx: int) -> bool:
    """Get the boolean flag for a given weekday index (0=Sat … 6=Fri)."""
    return bool(getattr(record, WEEKDAY_COLS[weekday_idx]))


# ── Store ─────────────────────────────────────────────────────────────


class ShiftStore:
    """PostgreSQL-backed store for WorkShift records and Personnel assignments."""

    def __init__(self, database: Database | str) -> None:
        self.database = ensure_database(database)
        self._lock = threading.RLock()
        self._init_db()

    def _connection(self) -> Connection:
        return self.database.connection()

    def _init_db(self) -> None:
        # The shared Database creates and validates the PostgreSQL schema.
        return None

    # ── CRUD ──────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_shift(row: Row) -> WorkShiftRecord:
        return WorkShiftRecord(
            id=row["id"],
            shift_name=row["shift_name"],
            shift_type=row["shift_type"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            max_minutes_delay=row["max_minutes_delay"],
            max_minutes_early=row["max_minutes_early"],
            max_overtime_hours=float(row["max_overtime_hours"]),
            works_saturday=bool(row["works_saturday"]),
            works_sunday=bool(row["works_sunday"]),
            works_monday=bool(row["works_monday"]),
            works_tuesday=bool(row["works_tuesday"]),
            works_wednesday=bool(row["works_wednesday"]),
            works_thursday=bool(row["works_thursday"]),
            works_friday=bool(row["works_friday"]),
            created_at_utc=row["created_at_utc"],
            updated_at_utc=row["updated_at_utc"],
        )

    def _validate(self, name: str, shift_type: str, start_time: str, end_time: str,
                  max_minutes_delay: int, max_minutes_early: int,
                  max_overtime_hours: float, weekday_flags: dict[str, bool]) -> None:
        if not name.strip():
            raise ValueError("Shift name is required")
        if shift_type not in VALID_SHIFT_TYPES:
            raise ValueError(f"Invalid shift type: {shift_type!r}. Must be one of: {sorted(VALID_SHIFT_TYPES)}")
        _validate_time(start_time)
        _validate_time(end_time)
        if max_minutes_delay < 0:
            raise ValueError("max_minutes_delay must be nonnegative")
        if max_minutes_early < 0:
            raise ValueError("max_minutes_early must be nonnegative")
        if max_overtime_hours < 0:
            raise ValueError("max_overtime_hours must be nonnegative")
        if not any(weekday_flags.get(col, False) for col in WEEKDAY_COLS):
            raise ValueError("At least one weekday must be selected")

    def create(
        self,
        shift_name: str,
        shift_type: str = "morning",
        start_time: str = "08:00",
        end_time: str = "16:00",
        max_minutes_delay: int = 15,
        max_minutes_early: int = 15,
        max_overtime_hours: float = 2.0,
        **weekday_flags: bool,
    ) -> WorkShiftRecord:
        self._validate(shift_name, shift_type, start_time, end_time,
                       max_minutes_delay, max_minutes_early, max_overtime_hours, weekday_flags)
        now = _now()
        with self._lock, self._connection() as conn:
            cols = ["shift_name", "shift_type", "start_time", "end_time",
                    "max_minutes_delay", "max_minutes_early", "max_overtime_hours",
                    *WEEKDAY_COLS, "created_at_utc", "updated_at_utc"]
            placeholders = ", ".join("?" for _ in cols)
            values = [
                shift_name.strip(), shift_type, start_time, end_time,
                max_minutes_delay, max_minutes_early, max_overtime_hours,
            ]
            for col in WEEKDAY_COLS:
                values.append(1 if weekday_flags.get(col, False) else 0)
            values.extend([now, now])
            cursor = conn.execute(
                f"INSERT INTO work_shifts ({', '.join(cols)}) VALUES ({placeholders})",
                values,
            )
            row = conn.execute(
                "SELECT * FROM work_shifts WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            if row is None:
                raise RuntimeError("Failed to retrieve created shift")
            return self._row_to_shift(row)

    def get(self, shift_id: int) -> WorkShiftRecord | None:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM work_shifts WHERE id = ?", (shift_id,)
            ).fetchone()
            return self._row_to_shift(row) if row is not None else None

    def update(
        self,
        shift_id: int,
        shift_name: str | None = None,
        shift_type: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        max_minutes_delay: int | None = None,
        max_minutes_early: int | None = None,
        max_overtime_hours: float | None = None,
        **weekday_flags: bool,
    ) -> WorkShiftRecord | None:
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                "SELECT * FROM work_shifts WHERE id = ?", (shift_id,)
            ).fetchone()
            if existing is None:
                return None
            new_name = shift_name.strip() if shift_name else existing["shift_name"]
            new_type = shift_type if shift_type is not None else existing["shift_type"]
            new_start = start_time if start_time is not None else existing["start_time"]
            new_end = end_time if end_time is not None else existing["end_time"]
            new_delay = max_minutes_delay if max_minutes_delay is not None else existing["max_minutes_delay"]
            new_early = max_minutes_early if max_minutes_early is not None else existing["max_minutes_early"]
            new_overtime = max_overtime_hours if max_overtime_hours is not None else existing["max_overtime_hours"]
            merged_flags = {}
            for col in WEEKDAY_COLS:
                if col in weekday_flags:
                    merged_flags[col] = weekday_flags[col]
                else:
                    merged_flags[col] = bool(existing[col])
            if shift_name is not None or shift_type is not None or start_time is not None or end_time is not None:
                self._validate(new_name, new_type, new_start, new_end,
                               new_delay, new_early, new_overtime, merged_flags)
            now = _now()
            set_clauses = ", ".join(f"{col} = ?" for col in
                ["shift_name", "shift_type", "start_time", "end_time",
                 "max_minutes_delay", "max_minutes_early", "max_overtime_hours",
                 *WEEKDAY_COLS, "updated_at_utc"])
            values = [new_name, new_type, new_start, new_end,
                      new_delay, new_early, new_overtime]
            for col in WEEKDAY_COLS:
                values.append(1 if merged_flags[col] else 0)
            values.append(now)
            values.append(shift_id)
            conn.execute(
                f"UPDATE work_shifts SET {set_clauses} WHERE id = ?", values
            )
            row = conn.execute(
                "SELECT * FROM work_shifts WHERE id = ?", (shift_id,)
            ).fetchone()
            return self._row_to_shift(row)

    def delete(self, shift_id: int) -> bool:
        """Delete a shift. Affected Personnel get shift_id set to NULL via application logic.

        Returns True if a shift was deleted.
        """
        with self._lock, self._connection() as conn:
            # Set personnel shift_id to NULL for this shift (personnel table may not exist
            # when ShiftStore is used independently)
            try:
                conn.execute(
                    "UPDATE personnel SET shift_id = NULL WHERE shift_id = ?",
                    (shift_id,),
                )
            except OperationalError:
                pass
            cursor = conn.execute(
                "DELETE FROM work_shifts WHERE id = ?", (shift_id,)
            )
            return cursor.rowcount > 0

    def list(
        self,
        offset: int = 0,
        limit: int = 50,
        shift_type: str | None = None,
        search: str | None = None,
    ) -> tuple[list[WorkShiftRecord], int]:
        where_clauses: list[str] = []
        params: list[Any] = []
        if shift_type is not None:
            where_clauses.append("shift_type = ?")
            params.append(shift_type)
        if search is not None:
            where_clauses.append("shift_name LIKE ?")
            params.append(f"%{search}%")
        where = ""
        if where_clauses:
            where = " WHERE " + " AND ".join(where_clauses)
        with self._lock, self._connection() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM work_shifts{where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM work_shifts{where} ORDER BY id DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            return [self._row_to_shift(r) for r in rows], int(total)

    def count(self) -> int:
        with self._lock, self._connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM work_shifts").fetchone()[0])

    # ── Personnel assignment ─────────────────────────────────────────

    def assign_personnel(self, personnel_id: int, shift_id: int) -> bool:
        """Assign a Personnel member to a shift. Returns True if updated."""
        with self._lock, self._connection() as conn:
            # Verify shift exists
            shift = conn.execute(
                "SELECT id FROM work_shifts WHERE id = ?", (shift_id,)
            ).fetchone()
            if shift is None:
                raise ValueError(f"Shift not found: {shift_id}")
            cursor = conn.execute(
                "UPDATE personnel SET shift_id = ? WHERE id = ?",
                (shift_id, personnel_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Personnel not found: {personnel_id}")
            return True

    def remove_personnel_shift(self, personnel_id: int) -> bool:
        """Remove a Personnel member's shift assignment."""
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "UPDATE personnel SET shift_id = NULL WHERE id = ? AND shift_id IS NOT NULL",
                (personnel_id,),
            )
            return cursor.rowcount > 0

    def list_personnel_in_shift(self, shift_id: int) -> list[dict[str, Any]]:
        """Return Personnel records assigned to a shift."""
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT id, fname, lname, national_code, employee_type "
                "FROM personnel WHERE shift_id = ? ORDER BY lname, fname",
                (shift_id,),
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "fname": r["fname"],
                    "lname": r["lname"],
                    "national_code": r["national_code"],
                    "employee_type": r["employee_type"],
                }
                for r in rows
            ]

    # ── Statistics ───────────────────────────────────────────────────

    def statistics(self) -> dict[str, Any]:
        """Return shift distribution statistics."""
        with self._lock, self._connection() as conn:
            total_personnel = 0
            assigned = 0
            unassigned = 0
            try:
                total_personnel = int(
                    conn.execute("SELECT COUNT(*) FROM personnel").fetchone()[0]
                )
                assigned = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM personnel WHERE shift_id IS NOT NULL"
                    ).fetchone()[0]
                )
                unassigned = total_personnel - assigned
            except OperationalError:
                pass  # personnel table may not exist when used independently
            try:
                shifts = conn.execute(
                    "SELECT s.id, s.shift_name, COUNT(p.id) as cnt "
                    "FROM work_shifts s LEFT JOIN personnel p ON p.shift_id = s.id "
                    "GROUP BY s.id ORDER BY s.shift_name"
                ).fetchall()
            except OperationalError:
                shifts = conn.execute(
                    "SELECT id, shift_name, 0 as cnt FROM work_shifts ORDER BY shift_name"
                ).fetchall()
            distribution = [
                {
                    "shift_id": r["id"],
                    "shift_name": r["shift_name"],
                    "assigned_count": int(r["cnt"]),
                }
                for r in shifts
            ]
            return {
                "total_shifts": self.count(),
                "total_personnel": total_personnel,
                "assigned_personnel": assigned,
                "unassigned_personnel": unassigned,
                "distribution": distribution,
            }
