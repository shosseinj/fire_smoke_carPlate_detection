"""WorkShift store — PostgreSQL with thread-safe RLock pattern."""

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
    timezone_name: str
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


@dataclass(frozen=True, slots=True)
class ShiftAssignmentRecord:
    id: int
    personnel_id: int
    shift_id: int
    start_date: date
    end_date: date
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
        # Alembic owns the PostgreSQL schema; runtime startup validates it.
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
            timezone_name=row.get("timezone_name", "Asia/Tehran"),
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

    @staticmethod
    def _row_to_assignment(row: Row) -> ShiftAssignmentRecord:
        start_date = row["start_date"]
        end_date = row["end_date"]
        return ShiftAssignmentRecord(
            id=int(row["id"]),
            personnel_id=int(row["personnel_id"]),
            shift_id=int(row["shift_id"]),
            start_date=(
                start_date
                if isinstance(start_date, date)
                else date.fromisoformat(str(start_date))
            ),
            end_date=(
                end_date
                if isinstance(end_date, date)
                else date.fromisoformat(str(end_date))
            ),
            created_at_utc=row["created_at_utc"],
            updated_at_utc=row["updated_at_utc"],
        )

    def _validate(self, name: str, shift_type: str, start_time: str, end_time: str,
                  max_minutes_delay: int, max_minutes_early: int,
                  max_overtime_hours: float, weekday_flags: dict[str, bool]) -> None:
        if not name.strip():
            raise ValueError("نام شیفت الزامی است")
        if shift_type not in VALID_SHIFT_TYPES:
            raise ValueError(f"Invalid shift type: {shift_type!r}. Must be one of: {sorted(VALID_SHIFT_TYPES)}")
        _validate_time(start_time)
        _validate_time(end_time)
        if max_minutes_delay < 0:
            raise ValueError("حداکثر دقیقه تأخیر باید غیرمنفی باشد")
        if max_minutes_early < 0:
            raise ValueError("حداکثر دقیقه زودآمدی باید غیرمنفی باشد")
        if max_overtime_hours < 0:
            raise ValueError("حداکثر ساعت اضافه‌کاری باید غیرمنفی باشد")
        if not any(weekday_flags.get(column, False) for column in WEEKDAY_COLS):
            raise ValueError("At least one weekday must be enabled")

    def create(
        self,
        shift_name: str,
        shift_type: str = "morning",
        start_time: str = "08:00",
        end_time: str = "16:00",
        timezone_name: str = "Asia/Tehran",
        max_minutes_delay: int = 0,
        max_minutes_early: int = 0,
        max_overtime_hours: float = 8.0,
        **weekday_flags: bool,
    ) -> WorkShiftRecord:
        self._validate(shift_name, shift_type, start_time, end_time,
                       max_minutes_delay, max_minutes_early, max_overtime_hours, weekday_flags)
        now = _now()
        with self._lock, self._connection() as conn:
            cols = ["shift_name", "shift_type", "start_time", "end_time", "timezone_name",
                    "max_minutes_delay", "max_minutes_early", "max_overtime_hours",
                    *WEEKDAY_COLS, "created_at_utc", "updated_at_utc"]
            placeholders = ", ".join("?" for _ in cols)
            values = [
                shift_name.strip(), shift_type, start_time, end_time, timezone_name,
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

    def get_by_name(self, shift_name: str, exclude_id: int | None = None) -> WorkShiftRecord | None:
        with self._lock, self._connection() as conn:
            query = "SELECT * FROM work_shifts WHERE shift_name = ?"
            parameters: list[Any] = [shift_name]
            if exclude_id is not None:
                query += " AND id != ?"
                parameters.append(exclude_id)
            row = conn.execute(query, parameters).fetchone()
            return self._row_to_shift(row) if row is not None else None

    def update(
        self,
        shift_id: int,
        shift_name: str | None = None,
        shift_type: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        timezone_name: str | None = None,
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
            new_tz = timezone_name if timezone_name is not None else existing.get("timezone_name", "Asia/Tehran")
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
            set_cols = ["shift_name", "shift_type", "start_time", "end_time", "timezone_name",
                        "max_minutes_delay", "max_minutes_early", "max_overtime_hours",
                        *WEEKDAY_COLS, "updated_at_utc"]
            set_clauses = ", ".join(f"{col} = ?" for col in set_cols)
            values = [new_name, new_type, new_start, new_end, new_tz,
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

    def delete(self, shift_id: int, force: bool = False) -> bool:
        """Delete a shift.

        Without force=True, raise ValueError if any personnel are assigned.
        With force=True, set affected personnel's shift_id to NULL atomically.

        Returns True if a shift was deleted.
        """
        with self._lock, self._connection() as conn:
            if not force:
                count = conn.execute(
                    "SELECT COUNT(DISTINCT personnel_id) FROM personnel_shift_assignments WHERE shift_id = ?", (shift_id,)
                ).fetchone()[0]
                if count > 0:
                    raise ValueError(
                        f"Cannot delete shift. This shift is assigned to {count} "
                        f"personnel. Use force=true for forced deletion."
                    )
            else:
                conn.execute(
                    "DELETE FROM personnel_shift_assignments WHERE shift_id = ?",
                    (shift_id,),
                )
                conn.execute(
                    "UPDATE personnel SET shift_id = NULL WHERE shift_id = ?",
                    (shift_id,),
                )
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

    def assign_personnel(
        self, personnel_id: int, shift_id: int, start_date: date, end_date: date
    ) -> ShiftAssignmentRecord:
        """Assign a shift for an inclusive date range without personnel overlap."""
        if end_date < start_date:
            raise ValueError("تاریخ پایان نمی‌تواند قبل از تاریخ شروع باشد")
        with self._lock, self._connection() as conn:
            shift = conn.execute(
                "SELECT id FROM work_shifts WHERE id = ?", (shift_id,)
            ).fetchone()
            if shift is None:
                raise ValueError("شیفت یافت نشد")
            if conn.execute("SELECT id FROM personnel WHERE id = ?", (personnel_id,)).fetchone() is None:
                raise ValueError("پرسنل یافت نشد")
            overlap = conn.execute(
                "SELECT id FROM personnel_shift_assignments "
                "WHERE personnel_id = ? AND start_date <= ? AND end_date >= ? LIMIT 1",
                (personnel_id, end_date, start_date),
            ).fetchone()
            if overlap is not None:
                raise ValueError("بازه تاریخ شیفت با شیفت دیگری برای این پرسنل هم‌پوشانی دارد")
            now = _now()
            try:
                cursor = conn.execute(
                    "INSERT INTO personnel_shift_assignments "
                    "(personnel_id, shift_id, start_date, end_date, created_at_utc, updated_at_utc) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (personnel_id, shift_id, start_date, end_date, now, now),
                )
            except IntegrityError as exc:
                if "overlap" in str(exc).lower() or "ex_personnel" in str(exc).lower():
                    raise ValueError("بازه تاریخ شیفت با شیفت دیگری برای این پرسنل هم‌پوشانی دارد") from exc
                raise
            # Compatibility projection for consumers not migrated to dated lookup yet.
            today = date.today()
            effective = conn.execute(
                "SELECT shift_id FROM personnel_shift_assignments WHERE personnel_id = ? "
                "AND start_date <= ? AND end_date >= ? LIMIT 1",
                (personnel_id, today, today),
            ).fetchone()
            conn.execute(
                "UPDATE personnel SET shift_id = ? WHERE id = ?",
                (effective["shift_id"] if effective is not None else None, personnel_id),
            )
            row = conn.execute(
                "SELECT * FROM personnel_shift_assignments WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            return self._row_to_assignment(row)

    def assign_personnel_bulk(
        self,
        personnel_ids: list[int],
        shift_id: int,
        start_date: date,
        end_date: date,
        skip_failed_records: bool = False,
    ) -> tuple[list[ShiftAssignmentRecord], list[dict[str, Any]]]:
        """Assign one dated shift to multiple personnel, optionally skipping failures."""
        if not personnel_ids:
            raise ValueError("حداقل یک پرسنل باید انتخاب شود")
        if len(personnel_ids) != len(set(personnel_ids)):
            raise ValueError("شناسه پرسنل تکراری مجاز نیست")
        if end_date < start_date:
            raise ValueError("تاریخ پایان نمی‌تواند قبل از تاریخ شروع باشد")

        with self._lock, self._connection() as conn:
            if conn.execute(
                "SELECT id FROM work_shifts WHERE id = ?", (shift_id,)
            ).fetchone() is None:
                raise ValueError("شیفت یافت نشد")

            placeholders = ", ".join("?" for _ in personnel_ids)
            existing_rows = conn.execute(
                f"SELECT id, national_code FROM personnel WHERE id IN ({placeholders})",
                personnel_ids,
            ).fetchall()
            personnel_codes = {
                int(row["id"]): str(row["national_code"]) for row in existing_rows
            }
            existing_ids = set(personnel_codes)
            missing_ids = [item for item in personnel_ids if item not in existing_ids]
            if missing_ids and not skip_failed_records:
                raise ValueError(
                    "پرسنل یافت نشد: " + ", ".join(str(item) for item in missing_ids)
                )

            overlap_rows = conn.execute(
                "SELECT DISTINCT a.personnel_id, p.national_code "
                "FROM personnel_shift_assignments a "
                "JOIN personnel p ON p.id = a.personnel_id "
                f"WHERE a.personnel_id IN ({placeholders}) "
                "AND a.start_date <= ? AND a.end_date >= ? ORDER BY a.personnel_id",
                [*personnel_ids, end_date, start_date],
            ).fetchall()
            overlap_ids = {int(row["personnel_id"]) for row in overlap_rows}
            overlap_codes = [str(row["national_code"]) for row in overlap_rows]
            if overlap_rows and not skip_failed_records:
                raise ValueError(
                    "بازه تاریخ شیفت برای این پرسنل هم‌پوشانی دارد: "
                    + ", ".join(overlap_codes)
                )

            failed_records: list[dict[str, Any]] = [
                {
                    "personnel_id": item,
                    "national_code": None,
                    "reason": "personnel_not_found",
                }
                for item in missing_ids
            ]
            failed_records.extend(
                {
                    "national_code": str(row["national_code"]),
                    "reason": "date_overlap",
                }
                for row in overlap_rows
            )
            valid_ids = [
                item
                for item in personnel_ids
                if item in existing_ids and item not in overlap_ids
            ]

            now = _now()
            assignment_ids: list[int] = []
            try:
                for personnel_id in valid_ids:
                    cursor = conn.execute(
                        "INSERT INTO personnel_shift_assignments "
                        "(personnel_id, shift_id, start_date, end_date, created_at_utc, updated_at_utc) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (personnel_id, shift_id, start_date, end_date, now, now),
                    )
                    assignment_ids.append(int(cursor.lastrowid))
            except IntegrityError as exc:
                if "overlap" in str(exc).lower() or "ex_personnel" in str(exc).lower():
                    raise ValueError("بازه تاریخ شیفت با شیفت دیگری هم‌پوشانی دارد") from exc
                raise

            today = date.today()
            if valid_ids and start_date <= today <= end_date:
                valid_placeholders = ", ".join("?" for _ in valid_ids)
                conn.execute(
                    f"UPDATE personnel SET shift_id = ? WHERE id IN ({valid_placeholders})",
                    [shift_id, *valid_ids],
                )
            if not assignment_ids:
                return [], failed_records
            rows = conn.execute(
                "SELECT * FROM personnel_shift_assignments "
                f"WHERE id IN ({', '.join('?' for _ in assignment_ids)}) ORDER BY id",
                assignment_ids,
            ).fetchall()
            return [self._row_to_assignment(row) for row in rows], failed_records

    def get_assignment_for_date(
        self, personnel_id: int, on_date: date
    ) -> ShiftAssignmentRecord | None:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM personnel_shift_assignments WHERE personnel_id = ? "
                "AND start_date <= ? AND end_date >= ? ORDER BY start_date DESC LIMIT 1",
                (personnel_id, on_date, on_date),
            ).fetchone()
            return self._row_to_assignment(row) if row is not None else None

    def list_assignments(
        self,
        personnel_id: int,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[ShiftAssignmentRecord]:
        clauses = ["personnel_id = ?"]
        params: list[Any] = [personnel_id]
        if start_date is not None:
            clauses.append("end_date >= ?")
            params.append(start_date)
        if end_date is not None:
            clauses.append("start_date <= ?")
            params.append(end_date)
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM personnel_shift_assignments WHERE "
                + " AND ".join(clauses)
                + " ORDER BY start_date, id",
                params,
            ).fetchall()
            return [self._row_to_assignment(row) for row in rows]

    def delete_assignment(self, assignment_id: int, shift_id: int | None = None) -> bool:
        with self._lock, self._connection() as conn:
            scope = " AND shift_id = ?" if shift_id is not None else ""
            parameters: tuple[Any, ...] = (
                (assignment_id, shift_id) if shift_id is not None else (assignment_id,)
            )
            row = conn.execute(
                "SELECT personnel_id FROM personnel_shift_assignments WHERE id = ?" + scope,
                parameters,
            ).fetchone()
            if row is None:
                return False
            personnel_id = int(row["personnel_id"])
            conn.execute("DELETE FROM personnel_shift_assignments WHERE id = ?", (assignment_id,))
            today = date.today()
            replacement = conn.execute(
                "SELECT shift_id FROM personnel_shift_assignments WHERE personnel_id = ? "
                "AND start_date <= ? AND end_date >= ? ORDER BY start_date DESC LIMIT 1",
                (personnel_id, today, today),
            ).fetchone()
            conn.execute(
                "UPDATE personnel SET shift_id = ? WHERE id = ?",
                (replacement["shift_id"] if replacement is not None else None, personnel_id),
            )
            return True

    def remove_personnel_shift(self, personnel_id: int) -> bool:
        """Remove a Personnel member's shift assignment."""
        with self._lock, self._connection() as conn:
            deleted = conn.execute(
                "DELETE FROM personnel_shift_assignments WHERE personnel_id = ?", (personnel_id,)
            )
            cursor = conn.execute(
                "UPDATE personnel SET shift_id = NULL WHERE id = ? AND shift_id IS NOT NULL",
                (personnel_id,),
            )
            return deleted.rowcount > 0 or cursor.rowcount > 0

    def list_personnel_in_shift(self, shift_id: int) -> list[dict[str, Any]]:
        """Return Personnel records assigned to a shift (legacy format)."""
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT p.id, p.fname, p.lname, p.national_code, p.department_id, "
                "p.degree, s.shift_name, s.shift_type, s.start_time, s.end_time, "
                "s.timezone_name, s.max_minutes_delay, s.max_minutes_early, "
                "s.max_overtime_hours, s.works_saturday, s.works_sunday, "
                "s.works_monday, s.works_tuesday, s.works_wednesday, "
                "s.works_thursday, s.works_friday "
                "FROM personnel_shift_assignments a JOIN personnel p ON p.id = a.personnel_id "
                "JOIN work_shifts s ON a.shift_id = s.id "
                "WHERE a.shift_id = ? GROUP BY p.id, s.id ORDER BY p.lname, p.fname",
                (shift_id,),
            ).fetchall()
            result: list[dict[str, Any]] = []
            for r in rows:
                shift_info: dict[str, Any] | None = None
                if r["shift_name"] is not None:
                    shift_info = {
                        "id": shift_id,
                        "shift_name": r["shift_name"],
                        "shift_type": r["shift_type"],
                        "start_time": r["start_time"],
                        "end_time": r["end_time"],
                        "timezone_name": r["timezone_name"] or "Asia/Tehran",
                        "max_minutes_delay": r["max_minutes_delay"],
                        "max_minutes_early": r["max_minutes_early"],
                        "max_overtime_hours": float(r["max_overtime_hours"]) if r["max_overtime_hours"] else 0,
                        "monday": bool(r["works_monday"]),
                        "tuesday": bool(r["works_tuesday"]),
                        "wednesday": bool(r["works_wednesday"]),
                        "thursday": bool(r["works_thursday"]),
                        "friday": bool(r["works_friday"]),
                        "saturday": bool(r["works_saturday"]),
                        "sunday": bool(r["works_sunday"]),
                        "personnel_count": 0,
                    }
                result.append({
                    "personnel_id": r["id"],
                    "full_name": f"{r['fname']} {r['lname']}",
                    "national_code": r["national_code"],
                    "department_id": r["department_id"],
                    "department_name": None,
                    "degree": r["degree"],
                    "shift": shift_info,
                })
            return result

    def count_personnel_in_shift(self, shift_id: int) -> int:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT personnel_id) FROM personnel_shift_assignments WHERE shift_id = ?", (shift_id,)
            ).fetchone()
            return int(row[0])

    # ── Statistics ───────────────────────────────────────────────────

    def statistics(self) -> dict[str, Any]:
        """Return shift distribution statistics (legacy format)."""
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
                        "SELECT COUNT(DISTINCT personnel_id) FROM personnel_shift_assignments"
                    ).fetchone()[0]
                )
                unassigned = total_personnel - assigned
            except OperationalError:
                pass
            try:
                shifts = conn.execute(
                    "SELECT s.id, s.shift_name, s.shift_type, COUNT(DISTINCT p.id) as cnt "
                    "FROM work_shifts s LEFT JOIN personnel_shift_assignments a ON a.shift_id = s.id "
                    "LEFT JOIN personnel p ON p.id = a.personnel_id "
                    "GROUP BY s.id, s.shift_name, s.shift_type ORDER BY s.shift_name"
                ).fetchall()
            except OperationalError:
                shifts = conn.execute(
                    "SELECT id, shift_name, shift_type, 0 as cnt FROM work_shifts ORDER BY shift_name"
                ).fetchall()
            distribution = [
                {
                    "shift_name": r["shift_name"],
                    "shift_type": r["shift_type"],
                    "personnel_count": int(r["cnt"]),
                }
                for r in shifts
                if int(r["cnt"]) > 0
            ]
            return {
                "total_shifts": self.count(),
                "total_personnel": total_personnel,
                "assigned_personnel": assigned,
                "unassigned_personnel": unassigned,
                "shift_distribution": distribution,
            }
