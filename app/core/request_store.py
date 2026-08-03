"""Personnel Request store — PostgreSQL with thread-safe RLock pattern."""

from __future__ import annotations

from app.database import Connection, Database, IntegrityError, OperationalError, Row, ensure_database
from app.time_utils import utc_now_text

import logging
import threading
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


VALID_REQUEST_TYPES = frozenset({
    "leave",
    "sick_leave",
    "remote_work",
    "mission",
    "overtime",
    "personal",
    "other",
    "earned_leave",
    "unpaid_leave",
})

VALID_REQUEST_STATUSES = frozenset({
    "pending",
    "approved",
    "rejected",
    "cancelled",
})

VALID_DURATION_TYPES = frozenset({"daily", "hourly"})

# Default concurrency guard (no overlapping approved requests of the same type for same person)
OVERLAP_CHECK_TYPES = frozenset({"leave", "sick_leave", "remote_work", "mission", "personal"})


@dataclass(frozen=True, slots=True)
class PersonnelRequestRecord:
    id: int
    personnel_id: int
    request_type: str
    duration_type: str | None
    start_date: str  # ISO date YYYY-MM-DD
    end_date: str  # ISO date YYYY-MM-DD
    start_time: str | None
    end_time: str | None
    duration_days: float | None
    duration_minutes: int | None
    reason: str | None
    status: str
    approved_by: int | None
    reviewed_by: int | None
    reviewed_at: str | None
    admin_notes: str | None
    rejection_reason: str | None
    created_at_utc: str
    updated_at_utc: str


def _now() -> str:
    return utc_now_text()


class RequestStore:
    """PostgreSQL-backed store for Personnel Requests with overlap validation."""

    def __init__(self, database: Database | str) -> None:
        self.database = ensure_database(database)
        self._lock = threading.RLock()
        self._init_db()

    def _connection(self) -> Connection:
        return self.database.connection()

    def _init_db(self) -> None:
        return None

    @staticmethod
    def _row_to_request(row: Row) -> PersonnelRequestRecord:
        return PersonnelRequestRecord(
            id=row["id"],
            personnel_id=row["personnel_id"],
            request_type=row["request_type"],
            duration_type=row.get("duration_type"),
            start_date=row["start_date"],
            end_date=row["end_date"],
            start_time=row.get("start_time"),
            end_time=row.get("end_time"),
            duration_days=row.get("duration_days"),
            duration_minutes=row.get("duration_minutes"),
            reason=row["reason"],
            status=row["status"],
            approved_by=row["approved_by"],
            reviewed_by=row.get("reviewed_by"),
            reviewed_at=row.get("reviewed_at"),
            admin_notes=row.get("admin_notes"),
            rejection_reason=row["rejection_reason"],
            created_at_utc=row["created_at_utc"],
            updated_at_utc=row["updated_at_utc"],
        )

    def _normalize_date(self, s: str) -> str:
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
            if 1200 <= y <= 1500:
                try:
                    g = parse_jalali_date(s)
                    return g.isoformat()
                except ValueError:
                    pass
            try:
                g = date(y, m, d_val)
                return g.isoformat()
            except (ValueError, IndexError):
                raise ValueError(f"Invalid Gregorian date: {s!r}")
        try:
            g = parse_jalali_date(s)
            return g.isoformat()
        except ValueError:
            raise ValueError(f"Invalid date: {s!r}")
    
    @staticmethod
    def _parse_time(t: str | None) -> str | None:
        if t is None:
            return None
        parts = t.split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid time: {t!r}")
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}:00"

    def _check_overlap(
        self, conn: Connection, personnel_id: int, request_type: str,
        start_date: str, end_date: str, exclude_id: int | None = None,
    ) -> None:
        if request_type not in OVERLAP_CHECK_TYPES:
            return
        exclude_clause = ""
        params: list[Any] = [personnel_id, request_type, start_date, end_date]
        if exclude_id is not None:
            exclude_clause = " AND id != ?"
            params.append(exclude_id)
        row = conn.execute(
            "SELECT id FROM personnel_requests WHERE personnel_id = ? AND request_type = ? "
            "AND status = 'approved' AND start_date <= ? AND end_date >= ?"
            + exclude_clause,
            params,
        ).fetchone()
        if row is not None:
            raise ValueError(
                f"Overlapping approved {request_type} request already exists "
                f"(id={row['id']}) for personnel {personnel_id}"
            )

    def create(
        self,
        personnel_id: int,
        request_type: str = "leave",
        start_date: str = "",
        end_date: str = "",
        duration_type: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        duration_days: float | None = None,
        duration_minutes: int | None = None,
        reason: str | None = None,
        status: str = "pending",
        reviewed_at: str | None = None,
        rejection_reason: str | None = None,
    ) -> PersonnelRequestRecord:
        if request_type not in VALID_REQUEST_TYPES:
            raise ValueError(f"Invalid request type: {request_type!r}")
        if not start_date or not end_date:
            raise ValueError("تاریخ شروع و پایان الزامی است")
        s = self._normalize_date(start_date)
        e = self._normalize_date(end_date)
        if s > e:
            raise ValueError("تاریخ شروع نباید بعد از تاریخ پایان باشد")
        st = self._parse_time(start_time)
        et = self._parse_time(end_time)
        if status not in VALID_REQUEST_STATUSES:
            raise ValueError(f"Invalid status: {status!r}")
        now = _now()
        with self._lock, self._connection() as conn:
            p = conn.execute(
                "SELECT id FROM personnel WHERE id = ?", (personnel_id,)
            ).fetchone()
            if p is None:
                raise ValueError(f"Personnel not found: {personnel_id}")
            row = conn.execute(
                "INSERT INTO personnel_requests "
                "(personnel_id, request_type, duration_type, start_date, end_date, "
                "start_time, end_time, duration_days, duration_minutes, "
                "reason, status, reviewed_at, rejection_reason, "
                "created_at_utc, updated_at_utc) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "RETURNING *",
                (personnel_id, request_type, duration_type, s, e,
                 st, et, duration_days, duration_minutes,
                 reason, status, reviewed_at, rejection_reason, now, now),
            ).fetchone()
            if row is None:
                raise RuntimeError("Failed to retrieve created request")
            return self._row_to_request(row)

    def create_many(
        self,
        personnel_id: int,
        requests: list[dict[str, Any]],
    ) -> list[PersonnelRequestRecord]:
        """Create multiple requests for one personnel in a single transaction."""
        if not requests:
            raise ValueError("At least one request is required")

        prepared: list[tuple[Any, ...]] = []
        now = _now()
        for item in requests:
            request_type = str(item.get("request_type", "leave"))
            start_date = str(item.get("start_date", ""))
            end_date = str(item.get("end_date", ""))
            status = str(item.get("status", "pending"))
            if request_type not in VALID_REQUEST_TYPES:
                raise ValueError(f"Invalid request type: {request_type!r}")
            if not start_date or not end_date:
                raise ValueError("Start and end dates are required")
            start = self._normalize_date(start_date)
            end = self._normalize_date(end_date)
            if start > end:
                raise ValueError("start_date must not be after end_date")
            if status not in VALID_REQUEST_STATUSES:
                raise ValueError(f"Invalid status: {status!r}")
            prepared.append(
                (
                    personnel_id,
                    request_type,
                    item.get("duration_type"),
                    start,
                    end,
                    self._parse_time(item.get("start_time")),
                    self._parse_time(item.get("end_time")),
                    item.get("duration_days"),
                    item.get("duration_minutes"),
                    item.get("reason"),
                    status,
                    item.get("reviewed_at"),
                    item.get("rejection_reason"),
                    now,
                    now,
                )
            )

        with self._lock, self._connection() as conn:
            personnel = conn.execute(
                "SELECT id FROM personnel WHERE id = ?", (personnel_id,)
            ).fetchone()
            if personnel is None:
                raise ValueError(f"Personnel not found: {personnel_id}")

            value_sql = "(" + ", ".join("?" for _ in range(15)) + ")"
            rows = conn.execute(
                "INSERT INTO personnel_requests "
                "(personnel_id, request_type, duration_type, start_date, end_date, "
                "start_time, end_time, duration_days, duration_minutes, "
                "reason, status, reviewed_at, rejection_reason, "
                "created_at_utc, updated_at_utc) VALUES "
                + ", ".join(value_sql for _ in prepared)
                + " RETURNING *",
                [value for row in prepared for value in row],
            ).fetchall()
            if len(rows) != len(prepared):
                raise RuntimeError("Failed to retrieve all created requests")
            return [self._row_to_request(row) for row in rows]

    def get(self, request_id: int) -> PersonnelRequestRecord | None:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM personnel_requests WHERE id = ?", (request_id,)
            ).fetchone()
            return self._row_to_request(row) if row is not None else None

    def update_status(
        self,
        request_id: int,
        status: str,
        reviewed_by: int | None = None,
        rejection_reason: str | None = None,
        admin_notes: str | None = None,
    ) -> PersonnelRequestRecord | None:
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                "SELECT * FROM personnel_requests WHERE id = ?", (request_id,)
            ).fetchone()
            if existing is None:
                return None
            if status not in VALID_REQUEST_STATUSES:
                raise ValueError(f"Invalid status: {status!r}")
            now = _now()
            conn.execute(
                "UPDATE personnel_requests SET status=?, reviewed_by=?, "
                "rejection_reason=?, admin_notes=?, reviewed_at=?, "
                "updated_at_utc=? WHERE id=?",
                (status, reviewed_by, rejection_reason, admin_notes, now, now, request_id),
            )
            row = conn.execute(
                "SELECT * FROM personnel_requests WHERE id = ?", (request_id,)
            ).fetchone()
            return self._row_to_request(row)

    def approve(
        self, request_id: int, approved_by: int, rejection_reason: str | None = None,
    ) -> PersonnelRequestRecord | None:
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                "SELECT * FROM personnel_requests WHERE id = ?", (request_id,)
            ).fetchone()
            if existing is None:
                return None
            if existing["status"] != "pending":
                raise ValueError(f"Request is already {existing['status']}")
            if rejection_reason:
                new_status = "rejected"
            else:
                self._check_overlap(
                    conn, existing["personnel_id"], existing["request_type"],
                    existing["start_date"], existing["end_date"], exclude_id=request_id,
                )
                new_status = "approved"
            now = _now()
            conn.execute(
                "UPDATE personnel_requests SET status=?, approved_by=?, reviewed_by=?, "
                "rejection_reason=?, reviewed_at=?, updated_at_utc=? WHERE id=?",
                (new_status, approved_by, approved_by, rejection_reason, now, now, request_id),
            )
            row = conn.execute(
                "SELECT * FROM personnel_requests WHERE id = ?", (request_id,)
            ).fetchone()
            return self._row_to_request(row)

    def reject(
        self, request_id: int, approved_by: int, rejection_reason: str = "",
    ) -> PersonnelRequestRecord | None:
        return self.approve(request_id, approved_by, rejection_reason=rejection_reason)

    def cancel(self, request_id: int) -> PersonnelRequestRecord | None:
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                "SELECT * FROM personnel_requests WHERE id = ?", (request_id,)
            ).fetchone()
            if existing is None:
                return None
            if existing["status"] != "pending":
                raise ValueError(f"Cannot cancel a request with status {existing['status']}")
            now = _now()
            conn.execute(
                "UPDATE personnel_requests SET status='cancelled', updated_at_utc=? WHERE id=?",
                (now, request_id),
            )
            row = conn.execute(
                "SELECT * FROM personnel_requests WHERE id = ?", (request_id,)
            ).fetchone()
            return self._row_to_request(row)

    def delete(self, request_id: int) -> bool:
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM personnel_requests WHERE id = ?", (request_id,)
            )
            return cursor.rowcount > 0

    def list(
        self,
        offset: int = 0,
        limit: int = 50,
        personnel_id: int | None = None,
        request_type: str | None = None,
        status: str | None = None,
        start_date_from: str | None = None,
        start_date_to: str | None = None,
        include_total: bool = True,
    ) -> tuple[list[PersonnelRequestRecord], int]:
        where_clauses: list[str] = []
        params: list[Any] = []
        if personnel_id is not None:
            where_clauses.append("personnel_id = ?")
            params.append(personnel_id)
        if request_type is not None:
            where_clauses.append("request_type = ?")
            params.append(request_type)
        if status is not None:
            where_clauses.append("status = ?")
            params.append(status)
        if start_date_from is not None:
            where_clauses.append("start_date >= ?")
            params.append(start_date_from)
        if start_date_to is not None:
            where_clauses.append("start_date <= ?")
            params.append(start_date_to)
        where = ""
        if where_clauses:
            where = " WHERE " + " AND ".join(where_clauses)
        with self._lock, self._connection() as conn:
            total = 0
            if include_total:
                total = int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM personnel_requests{where}", params
                    ).fetchone()[0]
                )
            rows = conn.execute(
                f"SELECT * FROM personnel_requests{where} "
                "ORDER BY created_at_utc DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            return [self._row_to_request(r) for r in rows], total

    def count(self) -> int:
        with self._lock, self._connection() as conn:
            return int(
                conn.execute("SELECT COUNT(*) FROM personnel_requests").fetchone()[0]
            )

    def count_by_status_for_personnel(self, personnel_id: int) -> dict[str, int]:
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM personnel_requests "
                "WHERE personnel_id = ? GROUP BY status",
                (personnel_id,),
            ).fetchall()
            return {r["status"]: int(r["cnt"]) for r in rows}

    def count_by_status(self) -> dict[str, int]:
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM personnel_requests GROUP BY status"
            ).fetchall()
            return {r["status"]: int(r["cnt"]) for r in rows}

    def count_by_type(self) -> dict[str, int]:
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT request_type, COUNT(*) as cnt FROM personnel_requests GROUP BY request_type"
            ).fetchall()
            return {r["request_type"]: int(r["cnt"]) for r in rows}

    def get_approved_requests_in_range(
        self, personnel_id: int | None, start: date, end: date,
    ) -> list[PersonnelRequestRecord]:
        end_str = end.isoformat()
        start_str = start.isoformat()
        params: list[Any] = [end_str, start_str]
        where = ""
        if personnel_id is not None:
            where = " AND personnel_id = ?"
            params.append(personnel_id)
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM personnel_requests WHERE status = 'approved' "
                "AND start_date <= ? AND end_date >= ?" + where,
                params,
            ).fetchall()
            return [self._row_to_request(r) for r in rows]
