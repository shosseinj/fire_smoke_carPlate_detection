"""Personnel Request store — raw SQLite with thread-safe RLock pattern."""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone
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
})

VALID_REQUEST_STATUSES = frozenset({
    "pending",
    "approved",
    "rejected",
    "cancelled",
})

# Default concurrency guard (no overlapping approved requests of the same type for same person)
OVERLAP_CHECK_TYPES = frozenset({"leave", "sick_leave", "remote_work", "mission", "personal"})


@dataclass(frozen=True, slots=True)
class PersonnelRequestRecord:
    id: int
    personnel_id: int
    request_type: str
    start_date: str  # ISO date YYYY-MM-DD
    end_date: str  # ISO date YYYY-MM-DD
    reason: str | None
    status: str
    approved_by: int | None
    rejection_reason: str | None
    created_at_utc: str
    updated_at_utc: str


def _now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RequestStore:
    """SQLite-backed store for Personnel Requests with overlap validation."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path.resolve()
        self._lock = threading.RLock()
        self._init_db()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS personnel_requests (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    personnel_id     INTEGER NOT NULL,
                    request_type     TEXT    NOT NULL DEFAULT 'leave',
                    start_date       TEXT    NOT NULL,
                    end_date         TEXT    NOT NULL,
                    reason           TEXT,
                    status           TEXT    NOT NULL DEFAULT 'pending',
                    approved_by      INTEGER,
                    rejection_reason TEXT,
                    created_at_utc   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    updated_at_utc   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    FOREIGN KEY (personnel_id) REFERENCES personnel(id)
                );
                CREATE INDEX IF NOT EXISTS idx_requests_personnel ON personnel_requests(personnel_id);
                CREATE INDEX IF NOT EXISTS idx_requests_status ON personnel_requests(status);
                CREATE INDEX IF NOT EXISTS idx_requests_dates ON personnel_requests(start_date, end_date);
            """)

    @staticmethod
    def _row_to_request(row: sqlite3.Row) -> PersonnelRequestRecord:
        return PersonnelRequestRecord(
            id=row["id"],
            personnel_id=row["personnel_id"],
            request_type=row["request_type"],
            start_date=row["start_date"],
            end_date=row["end_date"],
            reason=row["reason"],
            status=row["status"],
            approved_by=row["approved_by"],
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
            # Jalali years are typically 1200-1500; Gregorian years outside that range
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

    def _check_overlap(
        self, conn: sqlite3.Connection, personnel_id: int, request_type: str,
        start_date: str, end_date: str, exclude_id: int | None = None,
    ) -> None:
        """Raise ValueError if an approved request of the same type overlaps."""
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
        reason: str | None = None,
    ) -> PersonnelRequestRecord:
        if request_type not in VALID_REQUEST_TYPES:
            raise ValueError(f"Invalid request type: {request_type!r}")
        if not start_date or not end_date:
            raise ValueError("start_date and end_date are required")
        s = self._normalize_date(start_date)
        e = self._normalize_date(end_date)
        if s > e:
            raise ValueError("start_date must not be after end_date")
        now = _now()
        with self._lock, self._connection() as conn:
            # Verify personnel exists
            p = conn.execute(
                "SELECT id FROM personnel WHERE id = ?", (personnel_id,)
            ).fetchone()
            if p is None:
                raise ValueError(f"Personnel not found: {personnel_id}")
            # Overlap check (only for approved, but we check existing pending-approved too)
            self._check_overlap(conn, personnel_id, request_type, s, e)
            cursor = conn.execute(
                "INSERT INTO personnel_requests (personnel_id, request_type, start_date, end_date, "
                "reason, status, created_at_utc, updated_at_utc) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
                (personnel_id, request_type, s, e, reason, now, now),
            )
            row = conn.execute(
                "SELECT * FROM personnel_requests WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            if row is None:
                raise RuntimeError("Failed to retrieve created request")
            return self._row_to_request(row)

    def get(self, request_id: int) -> PersonnelRequestRecord | None:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM personnel_requests WHERE id = ?", (request_id,)
            ).fetchone()
            return self._row_to_request(row) if row is not None else None

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
                # Reject instead
                new_status = "rejected"
            else:
                # Approve — check overlap before finalizing
                self._check_overlap(
                    conn, existing["personnel_id"], existing["request_type"],
                    existing["start_date"], existing["end_date"], exclude_id=request_id,
                )
                new_status = "approved"
            now = _now()
            conn.execute(
                "UPDATE personnel_requests SET status=?, approved_by=?, rejection_reason=?, "
                "updated_at_utc=? WHERE id=?",
                (new_status, approved_by, rejection_reason, now, request_id),
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
        """Cancel a request (only pending)."""
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
        where = ""
        if where_clauses:
            where = " WHERE " + " AND ".join(where_clauses)
        with self._lock, self._connection() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM personnel_requests{where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM personnel_requests{where} "
                "ORDER BY created_at_utc DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            return [self._row_to_request(r) for r in rows], int(total)

    def count(self) -> int:
        with self._lock, self._connection() as conn:
            return int(
                conn.execute("SELECT COUNT(*) FROM personnel_requests").fetchone()[0]
            )

    def get_approved_requests_in_range(
        self, personnel_id: int | None, start: date, end: date,
    ) -> list[PersonnelRequestRecord]:
        """Return approved requests overlapping [start, end].

        Uses standard interval overlap: start_date <= range_end AND end_date >= range_start.
        """
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
