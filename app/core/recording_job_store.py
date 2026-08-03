from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.database import Database, IntegrityError, Row, ensure_database


ACTIVE_STATUSES = frozenset({"scheduled", "queued", "recording", "finalizing", "uploading"})
TERMINAL_STATUSES = frozenset({"completed", "partial", "failed", "cancelled"})
ALL_STATUSES = ACTIVE_STATUSES | TERMINAL_STATUSES
MAX_DURATION = timedelta(hours=2)
OBJECT_RETENTION = timedelta(days=30)
FAILED_SPOOL_RETENTION = timedelta(days=7)

_TRANSITIONS: dict[str, frozenset[str]] = {
    "scheduled": frozenset({"queued", "cancelled", "failed"}),
    "queued": frozenset({"recording", "cancelled", "failed"}),
    "recording": frozenset({"finalizing", "cancelled", "failed"}),
    "finalizing": frozenset({"uploading", "partial", "failed"}),
    "uploading": frozenset({"completed", "partial", "failed"}),
}


class RecordingJobError(Exception):
    pass


class RecordingOverlapError(RecordingJobError):
    pass


class InvalidRecordingTransition(RecordingJobError):
    pass


@dataclass(frozen=True, slots=True)
class RecordingJob:
    id: str
    source_uri: str
    created_by: int | None
    idempotency_key: str | None
    status: str
    scheduled_start_utc: datetime
    scheduled_end_utc: datetime
    started_at_utc: datetime | None = None
    finished_at_utc: datetime | None = None
    cancel_requested_at_utc: datetime | None = None
    object_key: str | None = None
    content_type: str | None = None
    size_bytes: int | None = None
    spool_path: str | None = None
    warning: str | None = None
    error: str | None = None
    is_partial: bool = False
    attempt_count: int = 0
    object_expires_at_utc: datetime | None = None
    spool_expires_at_utc: datetime | None = None
    created_at_utc: datetime | None = None
    updated_at_utc: datetime | None = None

    @property
    def has_content(self) -> bool:
        return self.status in {"completed", "partial"} and bool(self.object_key)


_COLUMNS = ", ".join(RecordingJob.__dataclass_fields__)


def _utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return value.astimezone(timezone.utc)


class RecordingJobStore:
    def __init__(self, database: Database | str) -> None:
        self.database = ensure_database(database)

    def create(
        self,
        *,
        source_uri: str,
        scheduled_start_utc: datetime,
        scheduled_end_utc: datetime,
        created_by: int | None,
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> tuple[RecordingJob, bool]:
        start = _utc(scheduled_start_utc, "scheduled_start_utc")
        end = _utc(scheduled_end_utc, "scheduled_end_utc")
        current = _utc(now or datetime.now(timezone.utc), "now")
        if end <= start:
            raise ValueError("scheduled_end_utc must be after scheduled_start_utc")
        if end - start > MAX_DURATION:
            raise ValueError("recording duration cannot exceed 2 hours")
        if end <= current:
            raise ValueError("scheduled_end_utc must be in the future")
        key = idempotency_key.strip() if idempotency_key else None
        if key and len(key) > 255:
            raise ValueError("idempotency_key cannot exceed 255 characters")
        if key:
            existing = self.get_by_idempotency(created_by, key)
            if existing is not None:
                if (existing.source_uri, existing.scheduled_start_utc, existing.scheduled_end_utc) != (source_uri, start, end):
                    raise ValueError("idempotency key was already used for a different request")
                return existing, False
        job_id = str(uuid.uuid4())
        try:
            with self.database.connection() as connection:
                source = connection.execute(
                    "SELECT source_uri FROM sources WHERE source_uri = ? AND source_uri <> '__default__'",
                    (source_uri,),
                ).fetchone()
                if source is None:
                    raise ValueError("recording source does not exist")
                overlap = connection.execute(
                    "SELECT id FROM recording_jobs WHERE source_uri = ? "
                    "AND status IN ('scheduled','queued','recording','finalizing','uploading') "
                    "AND scheduled_start_utc < ? AND scheduled_end_utc > ? LIMIT 1",
                    (source_uri, end, start),
                ).fetchone()
                if overlap:
                    raise RecordingOverlapError("recording overlaps an active job for this source")
                connection.execute(
                    "INSERT INTO recording_jobs (id, source_uri, created_by, idempotency_key, status, "
                    "scheduled_start_utc, scheduled_end_utc, created_at_utc, updated_at_utc) "
                    "VALUES (?, ?, ?, ?, 'scheduled', ?, ?, ?, ?)",
                    (job_id, source_uri, created_by, key, start, end, current, current),
                )
        except IntegrityError as exc:
            if key:
                existing = self.get_by_idempotency(created_by, key)
                if existing is not None:
                    return existing, False
            raise RecordingOverlapError("recording overlaps an active job for this source") from exc
        job = self.require(job_id)
        return job, True

    def get(self, job_id: str) -> RecordingJob | None:
        with self.database.connection() as connection:
            row = connection.execute(f"SELECT {_COLUMNS} FROM recording_jobs WHERE id = ?", (job_id,)).fetchone()
        return self._record(row) if row else None

    def require(self, job_id: str) -> RecordingJob:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def get_by_idempotency(self, created_by: int | None, key: str) -> RecordingJob | None:
        with self.database.connection() as connection:
            row = connection.execute(
                f"SELECT {_COLUMNS} FROM recording_jobs WHERE created_by IS NOT DISTINCT FROM ? AND idempotency_key = ?",
                (created_by, key),
            ).fetchone()
        return self._record(row) if row else None

    def list(self, *, status: str | None = None, source_uri: str | None = None, limit: int = 100, offset: int = 0) -> list[RecordingJob]:
        if status is not None and status not in ALL_STATUSES:
            raise ValueError("invalid recording status")
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if source_uri:
            clauses.append("source_uri = ?")
            params.append(source_uri)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend((max(1, min(limit, 500)), max(0, offset)))
        with self.database.connection() as connection:
            rows = connection.execute(
                f"SELECT {_COLUMNS} FROM recording_jobs{where} ORDER BY scheduled_start_utc DESC LIMIT ? OFFSET ?",
                params,
            ).fetchall()
        return [self._record(row) for row in rows]

    def transition(self, job_id: str, target: str, *, warning: str | None = None, error: str | None = None, now: datetime | None = None, **fields: Any) -> RecordingJob:
        current = self.require(job_id)
        if target not in _TRANSITIONS.get(current.status, frozenset()):
            raise InvalidRecordingTransition(f"cannot transition {current.status} to {target}")
        instant = _utc(now or datetime.now(timezone.utc), "now")
        allowed = {"object_key", "content_type", "size_bytes", "spool_path", "is_partial"}
        if set(fields) - allowed:
            raise ValueError("unsupported recording update field")
        updates: dict[str, Any] = {"status": target, "updated_at_utc": instant}
        updates.update(fields)
        if warning is not None:
            updates["warning"] = warning[:2000]
        if error is not None:
            updates["error"] = error[:2000]
        if target == "recording":
            updates["started_at_utc"] = instant
            updates["attempt_count"] = current.attempt_count + 1
        if target in TERMINAL_STATUSES:
            updates["finished_at_utc"] = instant
        if target in {"completed", "partial"}:
            updates["object_expires_at_utc"] = instant + OBJECT_RETENTION
        if target == "failed" and (fields.get("spool_path") or current.spool_path):
            updates["spool_expires_at_utc"] = instant + FAILED_SPOOL_RETENTION
        sets = ", ".join(f"{name} = ?" for name in updates)
        with self.database.connection() as connection:
            cursor = connection.execute(
                f"UPDATE recording_jobs SET {sets} WHERE id = ? AND status = ?",
                (*updates.values(), job_id, current.status),
            )
            if cursor.rowcount != 1:
                raise InvalidRecordingTransition("recording state changed concurrently")
        return self.require(job_id)

    def cancel(self, job_id: str, *, now: datetime | None = None) -> RecordingJob:
        current = self.require(job_id)
        if current.status in TERMINAL_STATUSES:
            return current
        instant = _utc(now or datetime.now(timezone.utc), "now")
        terminal_now = current.status in {"scheduled", "queued"}
        with self.database.connection() as connection:
            if terminal_now:
                connection.execute(
                    "UPDATE recording_jobs SET status = 'cancelled', cancel_requested_at_utc = ?, "
                    "finished_at_utc = ?, updated_at_utc = ? WHERE id = ? AND status = ?",
                    (instant, instant, instant, job_id, current.status),
                )
            else:
                connection.execute(
                    "UPDATE recording_jobs SET cancel_requested_at_utc = ?, is_partial = TRUE, "
                    "updated_at_utc = ? WHERE id = ? AND status = ?",
                    (instant, instant, job_id, current.status),
                )
        return self.require(job_id)

    def recover_spool(self, job_id: str, spool_path: str, *, now: datetime | None = None) -> RecordingJob:
        current = self.require(job_id)
        instant = _utc(now or datetime.now(timezone.utc), "now")
        if current.status == "recording":
            return self.transition(
                job_id, "finalizing", spool_path=spool_path, is_partial=True,
                warning="خروجی قابل پخش پس از راه‌اندازی مجدد بازیابی شد", now=instant,
            )
        if current.status in {"finalizing", "uploading"}:
            with self.database.connection() as connection:
                connection.execute(
                    "UPDATE recording_jobs SET spool_path = ?, is_partial = TRUE, updated_at_utc = ? "
                    "WHERE id = ? AND status = ?",
                    (spool_path, instant, job_id, current.status),
                )
            return self.require(job_id)
        return current

    def delete(self, job_id: str) -> bool:
        job = self.require(job_id)
        if job.status not in TERMINAL_STATUSES:
            raise InvalidRecordingTransition("active recording jobs cannot be deleted")
        with self.database.connection() as connection:
            cursor = connection.execute("DELETE FROM recording_jobs WHERE id = ?", (job_id,))
        return cursor.rowcount == 1

    def recoverable(self) -> list[RecordingJob]:
        with self.database.connection() as connection:
            rows = connection.execute(
                f"SELECT {_COLUMNS} FROM recording_jobs WHERE status IN "
                "('scheduled','queued','recording','finalizing','uploading') ORDER BY scheduled_start_utc"
            ).fetchall()
        return [self._record(row) for row in rows]

    @staticmethod
    def _record(row: Row) -> RecordingJob:
        values = {name: row.get(name) for name in RecordingJob.__dataclass_fields__}
        for name in ("scheduled_start_utc", "scheduled_end_utc", "started_at_utc", "finished_at_utc", "cancel_requested_at_utc", "object_expires_at_utc", "spool_expires_at_utc", "created_at_utc", "updated_at_utc"):
            value = values[name]
            if isinstance(value, datetime):
                values[name] = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
        values["id"] = str(values["id"])
        values["attempt_count"] = int(values["attempt_count"] or 0)
        values["is_partial"] = bool(values["is_partial"])
        values["size_bytes"] = int(values["size_bytes"]) if values["size_bytes"] is not None else None
        return RecordingJob(**values)
