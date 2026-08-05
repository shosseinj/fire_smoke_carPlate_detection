from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.database import Database, Row, ensure_database
from app.time_utils import utc_now_text


IMPORT_COLUMNS = (
    "id, import_type, source_filename, total_rows, imported_rows, "
    "skipped_rows, failed_rows, status, error_message, result_json, created_by, "
    "created_at_utc, updated_at_utc"
)


@dataclass(frozen=True, slots=True)
class ImportProgressRecord:
    id: int
    import_type: str
    source_filename: str
    total_rows: int
    imported_rows: int
    skipped_rows: int
    failed_rows: int
    status: str
    error_message: str | None
    result: dict[str, Any] | None
    created_by: int | None
    created_at_utc: str
    updated_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        terminal = self.status in ("completed", "completed_with_errors", "failed")
        return {
            "id": self.id,
            "import_type": self.import_type,
            "source_filename": self.source_filename,
            "total_rows": self.total_rows,
            "imported_rows": self.imported_rows,
            "skipped_rows": self.skipped_rows,
            "failed_rows": self.failed_rows,
            "status": self.status,
            "error_message": self.error_message,
            "result": self.result,
            "processed_rows": self.imported_rows + self.skipped_rows + self.failed_rows,
            "progress_percent": (
                100.0
                if terminal
                else round(min(100.0, (self.imported_rows + self.skipped_rows + self.failed_rows) * 100.0 / self.total_rows), 1)
                if self.total_rows > 0
                else 0.0
            ),
            "created_by": self.created_by,
            "created_at": self.created_at_utc,
            "updated_at": self.updated_at_utc,
        }


class ImportProgressStore:
    def __init__(self, database: Database | str) -> None:
        self.database = ensure_database(database)

    def _connection(self):
        return self.database.connection()

    def _row_to_record(self, row: Row) -> ImportProgressRecord:
        return ImportProgressRecord(
            id=int(row["id"]),
            import_type=str(row["import_type"]),
            source_filename=str(row["source_filename"]),
            total_rows=int(row["total_rows"]),
            imported_rows=int(row["imported_rows"]),
            skipped_rows=int(row["skipped_rows"]),
            failed_rows=int(row["failed_rows"]),
            status=str(row["status"]),
            error_message=str(row["error_message"]) if row["error_message"] is not None else None,
            result=json.loads(str(row["result_json"])) if row["result_json"] else None,
            created_by=int(row["created_by"]) if row["created_by"] is not None else None,
            created_at_utc=str(row["created_at_utc"]),
            updated_at_utc=str(row["updated_at_utc"]),
        )

    def create(
        self,
        import_type: str,
        source_filename: str,
        total_rows: int = 0,
        created_by: int | None = None,
        status: str = "running",
    ) -> ImportProgressRecord:
        now = utc_now_text()
        with self._connection() as conn:
            cursor = conn.execute(
                "INSERT INTO import_progress (import_type, source_filename, total_rows, imported_rows, skipped_rows, failed_rows, status, created_by, created_at_utc, updated_at_utc) VALUES (?, ?, ?, 0, 0, 0, ?, ?, ?, ?)",
                (import_type, source_filename, total_rows, status, created_by, now, now),
            )
            row = conn.execute(
                f"SELECT {IMPORT_COLUMNS} FROM import_progress WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
            if row is None:
                raise RuntimeError("Failed to create import progress")
            return self._row_to_record(row)

    def update(
        self,
        progress_id: int,
        *,
        imported_rows: int | None = None,
        total_rows: int | None = None,
        skipped_rows: int | None = None,
        failed_rows: int | None = None,
        status: str | None = None,
        error_message: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> ImportProgressRecord | None:
        sets: list[str] = []
        params: list[Any] = []
        if total_rows is not None:
            sets.append("total_rows = ?")
            params.append(total_rows)
        if imported_rows is not None:
            sets.append("imported_rows = ?")
            params.append(imported_rows)
        if skipped_rows is not None:
            sets.append("skipped_rows = ?")
            params.append(skipped_rows)
        if failed_rows is not None:
            sets.append("failed_rows = ?")
            params.append(failed_rows)
        if status is not None:
            sets.append("status = ?")
            params.append(status)
        if error_message is not None:
            sets.append("error_message = ?")
            params.append(error_message)
        if result is not None:
            sets.append("result_json = ?")
            params.append(json.dumps(result, ensure_ascii=False))
        if not sets:
            return self.get(progress_id)
        sets.append("updated_at_utc = ?")
        params.append(utc_now_text())
        params.append(progress_id)
        with self._connection() as conn:
            conn.execute(f"UPDATE import_progress SET {', '.join(sets)} WHERE id = ?", params)
            conn.commit()
        return self.get(progress_id)

    def get(self, progress_id: int) -> ImportProgressRecord | None:
        with self._connection() as conn:
            row = conn.execute(
                f"SELECT {IMPORT_COLUMNS} FROM import_progress WHERE id = ?",
                (progress_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_record(row)

    def list(
        self,
        *,
        import_type: str | None = None,
        status: str | None = None,
        created_by: int | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[ImportProgressRecord]:
        conditions: list[str] = []
        params: list[Any] = []
        if import_type is not None:
            conditions.append("import_type = ?")
            params.append(import_type)
        if status is not None:
            conditions.append("status = ?")
            params.append(status)
        if created_by is not None:
            conditions.append("created_by = ?")
            params.append(created_by)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT {IMPORT_COLUMNS} FROM import_progress{where} ORDER BY created_at_utc DESC LIMIT ? OFFSET ?"
        params.extend([limit, skip])
        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_record(row) for row in rows]

    def delete(self, progress_id: int) -> bool:
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM import_progress WHERE id = ?", (progress_id,))
            conn.commit()
            return cursor.rowcount > 0
