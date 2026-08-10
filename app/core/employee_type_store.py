from __future__ import annotations

import threading
from dataclasses import dataclass

from app.database import Connection, Database, IntegrityError, Row, ensure_database
from app.time_utils import utc_now_text


@dataclass(frozen=True, slots=True)
class EmployeeTypeRecord:
    id: int
    name: str
    description: str | None
    is_active: bool
    include_in_attendance_reports: bool
    created_at_utc: str
    updated_at_utc: str
    created_by: int | None = None
    updated_by: int | None = None


class EmployeeTypeStore:
    """PostgreSQL-backed lookup store for personnel employee types."""

    def __init__(self, database: Database | str) -> None:
        self.database = ensure_database(database)
        self._lock = threading.RLock()

    def _connection(self) -> Connection:
        return self.database.connection()

    @staticmethod
    def _row_to_record(row: Row) -> EmployeeTypeRecord:
        return EmployeeTypeRecord(
            id=int(row["id"]),
            name=str(row["name"]),
            description=row["description"],
            is_active=bool(row["is_active"]),
            include_in_attendance_reports=bool(row["include_in_attendance_reports"]),
            created_at_utc=row["created_at_utc"],
            updated_at_utc=row["updated_at_utc"],
            created_by=row.get("created_by"),
            updated_by=row.get("updated_by"),
        )

    @staticmethod
    def _normalize_name(name: str) -> str:
        normalized = name.strip()
        if not normalized:
            raise ValueError("name is required")
        if len(normalized) > 200:
            raise ValueError("name must be at most 200 characters")
        return normalized

    def create(
        self,
        *,
        name: str,
        description: str | None = None,
        is_active: bool = True,
        include_in_attendance_reports: bool = False,
        created_by: int | None = None,
    ) -> EmployeeTypeRecord:
        name = self._normalize_name(name)
        now = utc_now_text()
        with self._lock, self._connection() as conn:
            try:
                row = conn.execute(
                    "INSERT INTO employee_types "
                    "(name, description, is_active, include_in_attendance_reports, "
                    "created_at_utc, updated_at_utc, created_by) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING *",
                    (
                        name,
                        description,
                        bool(is_active),
                        bool(include_in_attendance_reports),
                        now,
                        now,
                        created_by,
                    ),
                ).fetchone()
            except IntegrityError as exc:
                raise ValueError(f"Employee type name already exists: {name}") from exc
            if row is None:
                raise RuntimeError("Failed to retrieve created employee type")
            return self._row_to_record(row)

    def get(self, employee_type_id: int) -> EmployeeTypeRecord | None:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM employee_types WHERE id = ?", (employee_type_id,)
            ).fetchone()
            return self._row_to_record(row) if row is not None else None

    def get_by_name(self, name: str) -> EmployeeTypeRecord | None:
        normalized = self._normalize_name(name)
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM employee_types WHERE LOWER(name) = LOWER(?)", (normalized,)
            ).fetchone()
            return self._row_to_record(row) if row is not None else None

    def list(self, *, is_active: bool | None = None) -> list[EmployeeTypeRecord]:
        sql = "SELECT * FROM employee_types"
        params: list[object] = []
        if is_active is not None:
            sql += " WHERE is_active = ?"
            params.append(bool(is_active))
        sql += " ORDER BY id"
        with self._lock, self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_record(row) for row in rows]

    def update(
        self,
        employee_type_id: int,
        *,
        name: str | None = None,
        description: str | None = None,
        is_active: bool | None = None,
        include_in_attendance_reports: bool | None = None,
        updated_by: int | None = None,
    ) -> EmployeeTypeRecord | None:
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                "SELECT * FROM employee_types WHERE id = ?", (employee_type_id,)
            ).fetchone()
            if existing is None:
                return None
            new_name = self._normalize_name(name) if name is not None else str(existing["name"])
            new_description = description if description is not None else existing["description"]
            new_active = bool(is_active) if is_active is not None else bool(existing["is_active"])
            new_report_flag = (
                bool(include_in_attendance_reports)
                if include_in_attendance_reports is not None
                else bool(existing["include_in_attendance_reports"])
            )
            try:
                row = conn.execute(
                    "UPDATE employee_types SET name=?, description=?, is_active=?, "
                    "include_in_attendance_reports=?, updated_at_utc=?, updated_by=? "
                    "WHERE id=? RETURNING *",
                    (
                        new_name,
                        new_description,
                        new_active,
                        new_report_flag,
                        utc_now_text(),
                        updated_by,
                        employee_type_id,
                    ),
                ).fetchone()
            except IntegrityError as exc:
                raise ValueError(f"Employee type name already exists: {new_name}") from exc
            return self._row_to_record(row) if row is not None else None

    def personnel_count(self, employee_type_id: int) -> int:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM personnel WHERE employee_type_id = ?",
                (employee_type_id,),
            ).fetchone()
            return int(row[0]) if row is not None else 0

    def delete(self, employee_type_id: int) -> bool:
        """Delete only when no personnel references this type.

        The database FK also uses ON DELETE RESTRICT as a second protection layer.
        """
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                "SELECT id FROM employee_types WHERE id = ?", (employee_type_id,)
            ).fetchone()
            if existing is None:
                return False
            count_row = conn.execute(
                "SELECT COUNT(*) FROM personnel WHERE employee_type_id = ?",
                (employee_type_id,),
            ).fetchone()
            assigned_count = int(count_row[0]) if count_row is not None else 0
            if assigned_count:
                raise ValueError(
                    f"امکان حذف نوع استخدام وجود ندارد؛ این نوع به {assigned_count} پرسنل اختصاص داده شده است"
                )
            try:
                conn.execute("DELETE FROM employee_types WHERE id = ?", (employee_type_id,))
            except IntegrityError as exc:
                raise ValueError(
                    "امکان حذف نوع استخدام وجود ندارد؛ این نوع به پرسنل اختصاص داده شده است"
                ) from exc
            return True
