from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.database import Database, ensure_database


class CarPlateStore:
    """Persistent registered-plate records for the legacy resource contract."""

    def __init__(self, database: Database | str) -> None:
        self.database = ensure_database(database)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def list(
        self,
        *,
        active_only: bool = True,
        search: str | None = None,
        usage_type: str | None = None,
        vehicle_type: str | None = None,
        owner_phone: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses = ["deleted_at_utc IS NULL"]
        params: list[Any] = []
        if active_only:
            clauses.append("is_active = 1")
        if search:
            clauses.append("(owner_name ILIKE ? OR owner_phone ILIKE ? OR brand ILIKE ? OR model ILIKE ?)")
            value = f"%{search.strip()}%"
            params.extend([value] * 4)
        for column, value in (("usage_type", usage_type), ("vehicle_type", vehicle_type), ("owner_phone", owner_phone)):
            if value:
                clauses.append(f"{column} = ?")
                params.append(value.strip())
        params.extend([max(0, skip), max(1, min(limit, 500))])
        with self.database.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM car_plates WHERE " + " AND ".join(clauses) + " ORDER BY id DESC OFFSET ? LIMIT ?",
                params,
            ).fetchall()
        return [self._serialize(dict(row)) for row in rows]

    def get(self, plate_id: int) -> dict[str, Any] | None:
        with self.database.connection() as connection:
            row = connection.execute("SELECT * FROM car_plates WHERE id = ? AND deleted_at_utc IS NULL", (plate_id,)).fetchone()
        return self._serialize(dict(row)) if row else None

    def create(self, values: dict[str, Any]) -> dict[str, Any]:
        now = self._now()
        fields = dict(values)
        fields.setdefault("plate_format", "standard")
        fields.setdefault("is_active", True)
        columns = list(fields) + ["created_at_utc", "updated_at_utc"]
        parameters = [int(fields[name]) if name == "is_active" else fields[name] for name in fields] + [now, now]
        with self.database.connection() as connection:
            row = connection.execute(
                f"INSERT INTO car_plates ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)}) RETURNING *",
                parameters,
            ).fetchone()
            if row is not None:
                connection.execute(
                    """
                    UPDATE plate_logs
                    SET plate_id = ?, updated_at = ?
                    WHERE plate_id IS NULL AND plate_number = ?
                    """,
                    (int(row["id"]), now, str(row["normalized_plate"])),
                )
        return self._serialize(dict(row)) if row is not None else {}

    def update(self, plate_id: int, values: dict[str, Any]) -> dict[str, Any] | None:
        if not values:
            return self.get(plate_id)
        assignments = [f"{name} = ?" for name in values] + ["updated_at_utc = ?"]
        params = [int(value) if name == "is_active" else value for name, value in values.items()]
        params.extend([self._now(), plate_id])
        with self.database.connection() as connection:
            row = connection.execute(
                f"UPDATE car_plates SET {', '.join(assignments)} "
                "WHERE id = ? AND deleted_at_utc IS NULL RETURNING *",
                params,
            ).fetchone()
        return self._serialize(dict(row)) if row is not None else None

    def delete(self, plate_id: int) -> bool:
        with self.database.connection() as connection:
            cursor = connection.execute(
                "UPDATE car_plates SET is_active = 0, deleted_at_utc = ?, updated_at_utc = ? WHERE id = ? AND deleted_at_utc IS NULL",
                (self._now(), self._now(), plate_id),
            )
            connection.commit()
            return cursor.rowcount > 0

    @staticmethod
    def _serialize(value: dict[str, Any]) -> dict[str, Any]:
        value["is_active"] = bool(value["is_active"])
        value["formatted_plate"] = f"{value['left_digits']} {value['plate_alphabet']} {value['right_digits']} ایران {value['iran_code']}"
        value["normalized_plate"] = f"{value['left_digits']}{value['plate_alphabet']}{value['right_digits']}{value['iran_code']}"
        return value
