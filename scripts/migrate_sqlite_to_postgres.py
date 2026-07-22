from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import sqlite3
from datetime import date, datetime, time, timezone
from typing import Any

from sqlalchemy import Date, DateTime, Time, delete, func, select, text

from app.database import get_database, metadata


def parse_value(column: Any, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(column.type, DateTime):
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    if isinstance(column.type, Date):
        return value if isinstance(value, date) else date.fromisoformat(str(value))
    if isinstance(column.type, Time):
        raw = str(value)
        if raw == "24:00":
            raw = "00:00"
        return value if isinstance(value, time) else time.fromisoformat(raw)
    return value


def sqlite_tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def load_rows(connection: sqlite3.Connection, table_name: str, target_table: Any) -> list[dict[str, Any]]:
    source_columns = {
        str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table_name}")')
    }
    columns = [column for column in target_table.columns if column.name in source_columns]
    if not columns:
        return []
    names = ", ".join(f'"{column.name}"' for column in columns)
    rows = connection.execute(f'SELECT {names} FROM "{table_name}"').fetchall()
    return [
        {column.name: parse_value(column, row[index]) for index, column in enumerate(columns)}
        for row in rows
    ]


def reset_sequence(connection: Any, table: Any) -> None:
    if "id" not in table.c or not table.c.id.primary_key:
        return
    sequence = connection.execute(
        text("SELECT pg_get_serial_sequence(:table_name, 'id')"),
        {"table_name": table.name},
    ).scalar_one_or_none()
    if not sequence:
        return
    max_id = connection.execute(select(func.max(table.c.id))).scalar_one_or_none()
    if max_id is None:
        connection.execute(text("SELECT setval(:sequence, 1, false)"), {"sequence": sequence})
    else:
        connection.execute(
            text("SELECT setval(:sequence, :value, true)"),
            {"sequence": sequence, "value": int(max_id)},
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy application tables from one or more legacy SQLite files into PostgreSQL."
    )
    parser.add_argument("sqlite_files", nargs="+", type=Path)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--truncate", action="store_true", help="Delete target rows before copying")
    args = parser.parse_args()

    database = get_database(args.database_url)
    copied: dict[str, int] = {}

    with database.engine.begin() as target:
        if args.truncate:
            for table in reversed(metadata.sorted_tables):
                target.execute(delete(table))

        for sqlite_path in args.sqlite_files:
            if not sqlite_path.is_file():
                raise FileNotFoundError(sqlite_path)
            with sqlite3.connect(sqlite_path) as source:
                available = sqlite_tables(source)
                for table in metadata.sorted_tables:
                    if table.name not in available:
                        continue
                    rows = load_rows(source, table.name, table)
                    if not rows:
                        continue
                    target.execute(table.insert(), rows)
                    copied[table.name] = copied.get(table.name, 0) + len(rows)

        for table in metadata.sorted_tables:
            reset_sequence(target, table)

    print("SQLite to PostgreSQL migration completed.")
    for table_name, count in sorted(copied.items()):
        print(f"  {table_name}: {count}")


if __name__ == "__main__":
    main()
