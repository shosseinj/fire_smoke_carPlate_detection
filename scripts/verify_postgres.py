from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import inspect, text

from app.config import settings
from app.database import get_database, metadata


def main() -> None:
    database = get_database(
        settings.database_url,
        echo=settings.database_echo,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
    )
    database.verify_connection()
    database.verify_schema()
    with database.engine.connect() as connection:
        timezone_name = connection.execute(text("SHOW timezone")).scalar_one()
        revision = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()
    tables = sorted(inspect(database.engine).get_table_names())
    expected = sorted(metadata.tables)
    print("PostgreSQL connection: OK")
    print(f"Session timezone: {timezone_name}")
    print(f"Alembic revision: {revision}")
    print(f"Application tables ({len(expected)}): {', '.join(expected)}")
    extra = sorted(set(tables) - set(expected) - {"alembic_version"})
    if extra:
        print(f"Additional tables: {', '.join(extra)}")


if __name__ == "__main__":
    main()
