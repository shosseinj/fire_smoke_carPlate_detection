from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import inspect, text

from app.config import settings
from app.database import get_database


def main() -> None:
    database = get_database(
        settings.database_url,
        echo=settings.database_echo,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
    )
    with database.engine.connect() as connection:
        connection.execute(text("SELECT 1"))
        timezone_name = connection.execute(text("SHOW timezone")).scalar_one()
    tables = sorted(inspect(database.engine).get_table_names())
    print("PostgreSQL connection: OK")
    print(f"Session timezone: {timezone_name}")
    print(f"Application tables ({len(tables)}): {', '.join(tables)}")


if __name__ == "__main__":
    main()
