from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import replace
from urllib.parse import urlsplit

import pytest

from app.database import Database, get_database, metadata
from app.core.source_registry import SourceRegistry
import app.config as app_config


def _test_database_url() -> str:
    url = os.getenv("TEST_DATABASE_URL", "").strip()
    if not url:
        pytest.skip("TEST_DATABASE_URL must point to a disposable PostgreSQL database")
    if not url.startswith(("postgresql://", "postgresql+psycopg2://")):
        pytest.fail("TEST_DATABASE_URL must use PostgreSQL with psycopg2")
    database_name = urlsplit(url.replace("postgresql+psycopg2://", "postgresql://", 1)).path.lstrip("/")
    if "test" not in database_name.lower():
        pytest.fail("TEST_DATABASE_URL database name must contain 'test'")
    configured_database = os.getenv("DATABASE_URL", "").strip()
    if configured_database and configured_database == url:
        pytest.fail("TEST_DATABASE_URL must not match DATABASE_URL")
    return url


def _truncate_application_tables(database: Database) -> None:
    table_names = ", ".join(f'"{name}"' for name in metadata.tables)
    with database.engine.begin() as connection:
        connection.exec_driver_sql(
            f"TRUNCATE TABLE {table_names} RESTART IDENTITY CASCADE"
        )


@pytest.fixture
def postgres_database() -> Iterator[Database]:
    database = get_database(_test_database_url())
    database.verify_connection()
    database.verify_schema()
    _truncate_application_tables(database)
    previous_database_url = os.environ.get("DATABASE_URL")
    previous_settings = app_config.settings
    os.environ["DATABASE_URL"] = database.url
    app_config.settings = replace(
        previous_settings,
        database_url=database.url,
        processor_mode="mock",
        video_ingestion_enabled=False,
    )
    try:
        yield database
    finally:
        app_config.settings = previous_settings
        if previous_database_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous_database_url
        _truncate_application_tables(database)


@pytest.fixture
def source_registry(postgres_database: Database) -> Iterator[SourceRegistry]:
    registry = SourceRegistry(postgres_database)
    try:
        yield registry
    finally:
        registry.close()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "postgresql: requires a disposable PostgreSQL database in TEST_DATABASE_URL",
    )
