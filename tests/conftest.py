from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import replace
from urllib.parse import urlsplit, urlunsplit

import pytest

from app.database import Database, get_database, metadata
from app.core.source_registry import SourceRegistry
import app.config as app_config


TEST_DATABASE_NAME = "ai_database_test"


def _derive_test_database_url(configured_url: str) -> str:
    parsed = urlsplit(
        configured_url.replace("postgresql+psycopg2://", "postgresql://", 1)
    )
    if parsed.scheme != "postgresql" or not parsed.hostname:
        pytest.fail("DATABASE_URL must use PostgreSQL with psycopg2")
    derived = urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            f"/{TEST_DATABASE_NAME}",
            parsed.query,
            parsed.fragment,
        )
    )
    if configured_url.startswith("postgresql+psycopg2://"):
        return derived.replace("postgresql://", "postgresql+psycopg2://", 1)
    return derived


def _test_database_url() -> str:
    configured_database = (
        os.getenv("DATABASE_URL", "").strip()
        or app_config.settings.database_url
    )
    explicit_test_url = os.getenv("TEST_DATABASE_URL", "").strip()
    url = explicit_test_url or _derive_test_database_url(configured_database)
    if not url.startswith(("postgresql://", "postgresql+psycopg2://")):
        pytest.fail("TEST_DATABASE_URL must use PostgreSQL with psycopg2")
    database_name = urlsplit(
        url.replace("postgresql+psycopg2://", "postgresql://", 1)
    ).path.lstrip("/")
    if "test" not in database_name.lower():
        pytest.fail("TEST_DATABASE_URL database name must contain 'test'")
    if not explicit_test_url and database_name != TEST_DATABASE_NAME:
        pytest.fail(f"Derived test database must be named {TEST_DATABASE_NAME}")
    if configured_database == url:
        pytest.fail("TEST_DATABASE_URL must not match DATABASE_URL")
    return url


def _truncate_application_tables(database: Database) -> None:
    table_names = ", ".join(f'"{name}"' for name in metadata.tables)
    with database.engine.begin() as connection:
        connection.exec_driver_sql(
            f"TRUNCATE TABLE {table_names} RESTART IDENTITY CASCADE"
        )
        connection.exec_driver_sql(
            "INSERT INTO employee_types "
            "(id, name, is_active, include_in_attendance_reports) VALUES "
            "(1, 'پیمانکار', TRUE, FALSE), "
            "(2, 'مشتری', TRUE, FALSE), "
            "(3, 'مهمان', TRUE, FALSE), "
            "(4, 'کارمند', TRUE, TRUE), "
            "(5, 'نامشخص', TRUE, FALSE)"
        )
        connection.exec_driver_sql(
            "SELECT setval(pg_get_serial_sequence('employee_types', 'id'), 5, true)"
        )


@pytest.fixture
def postgres_database() -> Iterator[Database]:
    database = get_database(_test_database_url())
    database.verify_connection()
    database.verify_schema()
    _truncate_application_tables(database)
    previous_database_url = os.environ.get("DATABASE_URL")
    previous_test_database_url = os.environ.get("TEST_DATABASE_URL")
    previous_settings = app_config.settings
    os.environ["DATABASE_URL"] = database.url
    os.environ["TEST_DATABASE_URL"] = database.url
    app_config.settings = replace(
        previous_settings,
        database_url=database.url,
        processor_mode="mock",
        video_ingestion_enabled=False,
        media_preview_enabled=False,
    )
    try:
        yield database
    finally:
        app_config.settings = previous_settings
        if previous_database_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous_database_url
        if previous_test_database_url is None:
            os.environ.pop("TEST_DATABASE_URL", None)
        else:
            os.environ["TEST_DATABASE_URL"] = previous_test_database_url
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
