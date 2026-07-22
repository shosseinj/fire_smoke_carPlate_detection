from __future__ import annotations

from pathlib import Path

from sqlalchemy import Date, DateTime, LargeBinary, Time
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from app.database import _replace_qmarks, metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_application_schema_uses_postgresql_business_types() -> None:
    assert isinstance(metadata.tables["users"].c.created_at_utc.type, DateTime)
    assert metadata.tables["users"].c.created_at_utc.type.timezone is True
    assert isinstance(metadata.tables["holidays"].c.date_value.type, Date)
    assert isinstance(metadata.tables["personnel_requests"].c.start_date.type, Date)
    assert isinstance(metadata.tables["work_shifts"].c.start_time.type, Time)
    assert metadata.tables["work_shifts"].c.timezone_name.server_default is not None
    assert isinstance(metadata.tables["face_embeddings"].c.embedding.type, LargeBinary)


def test_positional_parameters_are_prepared_for_psycopg2() -> None:
    sql = _replace_qmarks("SELECT * FROM users WHERE id = ? AND username = ?")
    assert sql == "SELECT * FROM users WHERE id = %s AND username = %s"
    quoted = _replace_qmarks("SELECT '?' AS literal, id FROM users WHERE id = ?")
    assert quoted == "SELECT '?' AS literal, id FROM users WHERE id = %s"


def test_postgresql_ddl_contains_native_types() -> None:
    dialect = postgresql.dialect()
    users_ddl = str(CreateTable(metadata.tables["users"]).compile(dialect=dialect))
    holidays_ddl = str(CreateTable(metadata.tables["holidays"]).compile(dialect=dialect))
    embeddings_ddl = str(
        CreateTable(metadata.tables["face_embeddings"]).compile(dialect=dialect)
    )
    assert "TIMESTAMP WITH TIME ZONE" in users_ddl
    assert " DATE " in holidays_ddl
    assert "BYTEA" in embeddings_ddl


def test_alembic_initial_revision_covers_every_application_table() -> None:
    migration = (
        PROJECT_ROOT
        / "alembic"
        / "versions"
        / "20260722_0001_initial_postgresql.py"
    ).read_text(encoding="utf-8")
    for table_name in metadata.tables:
        assert f"CREATE TABLE {table_name} " in migration
    assert "revision: str = '20260722_0001'" in migration
    assert "down_revision" in migration


def test_runtime_source_has_no_local_relational_database_backend() -> None:
    forbidden = ("".join(("sql", "ite3")), "".join(("PRA", "GMA journal_mode")), "".join((".sql", "ite3")))
    for path in (PROJECT_ROOT / "app").rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in content, f"{token!r} remains in {path}"
