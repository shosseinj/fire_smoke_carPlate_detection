from __future__ import annotations

import ast
import re
from pathlib import Path

from sqlalchemy import Date, DateTime, LargeBinary, Time
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from app.database import ALEMBIC_HEAD_REVISION, _replace_qmarks, metadata


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


def test_alembic_migration_chain_covers_every_application_table() -> None:
    migration = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((PROJECT_ROOT / "alembic" / "versions").glob("*.py"))
    )
    for table_name in metadata.tables:
        assert (
            f"CREATE TABLE {table_name} " in migration
            or f"CREATE TABLE IF NOT EXISTS {table_name} " in migration
            or re.search(
                rf"op\.create_table\(\s*['\"]{re.escape(table_name)}['\"]",
                migration,
            )
        )
    assert "revision: str = '20260722_0001'" in migration
    assert "down_revision" in migration


def test_application_expected_revision_matches_alembic_head() -> None:
    revisions: set[str] = set()
    parent_revisions: set[str] = set()
    for path in (PROJECT_ROOT / "alembic" / "versions").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        values: dict[str, str | None] = {}
        for node in tree.body:
            target = None
            value = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                target = node.target
                value = node.value
            if (
                isinstance(target, ast.Name)
                and target.id in {"revision", "down_revision"}
                and isinstance(value, ast.Constant)
                and (isinstance(value.value, str) or value.value is None)
            ):
                values[target.id] = value.value
        revision = values.get("revision")
        if revision is not None:
            revisions.add(revision)
        parent = values.get("down_revision")
        if parent is not None:
            parent_revisions.add(parent)

    assert revisions - parent_revisions == {ALEMBIC_HEAD_REVISION}


def test_runtime_source_has_no_local_relational_database_backend() -> None:
    forbidden = ("".join(("sql", "ite3")), "".join(("PRA", "GMA journal_mode")), "".join((".sql", "ite3")))
    for path in (PROJECT_ROOT / "app").rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in content, f"{token!r} remains in {path}"

import pytest

pytestmark = pytest.mark.unit
