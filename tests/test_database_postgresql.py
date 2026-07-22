from __future__ import annotations

from sqlalchemy import Date, DateTime, Time

from app.database import _translate_sql, metadata


def test_application_schema_uses_postgresql_business_types() -> None:
    assert isinstance(metadata.tables["users"].c.created_at_utc.type, DateTime)
    assert metadata.tables["users"].c.created_at_utc.type.timezone is True
    assert isinstance(metadata.tables["holidays"].c.date_value.type, Date)
    assert isinstance(metadata.tables["personnel_requests"].c.start_date.type, Date)
    assert isinstance(metadata.tables["work_shifts"].c.start_time.type, Time)
    assert metadata.tables["work_shifts"].c.timezone_name.server_default is not None


def test_sqlite_placeholders_and_idioms_are_translated() -> None:
    sql, _ = _translate_sql("SELECT * FROM users WHERE id = ?")
    assert sql == "SELECT * FROM users WHERE id = %s"

    sql, _ = _translate_sql(
        "INSERT OR IGNORE INTO revoked_tokens (jti, revoked_at_utc, expires_at_utc) "
        "VALUES (?, ?, ?)"
    )
    assert sql is not None
    assert "ON CONFLICT DO NOTHING" in sql
    assert sql.count("%s") == 3

    sql, _ = _translate_sql(
        "SELECT * FROM holidays WHERE substr(date_value, 6) = ?"
    )
    assert sql is not None
    assert "to_char(date_value, 'MM-DD')" in sql


def test_old_sqlite_schema_statements_are_noops() -> None:
    sql, _ = _translate_sql("PRAGMA journal_mode=WAL")
    assert sql is None
    sql, _ = _translate_sql("CREATE TABLE IF NOT EXISTS demo (id INTEGER)")
    assert sql is None
