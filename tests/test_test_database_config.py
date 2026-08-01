from __future__ import annotations

import pytest

import conftest as test_support


def test_explicit_test_database_url_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explicit_url = (
        "postgresql+psycopg2://test_user:test_password@test-db:5433/"
        "explicit_test?sslmode=require"
    )
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg2://app_user:app_password@main-db:5432/ai_database",
    )
    monkeypatch.setenv("TEST_DATABASE_URL", explicit_url)

    assert test_support._test_database_url() == explicit_url


@pytest.mark.parametrize(
    ("configured_url", "expected_url"),
    [
        (
            "postgresql+psycopg2://ai_user:db_password@db.internal:5432/"
            "ai_database?sslmode=require&application_name=tests",
            "postgresql+psycopg2://ai_user:db_password@db.internal:5432/"
            "ai_database_test?sslmode=require&application_name=tests",
        ),
        (
            "postgresql://ai_user:p%40ss@127.0.0.1:5434/ai_database",
            "postgresql://ai_user:p%40ss@127.0.0.1:5434/ai_database_test",
        ),
    ],
)
def test_absent_explicit_url_derives_named_test_database(
    monkeypatch: pytest.MonkeyPatch,
    configured_url: str,
    expected_url: str,
) -> None:
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", configured_url)

    assert test_support._test_database_url() == expected_url


@pytest.mark.parametrize(
    ("configured_url", "explicit_url", "message"),
    [
        (
            "postgresql://app:secret@db:5432/production_test",
            "postgresql://app:secret@db:5432/production_test",
            "must not match DATABASE_URL",
        ),
        (
            "postgresql://app:secret@db:5432/ai_database",
            "postgresql://app:secret@db:5432/ai_database",
            "database name must contain 'test'",
        ),
        (
            "postgresql://app:secret@db:5432/ai_database",
            "sqlite:///ai_database_test",
            "must use PostgreSQL",
        ),
    ],
)
def test_explicit_target_safety_rejects_production_or_suspicious_urls(
    monkeypatch: pytest.MonkeyPatch,
    configured_url: str,
    explicit_url: str,
    message: str,
) -> None:
    monkeypatch.setenv("DATABASE_URL", configured_url)
    monkeypatch.setenv("TEST_DATABASE_URL", explicit_url)

    with pytest.raises(pytest.fail.Exception, match=message):
        test_support._test_database_url()
