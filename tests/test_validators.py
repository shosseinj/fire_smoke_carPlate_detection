from __future__ import annotations

from app.core.personnel_store import normalize_national_code, validate_national_code


def test_normalize_national_code_keeps_only_exactly_ten_digits() -> None:
    assert normalize_national_code(" 008-457-5948 ") == "0084575948"
    assert normalize_national_code("123") == "123"
    assert normalize_national_code("123456789012") == "123456789012"


def test_validate_national_code_rejects_invalid_patterns() -> None:
    assert validate_national_code("1111111111") is False
    assert validate_national_code("1234567890") is False


def test_validate_national_code_accepts_known_valid_code() -> None:
    assert validate_national_code("0084575948") is True
import pytest

pytestmark = pytest.mark.unit
