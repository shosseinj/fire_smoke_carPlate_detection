from __future__ import annotations

from types import SimpleNamespace

from sqlalchemy.exc import IntegrityError

from app.core.personnel_store import _personnel_integrity_message


def _integrity_error(constraint_name: str) -> IntegrityError:
    original = SimpleNamespace(
        diag=SimpleNamespace(constraint_name=constraint_name),
    )
    return IntegrityError("statement", {}, original)


def test_national_code_constraint_reports_duplicate() -> None:
    message = _personnel_integrity_message(
        _integrity_error("personnel_national_code_key"),
        national_code="1234567891",
        department_id=None,
    )

    assert message == "National code already exists: 1234567891"


def test_department_constraint_reports_missing_department() -> None:
    message = _personnel_integrity_message(
        _integrity_error("personnel_department_id_fkey"),
        national_code="1234567891",
        department_id=7,
    )

    assert message == "Department not found: 7"


def test_unknown_constraint_does_not_claim_duplicate() -> None:
    message = _personnel_integrity_message(
        _integrity_error("personnel_future_constraint"),
        national_code="1234567891",
        department_id=None,
    )

    assert message == "Personnel data violates a database constraint"
