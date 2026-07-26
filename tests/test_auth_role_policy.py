from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.api.auth import _require_creation_permission, role_information
from app.api.auth_schemas import CreateUserRequest
from app.core.auth import normalize_role, require_role
from app.core.auth_store import UserRecord


def _user(role: str) -> UserRecord:
    return UserRecord(
        id=1,
        username="actor",
        password_hash="",
        role=role,
        is_active=True,
        created_at_utc="2026-01-01T00:00:00Z",
    )


@pytest.mark.parametrize(
    ("actor", "target"),
    [("superadmin", "superadmin"), ("superadmin", "admin"), ("superadmin", "user"),
     ("admin", "admin"), ("admin", "user")],
)
def test_role_creation_matrix_allows_requested_roles(actor: str, target: str) -> None:
    assert _require_creation_permission(_user(actor), target) in {"superadmin", "admin", "user"}


@pytest.mark.parametrize(
    ("actor", "target"),
    [("admin", "superadmin"), ("viewer", "user"), ("operator", "admin"), ("user", "admin")],
)
def test_role_creation_matrix_rejects_forbidden_roles(actor: str, target: str) -> None:
    with pytest.raises(HTTPException) as error:
        _require_creation_permission(_user(actor), target)
    assert error.value.status_code == 403
    assert "اجازه" in str(error.value.detail)


def test_creation_schema_preserves_explicit_role_and_defaults_to_user() -> None:
    common = {
        "username": "new-user",
        "password": "StrongPass1!",
        "email": "new@example.com",
        "confirm_password": "StrongPass1!",
    }
    assert CreateUserRequest(**common).role == "user"
    assert CreateUserRequest(**common, role="superadmin").role == "superadmin"


def test_legacy_role_aliases_are_safe_and_distinct() -> None:
    assert normalize_role("superuser") == "superadmin"
    assert normalize_role("user") == "user"
    assert normalize_role("operator") == "user"
    assert normalize_role("superadmin") != normalize_role("admin")


def test_old_permission_levels_map_to_the_three_roles() -> None:
    require_role("superuser")(_user("superadmin"))
    require_role("admin")(_user("superadmin"))
    require_role("admin")(_user("admin"))
    require_role("operator")(_user("user"))
    with pytest.raises(HTTPException) as error:
        require_role("admin")(_user("user"))
    assert error.value.status_code == 403
    assert "دسترسی" in str(error.value.detail)
    with pytest.raises(ValueError):
        require_role("typo")


def test_role_information_is_persian_and_lists_creation_capabilities() -> None:
    response = role_information()
    assert "نقش‌های کاربری" in response.message
    roles = {item.role: item for item in response.roles}
    assert set(roles) == {"superadmin", "admin", "user"}
    assert "سوپرادمین" in roles["superadmin"].title
    assert "ایجاد" in roles["superadmin"].description
    assert "سوپرادمین" in roles["admin"].description
