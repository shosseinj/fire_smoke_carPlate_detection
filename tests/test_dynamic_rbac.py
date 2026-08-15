from __future__ import annotations

import pytest
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

import app.core.auth as auth
from app.api.auth import router as auth_router
from app.api.auth_schemas import CreateUserRequest, UserPermissionGrant, WebSocketTicketRequest
from app.core.access_matrix import ACCESS_REGISTRY
from app.config import settings
from app.core.auth_store import UserRecord


class _FakeAuthStore:
    def __init__(self, permissions: set[str]) -> None:
        self.permissions = frozenset(permissions)

    def get_effective_permissions(self, user_id: int) -> frozenset[str]:
        del user_id
        return self.permissions

    def has_scoped_permission(self, user_id: int, permission: str, scope_type: str, scope_id: int) -> bool:
        del user_id
        return (permission, scope_type, scope_id) == ("personnel.create", "section", 7)

    def accessible_scope_ids(self, user_id: int, permission: str, target_type: str) -> set[int]:
        del user_id, permission, target_type
        return {7, 8}


def _user(username: str = "matrix-user") -> UserRecord:
    return UserRecord(id=42, username=username, password_hash="", is_active=True, created_at_utc="2026-08-11T00:00:00Z")


def test_direct_permission_satisfies_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "_auth_store", _FakeAuthStore({"auth.manage"}))
    assert auth.require_permission("auth.manage")(_user()).id == 42


def test_missing_permission_returns_persian_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "_auth_store", _FakeAuthStore(set()))
    with pytest.raises(HTTPException) as error:
        auth.require_permission("auth.manage")(_user())
    assert error.value.status_code == 403
    assert "مجوز" in error.value.detail


def test_protected_admin_has_implicit_wildcard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "_auth_store", _FakeAuthStore(set()))
    assert auth.effective_permissions(_user(settings.auth_default_admin_username)) == frozenset({"*"})


def test_personnel_scope_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "_auth_store", _FakeAuthStore(set()))
    assert auth.has_scoped_permission(_user(), "personnel.create", "section", 7)
    assert not auth.has_scoped_permission(_user(), "personnel.create", "section", 9)


def test_grant_scope_identifier_validation() -> None:
    assert UserPermissionGrant(application="personnel", action="create", scope_type="section", scope_id=3).scope_id == 3
    with pytest.raises(ValidationError):
        UserPermissionGrant(application="personnel", action="create", scope_type="section", scope_id=0)


def test_matrix_routes_replace_rbac_routes() -> None:
    app = FastAPI()
    app.include_router(auth_router)
    paths = app.openapi()["paths"]
    assert "/api/v1/auth/access/definitions" in paths
    assert {"get", "put"}.issubset(paths["/api/v1/auth/users/{user_id}/access"])
    assert not any("/rbac/" in path or path.endswith("/roles") for path in paths)


def test_user_creation_does_not_require_email() -> None:
    payload = CreateUserRequest(
        username="without-email",
        password="StrongPass1!",
        confirm_password="StrongPass1!",
    )
    assert payload.email is None


def test_live_channels_are_matrix_permissions() -> None:
    assert ("broadcast", "read") in ACCESS_REGISTRY
    assert ("video_wall", "read") in ACCESS_REGISTRY
    assert ("results", "read") in ACCESS_REGISTRY


def test_results_websocket_ticket_accepts_resource_scope() -> None:
    ticket = WebSocketTicketRequest(application="results", scope_type="room", scope_id=1)
    assert ticket.scope_type == "room"


def test_broad_application_permission_is_removed() -> None:
    assert not any(application == "application" for application, _ in ACCESS_REGISTRY)
    assert ("rooms", "assign") in ACCESS_REGISTRY
    assert ACCESS_REGISTRY[("rooms", "read")].scope_types[-2:] == ("camera", "room")


def test_dashboard_requests_and_attaches_websocket_tickets() -> None:
    dashboard = (Path(__file__).parents[1] / "app" / "web" / "dashboard.html").read_text(encoding="utf-8")
    assert 'requestWebSocketTicket("results")' in dashboard
    assert 'requestWebSocketTicket(application)' in dashboard
    assert 'parameters.set("ticket", ticket)' in dashboard
