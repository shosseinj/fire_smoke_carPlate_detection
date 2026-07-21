from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth import router
from app.core import auth as auth_core
from app.core.auth_store import AuthStore


APPROVED_LEGACY_AUTH_ROUTES = [
    ("POST", "/api/v1/auth/token"),
    ("POST", "/api/v1/auth/login"),
    ("POST", "/api/v1/auth/refresh"),
    ("POST", "/api/v1/auth/logout"),
    ("POST", "/api/v1/auth/create-admin"),
    ("POST", "/api/v1/auth/create-user"),
    ("GET", "/api/v1/auth/me"),
    ("PUT", "/api/v1/auth/me/password"),
    ("GET", "/api/v1/auth/users"),
    ("PUT", "/api/v1/auth/users/{user_id}/role"),
    ("DELETE", "/api/v1/auth/users/{user_id}"),
]


@pytest.fixture()
def auth_context(tmp_path: Path):
    store = AuthStore(tmp_path / "auth.sqlite3")
    store.seed_default_admin(
        "admin",
        "StrongPass1!",
        email="admin@example.com",
        full_name="Primary Admin",
    )
    old_store = auth_core._auth_store
    auth_core._auth_store = store
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    try:
        yield client, store, app
    finally:
        client.close()
        auth_core._auth_store = old_store


def login(client: TestClient, username: str = "admin", password: str = "StrongPass1!") -> dict:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200, response.text
    return response.json()


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_route_coverage_openapi_and_no_duplicates(auth_context):
    _, _, app = auth_context
    actual: set[tuple[str, str]] = set()
    duplicates: list[tuple[str, str]] = []
    for route in app.routes:
        path = getattr(route, "path", "").rstrip("/") or "/"
        for method in getattr(route, "methods", set()) or set():
            pair = (method, path)
            if pair in actual:
                duplicates.append(pair)
            actual.add(pair)

    assert not duplicates
    assert set(APPROVED_LEGACY_AUTH_ROUTES) <= actual
    assert ("POST", "/api/v1/auth/register") not in actual

    schema = app.openapi()
    for method, path in APPROVED_LEGACY_AUTH_ROUTES:
        assert path in schema["paths"]
        assert method.lower() in schema["paths"][path]
    assert "/api/v1/auth/register" not in schema["paths"]


def test_sqlite_additive_schema_upgrade(tmp_path: Path):
    db_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'viewer',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at_utc TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO users (username, password_hash, role, created_at_utc) "
            "VALUES ('legacy', 'hash', 'user', '2026-01-01T00:00:00Z')"
        )
    store = AuthStore(db_path)
    user = store.get_user_by_username("legacy")
    assert user is not None
    assert user.role == "viewer"
    assert user.login_attempts == 0
    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
    assert {"email", "full_name", "last_login_utc", "login_attempts", "locked_until_utc"} <= columns


def test_login_superset_tracking_and_lockout(auth_context):
    client, store, _ = auth_context
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "StrongPass1!"},
    )
    assert response.status_code == 200
    body = response.json()
    assert {
        "access_token",
        "refresh_token",
        "token_type",
        "expires_in",
        "user_id",
        "username",
        "role",
    } <= body.keys()
    assert body["role"] == "admin"
    user = store.get_user_by_username("admin")
    assert user is not None and user.last_login_utc is not None
    assert user.login_attempts == 0 and user.locked_until_utc is None

    for _ in range(5):
        failed = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "wrong"},
        )
        assert failed.status_code == 401
    user = store.get_user_by_username("admin")
    assert user is not None and user.login_attempts == 5
    assert user.locked_until_utc is not None
    locked = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "StrongPass1!"},
    )
    assert locked.status_code == 401
    assert "قفل" in locked.json()["detail"]


def test_oauth_token_does_not_mutate_login_tracking(auth_context):
    client, store, _ = auth_context
    before = store.get_user_by_username("admin")
    assert before is not None and before.last_login_utc is None

    failed = client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "wrong"},
    )
    assert failed.status_code == 401
    assert failed.headers["www-authenticate"] == "Bearer"
    after_failed = store.get_user_by_username("admin")
    assert after_failed is not None and after_failed.login_attempts == 0

    success = client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "StrongPass1!"},
    )
    assert success.status_code == 200
    assert success.json()["refresh_token"]
    after_success = store.get_user_by_username("admin")
    assert after_success is not None and after_success.last_login_utc is None


def test_refresh_body_query_rotation_and_conflict(auth_context):
    client, _, _ = auth_context
    first = login(client)
    refreshed = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": first["refresh_token"]},
    )
    assert refreshed.status_code == 200
    second = refreshed.json()
    assert second["refresh_token"] != first["refresh_token"]
    assert second["user_id"] == first["user_id"]

    reuse = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": first["refresh_token"]},
    )
    assert reuse.status_code == 401

    query_refresh = client.post(
        "/api/v1/auth/refresh",
        params={"refresh_token": second["refresh_token"]},
    )
    assert query_refresh.status_code == 200
    conflict = client.post(
        "/api/v1/auth/refresh",
        params={"refresh_token": "query-token"},
        json={"refresh_token": "body-token"},
    )
    assert conflict.status_code == 400


def test_logout_current_and_legacy_modes(auth_context):
    client, _, _ = auth_context
    current = login(client)
    body_logout = client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": current["refresh_token"]},
    )
    assert body_logout.status_code == 200
    assert body_logout.json()["message"] == "Logged out successfully"
    repeated = client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": current["refresh_token"]},
    )
    assert repeated.status_code == 200

    legacy = login(client)
    unauthenticated = client.post(
        "/api/v1/auth/logout",
        params={"refresh_token": legacy["refresh_token"]},
    )
    assert unauthenticated.status_code == 401
    authenticated = client.post(
        "/api/v1/auth/logout",
        params={"refresh_token": legacy["refresh_token"]},
        headers=bearer(legacy["access_token"]),
    )
    assert authenticated.status_code == 200
    assert authenticated.json()["message"] == "خروج با موفقیت انجام شد"


def test_legacy_and_current_user_creation_and_listing(auth_context):
    client, _, _ = auth_context
    tokens = login(client)
    headers = bearer(tokens["access_token"])

    current = client.post(
        "/api/v1/auth/create-user",
        headers=headers,
        json={
            "username": "operator1",
            "password": "Operator1!",
            "role": "operator",
        },
    )
    assert current.status_code == 201
    assert current.json()["role"] == "operator"
    assert current.json()["id"] == current.json()["user_id"]

    legacy = client.post(
        "/api/v1/auth/create-user",
        headers=headers,
        json={
            "username": "legacy_user",
            "email": "legacy@example.com",
            "password": "LegacyUser1!",
            "confirm_password": "LegacyUser1!",
            "full_name": "Legacy User",
            "role": "admin",
        },
    )
    assert legacy.status_code == 200
    assert legacy.json()["role"] == "viewer"
    assert legacy.json()["full_name"] == "Legacy User"

    legacy_list = client.get("/api/v1/auth/users", headers=headers)
    assert legacy_list.status_code == 200
    assert isinstance(legacy_list.json(), list)
    current_list = client.get("/api/v1/auth/users?offset=0", headers=headers)
    assert current_list.status_code == 200
    assert set(current_list.json()) == {"users", "total", "offset", "limit"}
    conflict = client.get("/api/v1/auth/users?offset=1&skip=2", headers=headers)
    assert conflict.status_code == 400


def test_legacy_role_password_and_delete_routes(auth_context):
    client, store, _ = auth_context
    admin = login(client)
    headers = bearer(admin["access_token"])

    created = client.post(
        "/api/v1/auth/create-user",
        headers=headers,
        json={"username": "target", "password": "TargetPass1!", "role": "viewer"},
    )
    assert created.status_code == 201
    target_id = created.json()["id"]

    role = client.put(
        f"/api/v1/auth/users/{target_id}/role",
        params={"role": "admin"},
        headers=headers,
    )
    assert role.status_code == 200
    assert role.json()["role"] == "admin"

    role_back = client.put(
        f"/api/v1/auth/users/{target_id}/role",
        params={"role": "user"},
        headers=headers,
    )
    assert role_back.status_code == 200
    assert role_back.json()["role"] == "viewer"

    changed = client.put(
        "/api/v1/auth/me/password",
        headers=headers,
        json={
            "old_password": "StrongPass1!",
            "new_password": "NewStrong2!",
            "confirm_new_password": "NewStrong2!",
        },
    )
    assert changed.status_code == 200
    assert changed.json()["message"] == "رمز عبور با موفقیت تغییر کرد"
    assert client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "NewStrong2!"},
    ).status_code == 200

    deleted = client.delete(f"/api/v1/auth/users/{target_id}", headers=headers)
    assert deleted.status_code == 200
    assert store.get_user_by_id(target_id) is None
    self_delete = client.delete("/api/v1/auth/users/1", headers=headers)
    assert self_delete.status_code == 400


def test_legacy_jwt_claims_and_refresh_without_jti(auth_context):
    from datetime import datetime, timedelta, timezone

    import jwt

    from app.config import settings

    client, _, _ = auth_context
    now = datetime.now(timezone.utc)
    access = jwt.encode(
        {
            "sub": "admin",
            "user_id": 1,
            "role": "superuser",
            "type": "access",
            "iat": now,
            "exp": now + timedelta(minutes=5),
        },
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    me = client.get("/api/v1/auth/me", headers=bearer(access))
    assert me.status_code == 200
    assert me.json()["role"] == "admin"

    legacy_refresh = jwt.encode(
        {
            "sub": "admin",
            "user_id": 1,
            "role": "superuser",
            "type": "refresh",
            "iat": now,
            "exp": now + timedelta(minutes=5),
        },
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    refreshed = client.post(
        "/api/v1/auth/refresh", params={"refresh_token": legacy_refresh}
    )
    assert refreshed.status_code == 200
    assert refreshed.json()["role"] == "admin"
    reused = client.post(
        "/api/v1/auth/refresh", params={"refresh_token": legacy_refresh}
    )
    assert reused.status_code == 401


def test_permissions_and_last_active_admin_protection(auth_context):
    client, _, _ = auth_context
    admin = login(client)
    admin_headers = bearer(admin["access_token"])
    viewer = client.post(
        "/api/v1/auth/create-user",
        headers=admin_headers,
        json={"username": "viewer2", "password": "ViewerPass2!", "role": "viewer"},
    )
    viewer_login = login(client, "viewer2", "ViewerPass2!")
    forbidden = client.post(
        "/api/v1/auth/create-user",
        headers=bearer(viewer_login["access_token"]),
        json={"username": "not_allowed", "password": "NoAccess1!", "role": "viewer"},
    )
    assert forbidden.status_code == 403

    demote_last_admin = client.patch(
        "/api/v1/auth/users/1/role",
        headers=admin_headers,
        json={"role": "viewer"},
    )
    assert demote_last_admin.status_code == 400
    deactivate_last_admin = client.put(
        "/api/v1/auth/users/1",
        headers=admin_headers,
        json={"is_active": False},
    )
    assert deactivate_last_admin.status_code == 400
