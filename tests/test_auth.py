from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
import time

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.core.auth import (
    create_access_token,
    create_refresh_token,
    decode_access_token,
    decode_refresh_token,
    get_auth_store,
    hash_password,
)
from app.core.auth_store import AuthStore
from app.database import Database
from app.runtime import build_runtime


pytestmark = pytest.mark.usefixtures("postgres_database")


def _test_database_url() -> str:
    """Return the disposable PostgreSQL URL configured for tests."""
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL must point to a disposable PostgreSQL database")
    return url


def _make_test_runtime(tmp_path: Path):
    """Build a minimal mock runtime with a clean auth database."""
    test_settings = replace(
        settings,
        processor_mode="mock",
        database_url=_test_database_url(),

        video_ingestion_enabled=False,
        auth_default_admin_username="admin",
        auth_default_admin_password="admin123",
    )
    return build_runtime(test_settings)


def _setup_client(tmp_path: Path):
    """Build runtime, swap into app main, return TestClient."""
    import app.main as main_module

    test_runtime = _make_test_runtime(tmp_path)
    old_runtime = main_module.runtime
    main_module.runtime = test_runtime

    # Auth store already initialized by build_runtime via initialize_auth_store
    return test_runtime, old_runtime, TestClient(main_module.app)


def _teardown(test_runtime, old_runtime):
    """Restore runtime."""
    import app.main as main_module

    test_runtime.close()
    main_module.runtime = old_runtime


def _admin_token(client: TestClient) -> str:
    """Helper: login as admin and return access token."""
    resp = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
    return resp.json()["access_token"]


# ═══════════════════════════════════════════════════════════════════
# Step 1: Existing login/me tests (must remain passing)
# ═══════════════════════════════════════════════════════════════════


def test_login_valid_credentials(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"},
        )
        assert response.status_code == 200
        body = response.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"
        assert body["role"] == "admin"
        payload = decode_access_token(body["access_token"])
        assert payload is not None
        assert payload["username"] == "admin"
        assert payload["role"] == "admin"
    finally:
        _teardown(test_runtime, old_runtime)


def test_login_invalid_password(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "wrong-password"},
        )
        assert response.status_code == 401
        body = response.json()
        assert "detail" in body
    finally:
        _teardown(test_runtime, old_runtime)


def test_login_nonexistent_user(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "nonexistent", "password": "somepass"},
        )
        assert response.status_code == 401
    finally:
        _teardown(test_runtime, old_runtime)


def test_login_empty_username(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "", "password": "admin123"},
        )
        assert response.status_code == 422
    finally:
        _teardown(test_runtime, old_runtime)


def test_login_missing_fields(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        response = client.post(
            "/api/v1/auth/login",
            json={},
        )
        assert response.status_code == 422
    finally:
        _teardown(test_runtime, old_runtime)


def test_me_with_valid_token(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        login_resp = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"},
        )
        token = login_resp.json()["access_token"]
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["username"] == "admin"
        assert body["role"] == "admin"
        assert body["is_active"] is True
        assert "id" in body
        assert "created_at_utc" in body
    finally:
        _teardown(test_runtime, old_runtime)


def test_me_without_token(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401
    finally:
        _teardown(test_runtime, old_runtime)


def test_me_with_invalid_token(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer this-is-not-a-valid-token"},
        )
        assert response.status_code == 401
    finally:
        _teardown(test_runtime, old_runtime)


def test_me_with_expired_token(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = create_access_token(
            user_id=1,
            username="admin",
            role="admin",
            expires_minutes=-60,
        )
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 401
    finally:
        _teardown(test_runtime, old_runtime)


def test_me_with_wrong_scheme(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Basic admin:admin123"},
        )
        assert response.status_code == 401
    finally:
        _teardown(test_runtime, old_runtime)


def test_token_contains_user_info(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        login_resp = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"},
        )
        assert login_resp.status_code == 200
        token = login_resp.json()["access_token"]
        payload = decode_access_token(token)
        assert payload is not None
        assert payload["sub"] == "1"
        assert payload["username"] == "admin"
        assert payload["role"] == "admin"
        assert payload["type"] == "access"
        assert "iat" in payload
        assert "exp" in payload
        assert "jti" in payload
    finally:
        _teardown(test_runtime, old_runtime)


def test_login_response_structure(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"},
        )
        assert response.status_code == 200
        body = response.json()
        assert {
            "access_token",
            "refresh_token",
            "token_type",
            "role",
            "username",
            "user_id",
            "expires_in",
        } <= set(body)
        assert isinstance(body["access_token"], str)
        assert len(body["access_token"]) > 0
        assert body["token_type"] == "bearer"
    finally:
        _teardown(test_runtime, old_runtime)


def test_auth_store_seed_only_once(postgres_database: Database) -> None:
    store = AuthStore(postgres_database)
    seeded = store.seed_default_admin("admin", "admin123")
    assert seeded is not None
    assert store.count_users() == 1
    seeded2 = store.seed_default_admin("admin2", "pass2")
    assert seeded2 is None
    assert store.count_users() == 1


def test_auth_store_verify_credentials(postgres_database: Database) -> None:
    store = AuthStore(postgres_database)
    store.seed_default_admin("operator", "op123", role="operator")
    user = store.verify_credentials("operator", "op123")
    assert user is not None
    assert user.username == "operator"
    assert user.role == "operator"
    assert store.verify_credentials("operator", "wrong") is None
    assert store.verify_credentials("nobody", "op123") is None


# ═══════════════════════════════════════════════════════════════════
# Step 2: Token lifecycle tests
# ═══════════════════════════════════════════════════════════════════


def test_token_oauth2_endpoint(tmp_path: Path) -> None:
    """POST /api/v1/auth/token with form data returns both access and refresh tokens."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        response = client.post(
            "/api/v1/auth/token",
            data={"username": "admin", "password": "admin123"},
        )
        assert response.status_code == 200
        body = response.json()
        assert "access_token" in body
        assert "refresh_token" in body
        assert body["token_type"] == "bearer"
        assert body["role"] == "admin"
        assert "expires_in" in body
        assert body["expires_in"] > 0

        # Access token is valid
        payload = decode_access_token(body["access_token"])
        assert payload is not None
        assert payload["username"] == "admin"

        # Refresh token is valid
        refresh_payload = decode_refresh_token(body["refresh_token"])
        assert refresh_payload is not None
        assert refresh_payload["username"] == "admin"
        assert refresh_payload["type"] == "refresh"
    finally:
        _teardown(test_runtime, old_runtime)


def test_token_oauth2_invalid_credentials(tmp_path: Path) -> None:
    """POST /api/v1/auth/token with wrong password returns 401."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        response = client.post(
            "/api/v1/auth/token",
            data={"username": "admin", "password": "wrongpass"},
        )
        assert response.status_code == 401
    finally:
        _teardown(test_runtime, old_runtime)


def test_token_oauth2_missing_fields(tmp_path: Path) -> None:
    """POST /api/v1/auth/token with missing fields returns 422."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        response = client.post("/api/v1/auth/token", data={})
        assert response.status_code == 422
    finally:
        _teardown(test_runtime, old_runtime)


def test_refresh_valid_token(tmp_path: Path) -> None:
    """POST /api/v1/auth/refresh with valid refresh token returns new access token."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        # Get a refresh token from /token
        token_resp = client.post(
            "/api/v1/auth/token",
            data={"username": "admin", "password": "admin123"},
        )
        refresh_token = token_resp.json()["refresh_token"]

        # Use it to refresh
        response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        assert response.status_code == 200
        body = response.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"
        assert body["role"] == "admin"
        assert "expires_in" in body

        # New access token is valid
        payload = decode_access_token(body["access_token"])
        assert payload is not None
        assert payload["username"] == "admin"
    finally:
        _teardown(test_runtime, old_runtime)


def test_refresh_token_single_use(tmp_path: Path) -> None:
    """A refresh token can only be used once (single-use)."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token_resp = client.post(
            "/api/v1/auth/token",
            data={"username": "admin", "password": "admin123"},
        )
        refresh_token = token_resp.json()["refresh_token"]

        # First use succeeds
        r1 = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        assert r1.status_code == 200

        # Second use fails (revoked)
        r2 = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        assert r2.status_code == 401
    finally:
        _teardown(test_runtime, old_runtime)


def test_refresh_invalid_token(tmp_path: Path) -> None:
    """POST /api/v1/auth/refresh with invalid string returns 401."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "this-is-not-a-valid-token"},
        )
        assert response.status_code == 401
    finally:
        _teardown(test_runtime, old_runtime)


def test_refresh_expired_token(tmp_path: Path) -> None:
    """POST /api/v1/auth/refresh with expired refresh token returns 401."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        expired = create_refresh_token(
            user_id=1, username="admin", role="admin", expires_minutes=-60,
        )
        response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": expired},
        )
        assert response.status_code == 401
    finally:
        _teardown(test_runtime, old_runtime)


def test_refresh_token_not_access_token(tmp_path: Path) -> None:
    """An access token cannot be used as a refresh token (type mismatch)."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        access = create_access_token(user_id=1, username="admin", role="admin")
        response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": access},
        )
        assert response.status_code == 401
    finally:
        _teardown(test_runtime, old_runtime)


def test_access_token_not_refresh_token(tmp_path: Path) -> None:
    """A refresh token cannot be used as an access token."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        refresh = create_refresh_token(user_id=1, username="admin", role="admin")
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {refresh}"},
        )
        assert response.status_code == 401
    finally:
        _teardown(test_runtime, old_runtime)


def test_logout_revokes_refresh_token(tmp_path: Path) -> None:
    """POST /api/v1/auth/logout revokes the refresh token."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token_resp = client.post(
            "/api/v1/auth/token",
            data={"username": "admin", "password": "admin123"},
        )
        refresh_token = token_resp.json()["refresh_token"]

        response = client.post(
            "/api/v1/auth/logout",
            json={"refresh_token": refresh_token},
        )
        assert response.status_code == 200
        assert response.json()["message"] == "Logged out successfully"

        # After logout, refresh fails
        r2 = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        assert r2.status_code == 401
    finally:
        _teardown(test_runtime, old_runtime)


def test_logout_invalid_token(tmp_path: Path) -> None:
    """Logout with invalid token returns 401."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        response = client.post(
            "/api/v1/auth/logout",
            json={"refresh_token": "invalid"},
        )
        assert response.status_code == 401
    finally:
        _teardown(test_runtime, old_runtime)


def test_logout_missing_token(tmp_path: Path) -> None:
    """Logout without token returns 422 (validation error)."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        response = client.post("/api/v1/auth/logout", json={})
        assert response.status_code == 422
    finally:
        _teardown(test_runtime, old_runtime)


def test_logout_missing_token_field(tmp_path: Path) -> None:
    """Logout with empty refresh_token string fails validation."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        response = client.post(
            "/api/v1/auth/logout",
            json={"refresh_token": ""},
        )
        assert response.status_code == 422
    finally:
        _teardown(test_runtime, old_runtime)


def test_logout_with_access_token(tmp_path: Path) -> None:
    """Logout with an access token (not refresh) returns 401."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        access = create_access_token(user_id=1, username="admin", role="admin")
        response = client.post(
            "/api/v1/auth/logout",
            json={"refresh_token": access},
        )
        assert response.status_code == 401
    finally:
        _teardown(test_runtime, old_runtime)


# ═══════════════════════════════════════════════════════════════════
# Step 3: User creation tests
# ═══════════════════════════════════════════════════════════════════


def test_create_admin_as_admin(tmp_path: Path) -> None:
    """Admin can create another admin."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        response = client.post(
            "/api/v1/auth/create-admin",
            json={"username": "admin2", "password": "StrongPass1", "role": "admin"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["username"] == "admin2"
        assert body["role"] == "admin"
        assert body["is_active"] is True
        assert "id" in body
        assert "password_hash" not in body
    finally:
        _teardown(test_runtime, old_runtime)


def test_create_admin_without_auth(tmp_path: Path) -> None:
    """Creating admin without auth returns 401."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        response = client.post(
            "/api/v1/auth/create-admin",
            json={"username": "admin2", "password": "StrongPass1"},
        )
        assert response.status_code == 401
    finally:
        _teardown(test_runtime, old_runtime)


def test_create_admin_as_non_admin(tmp_path: Path) -> None:
    """Non-admin (viewer) cannot create admin."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        # First create a viewer user
        admin_token = _admin_token(client)
        client.post(
            "/api/v1/auth/create-user",
            json={"username": "viewer1", "password": "StrongPass1", "role": "viewer"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        # Login as viewer
        login_resp = client.post(
            "/api/v1/auth/login",
            json={"username": "viewer1", "password": "StrongPass1"},
        )
        viewer_token = login_resp.json()["access_token"]

        response = client.post(
            "/api/v1/auth/create-admin",
            json={"username": "shouldfail", "password": "StrongPass1"},
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        assert response.status_code == 403
    finally:
        _teardown(test_runtime, old_runtime)


def test_create_user_as_admin(tmp_path: Path) -> None:
    """Admin can create a regular user."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        response = client.post(
            "/api/v1/auth/create-user",
            json={"username": "operator1", "password": "StrongPass1", "role": "operator"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["username"] == "operator1"
        assert body["role"] == "operator"
        assert "password_hash" not in body
    finally:
        _teardown(test_runtime, old_runtime)


def test_create_user_as_operator(tmp_path: Path) -> None:
    """Operator can create viewer users."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        client.post(
            "/api/v1/auth/create-user",
            json={"username": "op1", "password": "StrongPass1", "role": "operator"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        op_login = client.post(
            "/api/v1/auth/login",
            json={"username": "op1", "password": "StrongPass1"},
        )
        op_token = op_login.json()["access_token"]

        response = client.post(
            "/api/v1/auth/create-user",
            json={"username": "viewer1", "password": "StrongPass1", "role": "viewer"},
            headers={"Authorization": f"Bearer {op_token}"},
        )
        assert response.status_code == 201
        assert response.json()["role"] == "viewer"
    finally:
        _teardown(test_runtime, old_runtime)


def test_operator_cannot_create_admin(tmp_path: Path) -> None:
    """Operator cannot create an admin user."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        client.post(
            "/api/v1/auth/create-user",
            json={"username": "op1", "password": "StrongPass1", "role": "operator"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        op_login = client.post(
            "/api/v1/auth/login",
            json={"username": "op1", "password": "StrongPass1"},
        )
        op_token = op_login.json()["access_token"]

        response = client.post(
            "/api/v1/auth/create-user",
            json={"username": "badadmin", "password": "StrongPass1", "role": "admin"},
            headers={"Authorization": f"Bearer {op_token}"},
        )
        assert response.status_code == 403
    finally:
        _teardown(test_runtime, old_runtime)


def test_create_duplicate_username(tmp_path: Path) -> None:
    """Creating a user with an existing username returns 400."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        response = client.post(
            "/api/v1/auth/create-user",
            json={"username": "admin", "password": "StrongPass1"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]
    finally:
        _teardown(test_runtime, old_runtime)


def test_create_duplicate_email(tmp_path: Path) -> None:
    """Creating a user with an existing email returns 400."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        client.post(
            "/api/v1/auth/create-user",
            json={"username": "user1", "password": "StrongPass1", "email": "dup@example.com"},
            headers={"Authorization": f"Bearer {token}"},
        )
        response = client.post(
            "/api/v1/auth/create-user",
            json={"username": "user2", "password": "StrongPass1", "email": "dup@example.com"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]
    finally:
        _teardown(test_runtime, old_runtime)


def test_create_user_weak_password(tmp_path: Path) -> None:
    """Creating a user with a weak password returns 422."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        response = client.post(
            "/api/v1/auth/create-user",
            json={"username": "newuser", "password": "short", "role": "viewer"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 422
    finally:
        _teardown(test_runtime, old_runtime)


def test_admin_cannot_create_superadmin(tmp_path: Path) -> None:
    """An admin may not create a superadmin account."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        response = client.post(
            "/api/v1/auth/create-user",
            json={"username": "newuser", "password": "StrongPass1", "role": "superadmin"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403
        assert "اجازه" in response.json()["detail"]
    finally:
        _teardown(test_runtime, old_runtime)


# ═══════════════════════════════════════════════════════════════════
# Step 4: User management tests
# ═══════════════════════════════════════════════════════════════════


def test_list_users_as_admin(tmp_path: Path) -> None:
    """Admin can list users."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        response = client.get(
            "/api/v1/auth/users",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        assert len(body) >= 1
        # Verify no password hash in response
        assert "password_hash" not in body[0]
    finally:
        _teardown(test_runtime, old_runtime)


def test_list_users_as_non_admin(tmp_path: Path) -> None:
    """Non-admin cannot list users."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        # Need a non-admin user; create one
        admin_token = _admin_token(client)
        client.post(
            "/api/v1/auth/create-user",
            json={"username": "viewer1", "password": "StrongPass1", "role": "viewer"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        viewer_login = client.post(
            "/api/v1/auth/login",
            json={"username": "viewer1", "password": "StrongPass1"},
        )
        viewer_token = viewer_login.json()["access_token"]

        response = client.get(
            "/api/v1/auth/users",
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        assert response.status_code == 403
    finally:
        _teardown(test_runtime, old_runtime)


def test_list_users_without_auth(tmp_path: Path) -> None:
    """Listing users without auth returns 401."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        response = client.get("/api/v1/auth/users")
        assert response.status_code == 401
    finally:
        _teardown(test_runtime, old_runtime)


def test_get_user_by_id(tmp_path: Path) -> None:
    """Admin can get user details by ID."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        response = client.get(
            "/api/v1/auth/users/1",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["username"] == "admin"
        assert body["role"] == "admin"
        assert "password_hash" not in body
    finally:
        _teardown(test_runtime, old_runtime)


def test_get_user_not_found(tmp_path: Path) -> None:
    """Getting a nonexistent user returns 404."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        response = client.get(
            "/api/v1/auth/users/9999",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 404
    finally:
        _teardown(test_runtime, old_runtime)


def test_update_user(tmp_path: Path) -> None:
    """Admin can update user fields."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        # Create a user first
        client.post(
            "/api/v1/auth/create-user",
            json={"username": "updateuser", "password": "StrongPass1", "role": "viewer"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        response = client.put(
            "/api/v1/auth/users/2",
            json={"username": "updateduser", "email": "new@example.com"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["username"] == "updateduser"
        assert body["email"] == "new@example.com"
    finally:
        _teardown(test_runtime, old_runtime)


def test_update_user_not_found(tmp_path: Path) -> None:
    """Updating a nonexistent user returns 404."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        response = client.put(
            "/api/v1/auth/users/9999",
            json={"username": "nobody"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 404
    finally:
        _teardown(test_runtime, old_runtime)


def test_change_role(tmp_path: Path) -> None:
    """Admin can change another user's role."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        client.post(
            "/api/v1/auth/create-user",
            json={"username": "roleuser", "password": "StrongPass1", "role": "viewer"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        response = client.patch(
            "/api/v1/auth/users/2/role",
            json={"role": "operator"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code == 200
        assert response.json()["role"] == "operator"
    finally:
        _teardown(test_runtime, old_runtime)


def test_change_role_invalid(tmp_path: Path) -> None:
    """Changing role to invalid value returns 422."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        response = client.patch(
            "/api/v1/auth/users/2/role",
            json={"role": "superadmin"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 422
    finally:
        _teardown(test_runtime, old_runtime)


def test_change_role_not_found(tmp_path: Path) -> None:
    """Changing role of nonexistent user returns 404."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        response = client.patch(
            "/api/v1/auth/users/9999/role",
            json={"role": "viewer"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 404
    finally:
        _teardown(test_runtime, old_runtime)


def test_cannot_deactivate_last_admin(tmp_path: Path) -> None:
    """Cannot deactivate the last active admin."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        response = client.put(
            "/api/v1/auth/users/1",
            json={"is_active": False},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 400
        assert "last active admin" in response.json()["detail"]
    finally:
        _teardown(test_runtime, old_runtime)


def test_cannot_demote_last_admin(tmp_path: Path) -> None:
    """Cannot change the role of the last active admin."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        response = client.patch(
            "/api/v1/auth/users/1/role",
            json={"role": "viewer"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 400
        assert "last active admin" in response.json()["detail"]
    finally:
        _teardown(test_runtime, old_runtime)


# ═══════════════════════════════════════════════════════════════════
# Step 5: Password change tests
# ═══════════════════════════════════════════════════════════════════


def test_change_password_success(tmp_path: Path) -> None:
    """User can change their own password with correct current password."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        response = client.post(
            "/api/v1/auth/me/password",
            json={
                "current_password": "admin123",
                "new_password": "NewStrongPass1",
                "confirm_password": "NewStrongPass1",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert response.json()["message"] == "Password changed successfully"

        # Old password no longer works
        old_login = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"},
        )
        assert old_login.status_code == 401

        # New password works
        new_login = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "NewStrongPass1"},
        )
        assert new_login.status_code == 200
    finally:
        _teardown(test_runtime, old_runtime)


def test_change_password_wrong_current(tmp_path: Path) -> None:
    """Password change fails with incorrect current password."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        response = client.post(
            "/api/v1/auth/me/password",
            json={
                "current_password": "wrongpassword",
                "new_password": "NewStrongPass1",
                "confirm_password": "NewStrongPass1",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 400
        assert "incorrect" in response.json()["detail"]
    finally:
        _teardown(test_runtime, old_runtime)


def test_change_password_mismatch(tmp_path: Path) -> None:
    """Password change fails when new and confirm passwords don't match."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        response = client.post(
            "/api/v1/auth/me/password",
            json={
                "current_password": "admin123",
                "new_password": "NewStrongPass1",
                "confirm_password": "DifferentPass1",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 422
    finally:
        _teardown(test_runtime, old_runtime)


def test_change_password_same_as_old(tmp_path: Path) -> None:
    """Password change fails when new password is same as current."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        response = client.post(
            "/api/v1/auth/me/password",
            json={
                "current_password": "admin123",
                "new_password": "admin123",
                "confirm_password": "admin123",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 400
        assert "different" in response.json()["detail"]
    finally:
        _teardown(test_runtime, old_runtime)


def test_change_password_without_auth(tmp_path: Path) -> None:
    """Password change without auth returns 401."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        response = client.post(
            "/api/v1/auth/me/password",
            json={
                "current_password": "admin123",
                "new_password": "NewStrongPass1",
                "confirm_password": "NewStrongPass1",
            },
        )
        assert response.status_code == 401
    finally:
        _teardown(test_runtime, old_runtime)


def test_change_password_weak_new(tmp_path: Path) -> None:
    """Password change fails with a weak new password."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        response = client.post(
            "/api/v1/auth/me/password",
            json={
                "current_password": "admin123",
                "new_password": "short",
                "confirm_password": "short",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 422
    finally:
        _teardown(test_runtime, old_runtime)


def test_password_change_old_tokens_still_valid(tmp_path: Path) -> None:
    """Previously issued access tokens remain valid after password change."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        old_token = _admin_token(client)

        # Change password
        client.post(
            "/api/v1/auth/me/password",
            json={
                "current_password": "admin123",
                "new_password": "NewStrongPass1",
                "confirm_password": "NewStrongPass1",
            },
            headers={"Authorization": f"Bearer {old_token}"},
        )

        # Old access token still works (no forced invalidation)
        me_resp = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {old_token}"},
        )
        assert me_resp.status_code == 200, (
            "Old access tokens should remain valid after password change"
        )
    finally:
        _teardown(test_runtime, old_runtime)


# ═══════════════════════════════════════════════════════════════════
# Step 6: Disabled auth mode tests (DISABLE_AUTH=true)
# ═══════════════════════════════════════════════════════════════════


def test_disabled_auth_allows_access_without_token(tmp_path: Path) -> None:
    """With DISABLE_AUTH=true, protected endpoints work without a token."""
    os.environ["DISABLE_AUTH"] = "true"
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 200
        body = response.json()
        assert body["username"] == "dev"
        assert body["role"] == "admin"
    finally:
        os.environ.pop("DISABLE_AUTH", None)
        _teardown(test_runtime, old_runtime)


def test_disabled_auth_skips_role_checks(tmp_path: Path) -> None:
    """With DISABLE_AUTH=true, admin-only endpoints work without a token."""
    os.environ["DISABLE_AUTH"] = "true"
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        # /api/v1/auth/users is admin-only; should work without token
        response = client.get("/api/v1/auth/users")
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
    finally:
        os.environ.pop("DISABLE_AUTH", None)
        _teardown(test_runtime, old_runtime)


def test_disabled_auth_with_invalid_token(tmp_path: Path) -> None:
    """With DISABLE_AUTH=true, even an invalid token is accepted (bypassed)."""
    os.environ["DISABLE_AUTH"] = "true"
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer this-is-not-valid"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["username"] == "dev"
    finally:
        os.environ.pop("DISABLE_AUTH", None)
        _teardown(test_runtime, old_runtime)


def test_disabled_auth_login_still_works(tmp_path: Path) -> None:
    """Login endpoint still functions normally when auth is disabled."""
    os.environ["DISABLE_AUTH"] = "true"
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"},
        )
        # Login is an unprotected endpoint; it should still work
        assert response.status_code == 200
        assert "access_token" in response.json()
    finally:
        os.environ.pop("DISABLE_AUTH", None)
        _teardown(test_runtime, old_runtime)
