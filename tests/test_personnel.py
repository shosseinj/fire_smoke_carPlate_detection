from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.runtime import build_runtime


# ── Test helpers ─────────────────────────────────────────────────────


def _make_test_runtime(tmp_path: Path):
    """Build a minimal mock runtime with clean databases."""
    test_settings = replace(
        settings,
        processor_mode="mock",
        camera_db_path=tmp_path / "cameras.sqlite3",
        source_registry_path=tmp_path / "sources.json",
        plate_log_db_path=tmp_path / "plate_logs.sqlite3",
        auth_db_path=tmp_path / "auth.sqlite3",
        saved_media_path=tmp_path / "saved_media",
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

    # Replace global auth store
    from app.core import auth as auth_core
    from app.core.auth_store import AuthStore
    old_store = auth_core._auth_store
    auth_core._auth_store = AuthStore(tmp_path / "auth.sqlite3")
    auth_core._auth_store.seed_default_admin("admin", "admin123")

    return test_runtime, old_runtime, TestClient(main_module.app), old_store


def _teardown(test_runtime, old_runtime, old_store):
    """Restore runtime and auth store."""
    import app.main as main_module
    from app.core import auth as auth_core

    test_runtime.close()
    main_module.runtime = old_runtime
    auth_core._auth_store = old_store


def _admin_token(client: TestClient) -> str:
    """Helper: login as admin and return access token."""
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "admin123"},
    )
    return resp.json()["access_token"]


def _operator_token(client: TestClient) -> str:
    """Helper: create an operator user and return token."""
    admin_token = _admin_token(client)
    client.post(
        "/api/v1/auth/create-user",
        json={"username": "operator1", "password": "operator123", "role": "operator"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": "operator1", "password": "operator123"},
    )
    return resp.json()["access_token"]


def _viewer_token(client: TestClient) -> str:
    """Helper: create a viewer user and return token."""
    admin_token = _admin_token(client)
    client.post(
        "/api/v1/auth/create-user",
        json={"username": "viewer1", "password": "viewer1234", "role": "viewer"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": "viewer1", "password": "viewer1234"},
    )
    return resp.json()["access_token"]


_HEADERS = {"Content-Type": "application/json"}

# ── Valid Iranian national codes for testing ─────────────────────────
# These are computed to pass the checksum algorithm:
# 1234567891: total=210, 210%11=1 (<2) -> checksum=1 ✓
# 9876543210: total=330, 330%11=0 (<2) -> checksum=0 ✓
# 1234123411: total=122, 122%11=1 (<2) -> checksum=1 ✓
VALID_CODE_1 = "1234567891"
VALID_CODE_2 = "9876543210"
VALID_CODE_3 = "1234123411"


# ═══════════════════════════════════════════════════════════════════
# Personnel CRUD tests
# ═══════════════════════════════════════════════════════════════════


def test_create_personnel(tmp_path: Path) -> None:
    test_runtime, old_runtime, client, old_store = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        resp = client.post(
            "/api/v1/personnel/",
            json={
                "fname": "Ali",
                "lname": "Mohammadi",
                "national_code": VALID_CODE_1,
                "employee_type": "employee",
                "degree": "Bachelor",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["fname"] == "Ali"
        assert body["lname"] == "Mohammadi"
        assert body["national_code"] == VALID_CODE_1
        assert body["employee_type"] == "employee"
        assert body["degree"] == "Bachelor"
        assert body["id"] > 0
        assert body["created_at_utc"] is not None
    finally:
        _teardown(test_runtime, old_runtime, old_store)


def test_create_personnel_duplicate_national_code(tmp_path: Path) -> None:
    test_runtime, old_runtime, client, old_store = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        client.post(
            "/api/v1/personnel/",
            json={"fname": "Ali", "lname": "Mohammadi", "national_code": VALID_CODE_1},
            headers={"Authorization": f"Bearer {token}"},
        )
        resp = client.post(
            "/api/v1/personnel/",
            json={"fname": "Reza", "lname": "Ahmadi", "national_code": VALID_CODE_1},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 422
        assert "already exists" in resp.text
    finally:
        _teardown(test_runtime, old_runtime, old_store)


def test_create_personnel_invalid_national_code(tmp_path: Path) -> None:
    test_runtime, old_runtime, client, old_store = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        resp = client.post(
            "/api/v1/personnel/",
            json={"fname": "Ali", "lname": "Mohammadi", "national_code": "12345"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 422
        assert "Invalid" in resp.text
    finally:
        _teardown(test_runtime, old_runtime, old_store)


def test_create_personnel_viewer_forbidden(tmp_path: Path) -> None:
    test_runtime, old_runtime, client, old_store = _setup_client(tmp_path)
    try:
        token = _viewer_token(client)
        resp = client.post(
            "/api/v1/personnel/",
            json={"fname": "Ali", "lname": "Mohammadi", "national_code": VALID_CODE_1},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403
    finally:
        _teardown(test_runtime, old_runtime, old_store)


def test_get_personnel(tmp_path: Path) -> None:
    test_runtime, old_runtime, client, old_store = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        create_resp = client.post(
            "/api/v1/personnel/",
            json={"fname": "Ali", "lname": "Mohammadi", "national_code": VALID_CODE_1},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        person_id = create_resp.json()["id"]

        # Read by ID
        op_token = _operator_token(client)
        resp = client.get(
            f"/api/v1/personnel/{person_id}",
            headers={"Authorization": f"Bearer {op_token}"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["fname"] == "Ali"
        assert body["national_code"] == VALID_CODE_1
    finally:
        _teardown(test_runtime, old_runtime, old_store)


def test_get_personnel_not_found(tmp_path: Path) -> None:
    test_runtime, old_runtime, client, old_store = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        resp = client.get(
            "/api/v1/personnel/99999",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404
    finally:
        _teardown(test_runtime, old_runtime, old_store)


def test_update_personnel(tmp_path: Path) -> None:
    test_runtime, old_runtime, client, old_store = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        create_resp = client.post(
            "/api/v1/personnel/",
            json={"fname": "Ali", "lname": "Mohammadi", "national_code": VALID_CODE_1},
            headers={"Authorization": f"Bearer {token}"},
        )
        person_id = create_resp.json()["id"]

        resp = client.put(
            f"/api/v1/personnel/{person_id}",
            json={"fname": "Reza", "degree": "Master"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["fname"] == "Reza"
        assert body["lname"] == "Mohammadi"  # unchanged
        assert body["degree"] == "Master"
    finally:
        _teardown(test_runtime, old_runtime, old_store)


def test_delete_personnel(tmp_path: Path) -> None:
    test_runtime, old_runtime, client, old_store = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        create_resp = client.post(
            "/api/v1/personnel/",
            json={"fname": "Ali", "lname": "Mohammadi", "national_code": VALID_CODE_1},
            headers={"Authorization": f"Bearer {token}"},
        )
        person_id = create_resp.json()["id"]

        resp = client.delete(
            f"/api/v1/personnel/{person_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

        # Verify deleted
        get_resp = client.get(
            f"/api/v1/personnel/{person_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert get_resp.status_code == 404
    finally:
        _teardown(test_runtime, old_runtime, old_store)


# ═══════════════════════════════════════════════════════════════════
# Search tests
# ═══════════════════════════════════════════════════════════════════


def test_search_personnel_by_national_code(tmp_path: Path) -> None:
    test_runtime, old_runtime, client, old_store = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        client.post(
            "/api/v1/personnel/",
            json={"fname": "Ali", "lname": "Mohammadi", "national_code": VALID_CODE_1},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        op_token = _operator_token(client)
        resp = client.get(
            f"/api/v1/personnel/search/{VALID_CODE_1}",
            headers={"Authorization": f"Bearer {op_token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["national_code"] == VALID_CODE_1
    finally:
        _teardown(test_runtime, old_runtime, old_store)


def test_search_personnel_not_found(tmp_path: Path) -> None:
    test_runtime, old_runtime, client, old_store = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        resp = client.get(
            f"/api/v1/personnel/search/{VALID_CODE_2}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404
    finally:
        _teardown(test_runtime, old_runtime, old_store)


# ═══════════════════════════════════════════════════════════════════
# Personnel image tests
# ═══════════════════════════════════════════════════════════════════


def _create_test_person(client, token) -> dict:
    resp = client.post(
        "/api/v1/personnel/",
        json={"fname": "Ali", "lname": "Mohammadi", "national_code": VALID_CODE_1},
        headers={"Authorization": f"Bearer {token}"},
    )
    return resp.json()


def _jpeg_bytes() -> bytes:
    """Return minimal valid JPEG bytes."""
    import struct
    # Minimal JPEG SOI + EOI
    return b"\xff\xd8\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 14 + b"\xff\xd9"


def test_upload_image(tmp_path: Path) -> None:
    test_runtime, old_runtime, client, old_store = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        person = _create_test_person(client, admin_token)
        person_id = person["id"]

        img_data = _jpeg_bytes()
        resp = client.post(
            f"/api/v1/personnel/{person_id}/images",
            files={"file": ("test.jpg", img_data, "image/jpeg")},
            data={"description": "Face photo"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["personnel_id"] == person_id
        assert body["description"] == "Face photo"
        assert body["is_primary"] is True  # First image is primary
        assert body["storage_key"].startswith("personnel_snapshots/")
    finally:
        _teardown(test_runtime, old_runtime, old_store)


def test_list_images(tmp_path: Path) -> None:
    test_runtime, old_runtime, client, old_store = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        person = _create_test_person(client, admin_token)
        person_id = person["id"]

        img_data = _jpeg_bytes()
        client.post(
            f"/api/v1/personnel/{person_id}/images",
            files={"file": ("face.jpg", img_data, "image/jpeg")},
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        op_token = _operator_token(client)
        resp = client.get(
            f"/api/v1/personnel/{person_id}/images",
            headers={"Authorization": f"Bearer {op_token}"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] >= 1
        assert len(body["items"]) >= 1
    finally:
        _teardown(test_runtime, old_runtime, old_store)


def test_set_primary_image(tmp_path: Path) -> None:
    test_runtime, old_runtime, client, old_store = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        person = _create_test_person(client, admin_token)
        person_id = person["id"]

        img_data = _jpeg_bytes()
        # Upload two images
        resp1 = client.post(
            f"/api/v1/personnel/{person_id}/images",
            files={"file": ("first.jpg", img_data, "image/jpeg")},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        resp2 = client.post(
            f"/api/v1/personnel/{person_id}/images",
            files={"file": ("second.jpg", img_data, "image/jpeg")},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        first_id = resp1.json()["id"]
        second_id = resp2.json()["id"]

        # Second should not be primary initially
        assert resp1.json()["is_primary"] is True
        assert resp2.json()["is_primary"] is False

        # Set second as primary
        resp = client.put(
            f"/api/v1/personnel/images/{second_id}/set-primary",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["is_primary"] is True

        # First should no longer be primary
        first = client.get(
            f"/api/v1/personnel/images/{first_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert first.json()["is_primary"] is False
    finally:
        _teardown(test_runtime, old_runtime, old_store)


def test_delete_image_promotes_next(tmp_path: Path) -> None:
    test_runtime, old_runtime, client, old_store = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        person = _create_test_person(client, admin_token)
        person_id = person["id"]

        img_data = _jpeg_bytes()
        resp1 = client.post(
            f"/api/v1/personnel/{person_id}/images",
            files={"file": ("first.jpg", img_data, "image/jpeg")},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        resp2 = client.post(
            f"/api/v1/personnel/{person_id}/images",
            files={"file": ("second.jpg", img_data, "image/jpeg")},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        first_id = resp1.json()["id"]
        second_id = resp2.json()["id"]

        # Delete the first (primary) — second should become primary
        resp = client.delete(
            f"/api/v1/personnel/images/{first_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200

        second = client.get(
            f"/api/v1/personnel/images/{second_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert second.json()["is_primary"] is True
    finally:
        _teardown(test_runtime, old_runtime, old_store)


# ═══════════════════════════════════════════════════════════════════
# with-images endpoint
# ═══════════════════════════════════════════════════════════════════


def test_list_with_images(tmp_path: Path) -> None:
    test_runtime, old_runtime, client, old_store = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        person = _create_test_person(client, admin_token)

        # Upload an image
        img_data = _jpeg_bytes()
        client.post(
            f"/api/v1/personnel/{person['id']}/images",
            files={"file": ("face.jpg", img_data, "image/jpeg")},
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        op_token = _operator_token(client)
        resp = client.get(
            "/api/v1/personnel/with-images",
            headers={"Authorization": f"Bearer {op_token}"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] >= 1
        # The person record should have images nested
        found = [item for item in body["items"] if item["id"] == person["id"]]
        assert len(found) == 1
        assert len(found[0]["images"]) >= 1
    finally:
        _teardown(test_runtime, old_runtime, old_store)


# ═══════════════════════════════════════════════════════════════════
# Import-export tests
# ═══════════════════════════════════════════════════════════════════


def test_import_template(tmp_path: Path) -> None:
    test_runtime, old_runtime, client, old_store = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        resp = client.get(
            "/api/v1/personnel/import-template",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        assert len(resp.content) > 0
    finally:
        _teardown(test_runtime, old_runtime, old_store)


def test_import_excel(tmp_path: Path) -> None:
    test_runtime, old_runtime, client, old_store = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        # Build a minimal Excel file using openpyxl
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Personnel Import"
        ws.append(["fname", "lname", "national_code", "employee_type", "degree"])
        ws.append(["Sara", "Hosseini", VALID_CODE_2, "employee", "PhD"])
        ws.append(["Mohsen", "Rezaei", VALID_CODE_3, "contractor", ""])
        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)

        resp = client.post(
            "/api/v1/personnel/import-excel",
            files={"file": ("import.xlsx", buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["created"] == 2
        assert body["skipped"] == 0
        assert len(body["errors"]) == 0

        # Verify personnel were created
        search = client.get(
            f"/api/v1/personnel/search/{VALID_CODE_2}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert search.status_code == 200
        assert search.json()["fname"] == "Sara"
    finally:
        _teardown(test_runtime, old_runtime, old_store)


# ═══════════════════════════════════════════════════════════════════
# Auth guard tests
# ═══════════════════════════════════════════════════════════════════


def test_endpoints_require_auth(tmp_path: Path) -> None:
    test_runtime, old_runtime, client, old_store = _setup_client(tmp_path)
    try:
        # No auth header — expect 401 (not authenticated)
        endpoints = [
            ("GET", "/api/v1/personnel/"),
            ("POST", "/api/v1/personnel/"),
            ("GET", "/api/v1/personnel/1"),
            ("PUT", "/api/v1/personnel/1"),
            ("DELETE", "/api/v1/personnel/1"),
            ("GET", "/api/v1/personnel/search/1234567891"),
            ("GET", "/api/v1/personnel/with-images"),
            ("GET", "/api/v1/personnel/1/images"),
            ("GET", "/api/v1/personnel/images/1"),
            ("DELETE", "/api/v1/personnel/images/1"),
            ("PUT", "/api/v1/personnel/images/1/set-primary"),
            ("POST", "/api/v1/personnel/import-excel"),
            ("GET", "/api/v1/personnel/import-template"),
            ("POST", "/api/v1/personnel/upload-personnel-zip"),
        ]
        for method, url in endpoints:
            if method == "GET":
                resp = client.get(url)
            elif method == "POST":
                resp = client.post(url)
            elif method == "PUT":
                resp = client.put(url)
            elif method == "DELETE":
                resp = client.delete(url)
            else:
                continue
            assert resp.status_code == 401, f"{method} {url} returned {resp.status_code}"
    finally:
        _teardown(test_runtime, old_runtime, old_store)


def test_list_endpoint_operator_allowed(tmp_path: Path) -> None:
    test_runtime, old_runtime, client, old_store = _setup_client(tmp_path)
    try:
        token = _operator_token(client)
        resp = client.get(
            "/api/v1/personnel/",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
    finally:
        _teardown(test_runtime, old_runtime, old_store)


def test_admin_required_for_create(tmp_path: Path) -> None:
    test_runtime, old_runtime, client, old_store = _setup_client(tmp_path)
    try:
        token = _operator_token(client)
        resp = client.post(
            "/api/v1/personnel/",
            json={"fname": "Ali", "lname": "Mohammadi", "national_code": VALID_CODE_1},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403
    finally:
        _teardown(test_runtime, old_runtime, old_store)
