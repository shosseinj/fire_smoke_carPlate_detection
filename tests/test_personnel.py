from __future__ import annotations

import io
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.runtime import build_runtime


pytestmark = pytest.mark.usefixtures("postgres_database")


# ── Test helpers ─────────────────────────────────────────────────────


def _test_database_url() -> str:
    """Return the disposable PostgreSQL URL configured for tests."""
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL must point to a disposable PostgreSQL database")
    return url


def _make_test_runtime(tmp_path: Path):
    """Build a minimal mock runtime with clean databases."""
    test_settings = replace(
        settings,
        processor_mode="mock",
        database_url=_test_database_url(),
        source_registry_path=tmp_path / "sources.json",
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

    # Auth store already initialized by build_runtime via initialize_auth_store
    return test_runtime, old_runtime, TestClient(main_module.app)


def _teardown(test_runtime, old_runtime):
    """Restore runtime."""
    import app.main as main_module

    test_runtime.close()
    main_module.runtime = old_runtime


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
    test_runtime, old_runtime, client = _setup_client(tmp_path)
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
        assert body["created_at"] is not None
    finally:
        _teardown(test_runtime, old_runtime)


def test_create_personnel_duplicate_national_code(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
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
        _teardown(test_runtime, old_runtime)


def test_create_personnel_invalid_national_code(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
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
        _teardown(test_runtime, old_runtime)


def test_create_personnel_viewer_forbidden(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _viewer_token(client)
        resp = client.post(
            "/api/v1/personnel/",
            json={"fname": "Ali", "lname": "Mohammadi", "national_code": VALID_CODE_1},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403
    finally:
        _teardown(test_runtime, old_runtime)


def test_get_personnel(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
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
        _teardown(test_runtime, old_runtime)


def test_get_personnel_not_found(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        resp = client.get(
            "/api/v1/personnel/99999",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404
    finally:
        _teardown(test_runtime, old_runtime)


def test_update_personnel(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
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
        _teardown(test_runtime, old_runtime)


def test_delete_personnel(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
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
        assert resp.status_code == 204, resp.text
        assert resp.content == b""  # 204 No Content

        # Verify deleted
        get_resp = client.get(
            f"/api/v1/personnel/{person_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert get_resp.status_code == 404
    finally:
        _teardown(test_runtime, old_runtime)


# ═══════════════════════════════════════════════════════════════════
# Search tests
# ═══════════════════════════════════════════════════════════════════


def test_search_personnel_by_national_code(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
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
        _teardown(test_runtime, old_runtime)


def test_search_personnel_not_found(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        resp = client.get(
            f"/api/v1/personnel/search/{VALID_CODE_2}?contract=current",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404
    finally:
        _teardown(test_runtime, old_runtime)


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
    """Return a small valid JPEG image encoded via OpenCV."""
    import cv2
    import numpy as np
    # Create a small colored image (blue square)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = (200, 100, 50)  # BGR
    success, encoded = cv2.imencode(".jpg", img)
    if not success:
        # Fallback to minimal JPEG header if cv2 fails
        return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00\x43\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\x09\x09\x08\x0a\x0c\x14\x0d\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c\x20\x24\x2e\x27\x20\x22\x2c\x23\x1c\x1c\x28\x37\x29\x2c\x30\x31\x34\x34\x34\x1f\x27\x39\x3d\x38\x32\x3c\x2e\x33\x34\x32\xff\xc0\x00\x0b\x08\x00\x64\x00\x64\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x11\x04\x12\x21\x31\x41\x05\x13\x51\x61\x22\x71\x81\x32\x06\x14\x91\xa1\xb1\xc1\x09\x23\x52\xf0\x15\x42\xd1\xe1\xf1\x33\x62\x72\x82\x92\x43\x53\x63\x73\x93\xa2\xb2\xc2\xd2\xe2\xf2\x24\x34\x54\x64\x74\x84\x94\xa3\xb3\xc3\xd3\xe3\xf3\x35\x55\x65\x75\x85\x95\xa5\xb5\xc5\xd5\xe5\xf5\x36\x46\x56\x66\x76\x86\x96\xa6\xb6\xc6\xd6\xe6\xf6\x37\x47\x57\x67\x77\x87\x97\xa7\xb7\xc7\xd7\xe7\xf7\x38\x48\x58\x68\x78\x88\x98\xa8\xb8\xc8\xd8\xe8\xf8\x39\x49\x59\x69\x79\x89\x99\xa9\xb9\xc9\xd9\xe9\xf9\x3a\x4a\x5a\x6a\x7a\x8a\x9a\xaa\xba\xca\xda\xea\xfa\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\xfb\xa5\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\x11\xff\xd9"
    return encoded.tobytes()


def _upload_one_image(client, token, person_id, filename=None):
    """Helper: POST a single image and return the batch response."""
    img_data = _jpeg_bytes()
    resp = client.post(
        f"/api/v1/personnel/{person_id}/images",
        files={"files": (filename or "test.jpg", img_data, "image/jpeg")},
        data={"enable_cropping": "false"},
        headers={"Authorization": f"Bearer {token}"},
    )
    return resp


def test_upload_image(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        person = _create_test_person(client, admin_token)
        person_id = person["id"]

        resp = _upload_one_image(client, admin_token, person_id)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["total_success"] == 1
        assert body["total_failed"] == 0
        assert len(body["results"]) == 1
        assert body["results"][0]["success"] is True
        assert body["results"][0]["image"]["personnel_id"] == person_id
        assert body["results"][0]["image"]["is_primary"] is True
        assert body["results"][0]["image"]["storage_key"].startswith("personnel_snapshots/")
        # Personnel in response should have images
        assert len(body["personnel"]["images"]) >= 1
    finally:
        _teardown(test_runtime, old_runtime)


def test_list_images(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        person = _create_test_person(client, admin_token)
        person_id = person["id"]

        _upload_one_image(client, admin_token, person_id)

        op_token = _operator_token(client)
        resp = client.get(
            f"/api/v1/personnel/{person_id}/images?contract=current",
            headers={"Authorization": f"Bearer {op_token}"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] >= 1
        assert len(body["items"]) >= 1
    finally:
        _teardown(test_runtime, old_runtime)


def test_set_primary_image(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        person = _create_test_person(client, admin_token)
        person_id = person["id"]

        # Upload two images
        resp1 = _upload_one_image(client, admin_token, person_id)
        resp2 = _upload_one_image(client, admin_token, person_id, "second.jpg")
        first_id = resp1.json()["results"][0]["image"]["id"]
        second_id = resp2.json()["results"][0]["image"]["id"]

        # Second should not be primary initially
        assert resp1.json()["results"][0]["image"]["is_primary"] is True
        assert resp2.json()["results"][0]["image"]["is_primary"] is False

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
        _teardown(test_runtime, old_runtime)


def test_delete_image_promotes_next(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        person = _create_test_person(client, admin_token)
        person_id = person["id"]

        resp1 = _upload_one_image(client, admin_token, person_id)
        resp2 = _upload_one_image(client, admin_token, person_id, "second.jpg")
        first_id = resp1.json()["results"][0]["image"]["id"]
        second_id = resp2.json()["results"][0]["image"]["id"]

        # Delete the first (primary) — second should become primary
        resp = client.delete(
            f"/api/v1/personnel/images/{first_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 204, resp.text
        assert resp.content == b""  # 204 No Content

        second = client.get(
            f"/api/v1/personnel/images/{second_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert second.json()["is_primary"] is True
    finally:
        _teardown(test_runtime, old_runtime)


# ═══════════════════════════════════════════════════════════════════
# with-images endpoint
# ═══════════════════════════════════════════════════════════════════


def test_list_with_images(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        person = _create_test_person(client, admin_token)

        # Upload an image
        _upload_one_image(client, admin_token, person["id"])

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
        _teardown(test_runtime, old_runtime)


# ═══════════════════════════════════════════════════════════════════
# Import-export tests
# ═══════════════════════════════════════════════════════════════════


def test_import_template(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
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
        _teardown(test_runtime, old_runtime)


def test_import_excel(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
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
            "/api/v1/personnel/import-excel?contract=current",
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
        _teardown(test_runtime, old_runtime)


# ═══════════════════════════════════════════════════════════════════
# Auth guard tests
# ═══════════════════════════════════════════════════════════════════


def test_endpoints_require_auth(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
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
        _teardown(test_runtime, old_runtime)


def test_list_endpoint_operator_allowed(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _operator_token(client)
        resp = client.get(
            "/api/v1/personnel/",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
    finally:
        _teardown(test_runtime, old_runtime)


def test_admin_required_for_create(tmp_path: Path) -> None:
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _operator_token(client)
        resp = client.post(
            "/api/v1/personnel/",
            json={"fname": "Ali", "lname": "Mohammadi", "national_code": VALID_CODE_1},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403
    finally:
        _teardown(test_runtime, old_runtime)


# ═══════════════════════════════════════════════════════════════════
# DELETE image route tests (204 + vector cleanup + storage cleanup)
# ═══════════════════════════════════════════════════════════════════


def test_delete_image_204_no_content(tmp_path: Path) -> None:
    """DELETE image returns 204 with empty body."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        person = _create_test_person(client, admin_token)
        resp = _upload_one_image(client, admin_token, person["id"])
        image_id = resp.json()["results"][0]["image"]["id"]

        resp = client.delete(
            f"/api/v1/personnel/images/{image_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 204
        assert resp.content == b""

        # Verify image is gone
        get_resp = client.get(
            f"/api/v1/personnel/images/{image_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert get_resp.status_code == 404
    finally:
        _teardown(test_runtime, old_runtime)


def test_delete_image_not_found(tmp_path: Path) -> None:
    """DELETE image with non-existent ID returns 404."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        resp = client.delete(
            "/api/v1/personnel/images/99999",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404
    finally:
        _teardown(test_runtime, old_runtime)


def test_delete_image_forbidden_for_operator(tmp_path: Path) -> None:
    """DELETE image requires admin role."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        person = _create_test_person(client, admin_token)
        resp = _upload_one_image(client, admin_token, person["id"])
        image_id = resp.json()["results"][0]["image"]["id"]

        op_token = _operator_token(client)
        resp = client.delete(
            f"/api/v1/personnel/images/{image_id}",
            headers={"Authorization": f"Bearer {op_token}"},
        )
        assert resp.status_code == 403
    finally:
        _teardown(test_runtime, old_runtime)


def test_delete_image_cleans_up_storage(tmp_path: Path) -> None:
    """DELETE image removes the physical file from disk."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        person = _create_test_person(client, admin_token)
        resp = _upload_one_image(client, admin_token, person["id"])
        image_id = resp.json()["results"][0]["image"]["id"]
        storage_key = resp.json()["results"][0]["image"]["storage_key"]
        media_root = test_runtime.personnel_store._media_root
        file_path = media_root / storage_key
        assert file_path.is_file(), "Storage file should exist before delete"

        client.delete(
            f"/api/v1/personnel/images/{image_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert not file_path.is_file(), "Storage file should be removed after delete"
    finally:
        _teardown(test_runtime, old_runtime)


def test_delete_image_last_primary_no_promotion(tmp_path: Path) -> None:
    """Deleting the only image leaves no primary; it should not crash."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        person = _create_test_person(client, admin_token)
        resp = _upload_one_image(client, admin_token, person["id"])
        image_id = resp.json()["results"][0]["image"]["id"]

        # Delete the only image
        resp = client.delete(
            f"/api/v1/personnel/images/{image_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 204

        # Verify the person has no images
        list_resp = client.get(
                f"/api/v1/personnel/{person['id']}/images?contract=current",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert list_resp.json()["count"] == 0
    finally:
        _teardown(test_runtime, old_runtime)


# ═══════════════════════════════════════════════════════════════════
# DELETE personnel route tests (204 + cleanup + Detection preservation)
# ═══════════════════════════════════════════════════════════════════


def test_delete_personnel_204_no_content(tmp_path: Path) -> None:
    """DELETE personnel returns 204 with empty body."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        person = _create_test_person(client, token)
        resp = client.delete(
            f"/api/v1/personnel/{person['id']}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 204
        assert resp.content == b""

        # Verify gone
        get_resp = client.get(
            f"/api/v1/personnel/{person['id']}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert get_resp.status_code == 404
    finally:
        _teardown(test_runtime, old_runtime)


def test_delete_personnel_not_found(tmp_path: Path) -> None:
    """DELETE personnel with non-existent ID returns 404."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        resp = client.delete(
            "/api/v1/personnel/99999",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404
    finally:
        _teardown(test_runtime, old_runtime)


def test_delete_personnel_forbidden_for_operator(tmp_path: Path) -> None:
    """DELETE personnel requires admin role."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        person = _create_test_person(client, admin_token)
        op_token = _operator_token(client)
        resp = client.delete(
            f"/api/v1/personnel/{person['id']}",
            headers={"Authorization": f"Bearer {op_token}"},
        )
        assert resp.status_code == 403
    finally:
        _teardown(test_runtime, old_runtime)


def test_delete_personnel_cleans_up_images(tmp_path: Path) -> None:
    """DELETE personnel removes all image storage files."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        person = _create_test_person(client, admin_token)
        # Upload two images
        resp1 = _upload_one_image(client, admin_token, person["id"])
        resp2 = _upload_one_image(client, admin_token, person["id"], "second.jpg")
        sk1 = resp1.json()["results"][0]["image"]["storage_key"]
        sk2 = resp2.json()["results"][0]["image"]["storage_key"]
        media_root = test_runtime.personnel_store._media_root
        assert (media_root / sk1).is_file()
        assert (media_root / sk2).is_file()

        client.delete(
            f"/api/v1/personnel/{person['id']}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        assert not (media_root / sk1).is_file()
        assert not (media_root / sk2).is_file()
    finally:
        _teardown(test_runtime, old_runtime)


def test_delete_personnel_preserves_detections(tmp_path: Path) -> None:
    """DELETE personnel preserves human_logs rows by setting personnel_id to NULL."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        person = _create_test_person(client, admin_token)

        # Insert a human_log row referencing this personnel
        store = test_runtime.personnel_store
        # Direct DB insert to simulate an existing detection
        with store._lock, store._connection() as conn:
            conn.execute(
                """INSERT INTO human_logs
                   (session_id, camera, track_id, name, first_seen, last_seen,
                    recognition_score, snapshot_url, video_url, face_video_url,
                    snapshot_quality, best_face_quality, full_frame_video_frames,
                    accepted_face_frames, personnel_id, counts_for_attendance)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("test-session", "cam1", 1, "Test User",
                 "2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z",
                 0.95, "", "", "",
                 0.8, 0.9, 10, 5, person["id"], 1),
            )

        # Verify the log exists with personnel_id
        with store._lock, store._connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM human_logs WHERE personnel_id = ?",
                (person["id"],),
            )
            assert cursor.fetchone()[0] == 1

        # Delete personnel
        client.delete(
            f"/api/v1/personnel/{person['id']}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        # Verify detection record still exists but personnel_id is NULL
        with store._lock, store._connection() as conn:
            cursor = conn.execute(
                "SELECT personnel_id FROM human_logs WHERE session_id = ? AND camera = ? AND track_id = ?",
                ("test-session", "cam1", 1),
            )
            row = cursor.fetchone()
        assert row is not None, "Detection record should still exist"
        assert row["personnel_id"] is None, "personnel_id should be NULL"
    finally:
        _teardown(test_runtime, old_runtime)


# ═══════════════════════════════════════════════════════════════════
# Upload with is_primary form parameter tests
# ═══════════════════════════════════════════════════════════════════


def test_upload_with_is_primary_true(tmp_path: Path) -> None:
    """Uploading with is_primary=True sets that image as primary (replaces existing)."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        person = _create_test_person(client, admin_token)

        # Upload first image (auto-becomes primary)
        resp1 = _upload_one_image(client, admin_token, person["id"])
        first_id = resp1.json()["results"][0]["image"]["id"]
        assert resp1.json()["results"][0]["image"]["is_primary"] is True

        # Upload second with is_primary=True
        img_data = _jpeg_bytes()
        resp2 = client.post(
            f"/api/v1/personnel/{person['id']}/images",
            files={"files": ("second.jpg", img_data, "image/jpeg")},
            data={"enable_cropping": "false", "is_primary": "true"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp2.status_code == 201
        second_id = resp2.json()["results"][0]["image"]["id"]
        assert resp2.json()["results"][0]["image"]["is_primary"] is True

        # First should no longer be primary
        first_resp = client.get(
            f"/api/v1/personnel/images/{first_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert first_resp.json()["is_primary"] is False
        assert first_resp.json()["id"] != second_id
    finally:
        _teardown(test_runtime, old_runtime)


def test_upload_with_is_primary_false(tmp_path: Path) -> None:
    """Uploading with is_primary=False marks it non-primary explicitly."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        person = _create_test_person(client, admin_token)

        # Upload first with is_primary=False (should NOT become primary)
        img_data = _jpeg_bytes()
        resp = client.post(
            f"/api/v1/personnel/{person['id']}/images",
            files={"files": ("first.jpg", img_data, "image/jpeg")},
            data={"enable_cropping": "false", "is_primary": "false"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 201
        assert resp.json()["results"][0]["image"]["is_primary"] is False

        # Upload second (default auto) should auto-become primary
        resp2 = _upload_one_image(client, admin_token, person["id"], "second.jpg")
        assert resp2.json()["results"][0]["image"]["is_primary"] is True
    finally:
        _teardown(test_runtime, old_runtime)


def test_upload_is_primary_batch_last_wins(tmp_path: Path) -> None:
    """Uploading multiple files with is_primary=True: the last processed becomes primary."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        admin_token = _admin_token(client)
        person = _create_test_person(client, admin_token)

        # Upload two files both with is_primary=True
        img_data_1 = _jpeg_bytes()
        img_data_2 = _jpeg_bytes()
        from io import BytesIO
        resp = client.post(
            f"/api/v1/personnel/{person['id']}/images",
            files=[
                ("files", ("first.jpg", img_data_1, "image/jpeg")),
                ("files", ("second.jpg", img_data_2, "image/jpeg")),
            ],
            data={"enable_cropping": "false", "is_primary": "true"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 201
        results = resp.json()["results"]
        assert len(results) == 2
        # The last file (second.jpg) should end up as primary
        assert results[0]["success"] is True
        assert results[1]["success"] is True
        # Last processed image should be the primary
        second = client.get(
            f"/api/v1/personnel/images/{results[1]['image']['id']}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert second.json()["is_primary"] is True
    finally:
        _teardown(test_runtime, old_runtime)


# ═══════════════════════════════════════════════════════════════════
# ZIP upload test
# ═══════════════════════════════════════════════════════════════════


def _make_test_zip(national_code: str) -> bytes:
    """Create a minimal ZIP with folder-per-person structure and a JPEG image."""
    import io as io_mod
    import zipfile
    import cv2 as cv_mod
    import numpy as np_mod

    # Small valid JPEG image
    img = np_mod.zeros((50, 50, 3), dtype=np_mod.uint8)
    img[:, :] = (100, 150, 200)
    ok, encoded = cv_mod.imencode(".jpg", img)
    assert ok

    buf = io_mod.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{national_code}/photo.jpg", encoded.tobytes())
    return buf.getvalue()


def test_upload_personnel_zip_folder_structure(tmp_path: Path) -> None:
    """ZIP with folder-per-person structure creates personnel + image records."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        zip_bytes = _make_test_zip("1234567891")
        resp = client.post(
            "/api/v1/personnel/upload-personnel-zip",
            files={"file": ("test.zip", zip_bytes, "application/zip")},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is True
        assert body["summary"]["total_persons"] == 1
        assert body["summary"]["total_images_saved"] == 1
        assert body["summary"]["total_images_in_zip"] == 1
        assert body["summary"]["total_failed"] == 0
        assert body["summary"]["qdrant_enrolled_count"] == 0  # no face processor in mock mode
        assert len(body["details"]) == 1
        assert body["details"][0]["status"] in ("ok", "failed")
        assert body["details"][0]["file"] == "1234567891/photo.jpg"
        assert len(body.get("errors", [])) == 0

        # Verify the person was created with descriptive name from national code
        person_resp = client.get(
            "/api/v1/personnel/search/1234567891",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert person_resp.status_code == 200
        person = person_resp.json()
        assert person["national_code"] == "1234567891"
        assert person["fname"] == "person_1234567891"
        assert person["lname"] == ""
    finally:
        _teardown(test_runtime, old_runtime)


def test_upload_personnel_zip_with_enable_cropping(tmp_path: Path) -> None:
    """ZIP upload accepts enable_cropping parameter without error (mock mode)."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        zip_bytes = _make_test_zip("9876543210")
        resp = client.post(
            "/api/v1/personnel/upload-personnel-zip",
            files={"file": ("test.zip", zip_bytes, "application/zip")},
            data={"enable_cropping": "true"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is True
        assert body["summary"]["total_persons"] == 1
        assert body["summary"]["total_images_in_zip"] == 1
        assert body["summary"]["qdrant_enrolled_count"] == 0
        assert len(body["details"]) == 1
    finally:
        _teardown(test_runtime, old_runtime)


def test_upload_personnel_zip_invalid_national_code_saves_error(tmp_path: Path) -> None:
    """ZIP images with non-numeric filenames (no valid national code) are saved to error folder."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        import io as io_mod
        import zipfile
        import cv2 as cv2_mod
        import numpy as np_mod

        img = np_mod.zeros((50, 50, 3), dtype=np_mod.uint8)
        ok, encoded = cv2_mod.imencode(".jpg", img)
        assert ok
        buf = io_mod.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("no_nc_photo.jpg", encoded.tobytes())
        zip_bytes = buf.getvalue()

        token = _admin_token(client)
        resp = client.post(
            "/api/v1/personnel/upload-personnel-zip",
            files={"file": ("bad.zip", zip_bytes, "application/zip")},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is False
        assert body["summary"]["total_images_in_zip"] == 1
        assert body["summary"]["total_persons"] == 0
        assert body["summary"]["total_images_saved"] == 0
        assert body["summary"]["total_failed"] == 1
        assert len(body["errors"]) == 1
        assert "Cannot determine personnel" in body["errors"][0]["error"]
        assert len(body["details"]) == 1
        assert body["details"][0]["status"] == "failed"
        assert body["details"][0]["saved_to_error_folder"] is not None
        assert body["details"][0]["enrolled_in_qdrant"] is False
    finally:
        _teardown(test_runtime, old_runtime)


def test_upload_personnel_zip_folder_creates_person_with_national_code_name(tmp_path: Path) -> None:
    """ZIP with folder-per-person structure creates person named person_{national_code}
    instead of 'Unknown Unknown'."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        nc = VALID_CODE_3  # "1234123411" — valid checksum
        zip_bytes = _make_test_zip(nc)
        resp = client.post(
            "/api/v1/personnel/upload-personnel-zip",
            files={"file": ("test.zip", zip_bytes, "application/zip")},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is True
        assert body["summary"]["qdrant_enrolled_count"] == 0  # no real face processor
        assert len(body["details"]) == 1
        assert body["details"][0]["file"] == f"{nc}/photo.jpg"

        # Verify the person was created with a descriptive name
        person_resp = client.get(
            f"/api/v1/personnel/search/{nc}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert person_resp.status_code == 200
        person = person_resp.json()
        assert person["fname"] == f"person_{nc}"
        assert person["lname"] == ""
    finally:
        _teardown(test_runtime, old_runtime)


def test_upload_personnel_zip_updates_existing_unknown_name(tmp_path: Path) -> None:
    """Re-upload for a national code whose person record has 'Unknown' name
    (e.g. from a prior upload before the naming fix) updates the record to
    'person_{national_code}' so that vector enrollment can proceed."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        token = _admin_token(client)
        nc = VALID_CODE_1  # "1234567891"

        # 1. Create a person with "Unknown" name (simulating a pre-fix upload)
        resp = client.post(
            "/api/v1/personnel/",
            json={
                "fname": "Unknown",
                "lname": "Unknown",
                "national_code": nc,
                "employee_type": "unknown",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 201, resp.text

        # 2. Upload a ZIP for the same national code
        zip_bytes = _make_test_zip(nc)
        resp = client.post(
            "/api/v1/personnel/upload-personnel-zip",
            files={"file": ("test.zip", zip_bytes, "application/zip")},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is True
        assert body["summary"]["total_images_in_zip"] == 1

        # 3. Verify the person's name was updated
        person_resp = client.get(
            f"/api/v1/personnel/search/{nc}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert person_resp.status_code == 200
        person = person_resp.json()
        assert person["fname"] == f"person_{nc}"
        assert person["lname"] == ""
        # Should have 0 created_personnel (reused existing record)
        assert body["summary"]["total_persons"] == 0
    finally:
        _teardown(test_runtime, old_runtime)


# ═══════════════════════════════════════════════════════════════════
# Smoke test endpoint test
# ═══════════════════════════════════════════════════════════════════


def test_personnel_smoke_endpoint(tmp_path: Path) -> None:
    """Call the personnel smoke-test endpoint and verify every step passes."""
    test_runtime, old_runtime, client = _setup_client(tmp_path)
    try:
        resp = client.post("/api/v1/tests/personnel/smoke")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        summary = body.get("_summary", {})
        assert summary.get("failed", -1) == 0, (
            f"Smoke test had failures: {body}"
        )
        assert summary.get("passed", 0) >= 10, (
            f"Smoke test passed {summary.get('passed')} steps, expected >= 10"
        )
    finally:
        _teardown(test_runtime, old_runtime)
