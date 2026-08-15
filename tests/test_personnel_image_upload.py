"""Comprehensive tests for the batch personnel image upload endpoint.

POST /api/v1/personnel/{personnel_id}/images

Tests endpoint validation, batch behavior, authentication, face processing,
cleanup, primary-image policy, and response format.
"""

from __future__ import annotations

import io
import os
import struct
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.core.personnel_image_service import ImageProcessResult
from app.runtime import build_runtime


pytestmark = pytest.mark.usefixtures("postgres_database")


# ── Helpers ─────────────────────────────────────────────────────────


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

        saved_media_path=tmp_path / "saved_media",
        video_ingestion_enabled=False,
        media_preview_enabled=False,
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
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "admin123"},
    )
    return resp.json()["access_token"]


def _operator_token(client: TestClient) -> str:
    admin_token = _admin_token(client)
    client.post(
        "/api/v1/auth/users",
        json={"username": "operator1", "password": "operator123", "role": "operator"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": "operator1", "password": "operator123"},
    )
    return resp.json()["access_token"]


def _viewer_token(client: TestClient) -> str:
    admin_token = _admin_token(client)
    client.post(
        "/api/v1/auth/users",
        json={"username": "viewer1", "password": "viewer1234", "role": "viewer"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": "viewer1", "password": "viewer1234"},
    )
    return resp.json()["access_token"]


VALID_CODE_1 = "1234567891"
VALID_CODE_2 = "9876543210"


def _create_person(client, token: str) -> dict[str, Any]:
    resp = client.post(
        "/api/v1/personnel/",
        json={
            "fname": "Ali",
            "lname": "Mohammadi",
            "national_code": VALID_CODE_1,
            "employee_type": "employee",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    return resp.json()


def _valid_jpeg_bytes() -> bytes:
    """Return a small valid JPEG image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = (200, 100, 50)
    success, encoded = cv2.imencode(".jpg", img)
    assert success
    return encoded.tobytes()


def _valid_png_bytes() -> bytes:
    """Return a small valid PNG image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = (50, 200, 100)
    success, encoded = cv2.imencode(".png", img)
    assert success
    return encoded.tobytes()


def _upload(
    client: TestClient,
    token: str,
    person_id: int,
    files: list[tuple[str, bytes, str]],
    enable_cropping: str = "false",
) -> Any:
    """POST a batch upload request.

    files: list of (filename, data, content_type)
    """
    # Build files dict for httpx: files={"files": (name, data, type), ...}
    # For multiple files with the same field name, we need a list of tuples
    upload_files = []
    for filename, data, content_type in files:
        upload_files.append(
            ("files", (filename, data, content_type))
        )
    return client.post(
        f"/api/v1/personnel/{person_id}/images",
        files=upload_files,
        data={"enable_cropping": enable_cropping},
        headers={"Authorization": f"Bearer {token}"},
    )


def _upload_single(
    client: TestClient,
    token: str,
    person_id: int,
    filename: str = "test.jpg",
    data: bytes | None = None,
    content_type: str = "image/jpeg",
    enable_cropping: str = "false",
) -> Any:
    """Helper for single-file upload."""
    if data is None:
        data = _valid_jpeg_bytes()
    return _upload(client, token, person_id, [(filename, data, content_type)], enable_cropping)


# ═══════════════════════════════════════════════════════════════════
# Endpoint validation
# ═══════════════════════════════════════════════════════════════════


class TestEndpointValidation:
    def test_personnel_not_found(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            resp = _upload_single(client, token, 99999)
            assert resp.status_code == 404
        finally:
            _teardown(runtime, old_runtime)

    def test_no_files(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            resp = client.post(
                f"/api/v1/personnel/{person['id']}/images",
                data={"enable_cropping": "false"},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 422
            # FastAPI validation catches missing required field before handler runs
        finally:
            _teardown(runtime, old_runtime)

    def test_too_many_files(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            # Create more than max_images_per_request files
            max_files = settings.max_images_per_request
            files = [("test.jpg", _valid_jpeg_bytes(), "image/jpeg") for _ in range(max_files + 1)]
            resp = _upload(client, token, person["id"], files)
            assert resp.status_code == 422
            assert "Maximum" in resp.text
        finally:
            _teardown(runtime, old_runtime)

    def test_invalid_extension(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            resp = _upload_single(client, token, person["id"], filename="test.gif", data=_valid_jpeg_bytes(), content_type="image/gif")
            assert resp.status_code == 422
            assert "unsupported_extension" in resp.text.lower() or "پسوند تصویر پشتیبانی نمی‌شود" in resp.text
        finally:
            _teardown(runtime, old_runtime)


class TestVectorEnrollmentContract:
    def test_uses_created_image_id_as_vector_reference(self, tmp_path: Path, monkeypatch) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        references: list[str | int | None] = []

        class RecordingProcessor:
            def process_image(self, _data, *, person_name, ref_img_id, enable_cropping):
                references.append(ref_img_id)
                return ImageProcessResult(success=True, vector_point_id="point-1")

            def delete_vector(self, _point_id):
                return True

        monkeypatch.setattr("app.api.personnel._image_processor", lambda _runtime: RecordingProcessor())
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            response = _upload_single(client, token, person["id"])
            assert response.status_code == 201, response.text
            image_id = response.json()["results"][0]["image"]["id"]
            assert references == [str(image_id)]
            assert str(image_id) != str(person["id"])
        finally:
            _teardown(runtime, old_runtime)

    def test_non_success_status_removes_generated_image(self, tmp_path: Path, monkeypatch) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)

        class FailingProcessor:
            def process_image(self, _data, *, person_name, ref_img_id, enable_cropping):
                return ImageProcessResult(
                    success=False,
                    failure_code="enrollment_failed",
                    failure_message="vector rejected",
                )

            def delete_vector(self, _point_id):
                return True

        monkeypatch.setattr("app.api.personnel._image_processor", lambda _runtime: FailingProcessor())
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            response = _upload_single(client, token, person["id"])
            assert response.status_code == 422
            assert runtime.personnel_store.list_images(person["id"]) == []
            snapshot_dir = tmp_path / "saved_media" / "human" / "reference_images"
            assert not list(snapshot_dir.glob("*"))
        finally:
            _teardown(runtime, old_runtime)


class TestEndpointValidationContinued:
    def test_invalid_mime_type(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            resp = _upload_single(client, token, person["id"], content_type="text/plain")
            assert resp.status_code == 422
            result = resp.json()
            assert len(result.get("detail", {}).get("results", [])) > 0 or "پشتیبانی نمی‌شود" in resp.text
        finally:
            _teardown(runtime, old_runtime)

    def test_corrupt_image(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            corrupt = b"this is not an image file"
            resp = _upload_single(client, token, person["id"], data=corrupt, content_type="image/jpeg")
            assert resp.status_code == 422
        finally:
            _teardown(runtime, old_runtime)

    def test_path_traversal_filename(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            resp = _upload_single(client, token, person["id"], filename="../../etc/passwd.jpg")
            assert resp.status_code == 422
        finally:
            _teardown(runtime, old_runtime)

    def test_empty_file(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            resp = _upload_single(client, token, person["id"], data=b"", content_type="image/jpeg")
            assert resp.status_code == 422
        finally:
            _teardown(runtime, old_runtime)


# ═══════════════════════════════════════════════════════════════════
# Batch behavior
# ═══════════════════════════════════════════════════════════════════


class TestBatchBehavior:
    def test_single_successful_upload(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            resp = _upload_single(client, token, person["id"])
            assert resp.status_code == 201
            body = resp.json()
            assert body["total_success"] == 1
            assert body["total_failed"] == 0
            assert len(body["results"]) == 1
            assert body["results"][0]["success"] is True
            assert body["results"][0]["image"]["personnel_id"] == person["id"]
            assert body["results"][0]["image"]["is_primary"] is True
            assert len(body["personnel"]["images"]) >= 1
        finally:
            _teardown(runtime, old_runtime)

    def test_multiple_successful_uploads_in_one_request(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            files = [
                ("face1.jpg", _valid_jpeg_bytes(), "image/jpeg"),
                ("face2.png", _valid_png_bytes(), "image/png"),
            ]
            resp = _upload(client, token, person["id"], files)
            assert resp.status_code == 201
            body = resp.json()
            assert body["total_success"] == 2
            assert body["total_failed"] == 0
            assert len(body["results"]) == 2
            # First image should be primary
            assert body["results"][0]["image"]["is_primary"] is True
            assert body["results"][1]["image"]["is_primary"] is False
        finally:
            _teardown(runtime, old_runtime)

    def test_first_file_fails_later_succeeds(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            files = [
                ("corrupt.jpg", b"notanimage", "image/jpeg"),
                ("valid.jpg", _valid_jpeg_bytes(), "image/jpeg"),
            ]
            resp = _upload(client, token, person["id"], files)
            assert resp.status_code == 201
            body = resp.json()
            assert body["total_success"] == 1
            assert body["total_failed"] == 1
            assert body["results"][0]["success"] is False
            assert body["results"][1]["success"] is True
        finally:
            _teardown(runtime, old_runtime)

    def test_first_file_succeeds_later_fails(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            files = [
                ("valid.jpg", _valid_jpeg_bytes(), "image/jpeg"),
                ("corrupt.png", b"notanimage", "image/png"),
            ]
            resp = _upload(client, token, person["id"], files)
            assert resp.status_code == 201
            body = resp.json()
            assert body["total_success"] == 1
            assert body["total_failed"] == 1
            assert body["results"][0]["success"] is True
            assert body["results"][1]["success"] is False
        finally:
            _teardown(runtime, old_runtime)

    def test_all_validation_failures(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            files = [
                ("bad1.gif", _valid_jpeg_bytes(), "image/jpeg"),
                ("bad2.jpg", b"", "image/jpeg"),
            ]
            resp = _upload(client, token, person["id"], files)
            assert resp.status_code == 422  # all-failed returns 422
        finally:
            _teardown(runtime, old_runtime)

    def test_all_face_detection_failures_mock(self, tmp_path: Path) -> None:
        """In mock mode, face detection always succeeds so this test verifies
        the endpoint accepts valid images even without real face processing."""
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            resp = _upload_single(client, token, person["id"])
            assert resp.status_code == 201
            assert resp.json()["total_success"] == 1
        finally:
            _teardown(runtime, old_runtime)

    def test_mixed_valid_and_corrupt(self, tmp_path: Path) -> None:
        """Partial success: some valid, some corrupt."""
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            files = [
                ("good1.jpg", _valid_jpeg_bytes(), "image/jpeg"),
                ("bad.jpg", b"\x00\x01\x02corrupt", "image/jpeg"),
                ("good2.png", _valid_png_bytes(), "image/png"),
            ]
            resp = _upload(client, token, person["id"], files)
            assert resp.status_code == 201
            body = resp.json()
            assert body["total_success"] == 2
            assert body["total_failed"] == 1
        finally:
            _teardown(runtime, old_runtime)


# ═══════════════════════════════════════════════════════════════════
# Response format
# ═══════════════════════════════════════════════════════════════════


class TestResponse:
    def test_response_contains_personnel(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            resp = _upload_single(client, token, person["id"])
            body = resp.json()
            assert "personnel" in body
            p = body["personnel"]
            assert p["id"] == person["id"]
            assert p["fname"] == person["fname"]
            assert p["national_code"] == person["national_code"]
            assert "images" in p
        finally:
            _teardown(runtime, old_runtime)

    def test_response_contains_results(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            resp = _upload_single(client, token, person["id"])
            body = resp.json()
            assert "results" in body
            assert "total_success" in body
            assert "total_failed" in body
            assert isinstance(body["results"], list)
        finally:
            _teardown(runtime, old_runtime)

    def test_no_legacy_fields_in_response(self, tmp_path: Path) -> None:
        """Verify no Department, WorkShift, or other legacy fields."""
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            resp = _upload_single(client, token, person["id"])
            body = resp.json()
            p = body["personnel"]
            assert "department" not in p
            assert "department_id" not in p
            assert "department_name" in p
            assert "WorkShift" not in p
            assert "shift_id" not in p
            assert "Room" not in p
        finally:
            _teardown(runtime, old_runtime)


# ═══════════════════════════════════════════════════════════════════
# Primary image
# ═══════════════════════════════════════════════════════════════════


class TestPrimaryImage:
    def test_first_image_becomes_primary(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            resp = _upload_single(client, token, person["id"])
            assert resp.json()["results"][0]["image"]["is_primary"] is True
        finally:
            _teardown(runtime, old_runtime)

    def test_existing_primary_remains_primary(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            # Upload first image
            resp1 = _upload_single(client, token, person["id"], "first.jpg")
            first_id = resp1.json()["results"][0]["image"]["id"]
            # Upload second image in another batch
            resp2 = _upload_single(client, token, person["id"], "second.jpg")
            second_id = resp2.json()["results"][0]["image"]["id"]
            # First should still be primary, second not
            resp_get = client.get(
                f"/api/v1/personnel/images/{first_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp_get.json()["is_primary"] is True
            resp_get2 = client.get(
                f"/api/v1/personnel/images/{second_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp_get2.json()["is_primary"] is False
        finally:
            _teardown(runtime, old_runtime)

    def test_failed_first_image_does_not_become_primary(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            # First upload is corrupt (should fail)
            files = [
                ("corrupt.jpg", b"notanimage", "image/jpeg"),
                ("valid.jpg", _valid_jpeg_bytes(), "image/jpeg"),
            ]
            resp = _upload(client, token, person["id"], files)
            assert resp.status_code == 201
            body = resp.json()
            assert body["results"][0]["success"] is False
            assert body["results"][1]["success"] is True
            # The successful image should be primary
            assert body["results"][1]["image"]["is_primary"] is True
        finally:
            _teardown(runtime, old_runtime)

    def test_only_one_primary_after_multiple_uploads(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            # Upload 3 files in one request
            files = [
                ("a.jpg", _valid_jpeg_bytes(), "image/jpeg"),
                ("b.png", _valid_png_bytes(), "image/png"),
                ("c.jpg", _valid_jpeg_bytes(), "image/jpeg"),
            ]
            resp = _upload(client, token, person["id"], files)
            assert resp.status_code == 201
            body = resp.json()
            primary_count = sum(
                1 for r in body["results"] if r["success"] and r["image"]["is_primary"]
            )
            assert primary_count == 1
        finally:
            _teardown(runtime, old_runtime)


# ═══════════════════════════════════════════════════════════════════
# Authentication and permissions
# ═══════════════════════════════════════════════════════════════════


class TestAuth:
    def test_unauthenticated(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            resp = client.post(
                f"/api/v1/personnel/{person['id']}/images",
                files={"files": ("test.jpg", _valid_jpeg_bytes(), "image/jpeg")},
                data={"enable_cropping": "false"},
            )
            assert resp.status_code == 401
        finally:
            _teardown(runtime, old_runtime)

    def test_viewer_forbidden(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _viewer_token(client)
            person = _create_person(client, _admin_token(client))
            resp = _upload_single(client, token, person["id"])
            assert resp.status_code == 403
        finally:
            _teardown(runtime, old_runtime)

    def test_operator_forbidden(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _operator_token(client)
            person = _create_person(client, _admin_token(client))
            resp = _upload_single(client, token, person["id"])
            assert resp.status_code == 403
        finally:
            _teardown(runtime, old_runtime)

    def test_admin_authorized(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            resp = _upload_single(client, token, person["id"])
            assert resp.status_code == 201
        finally:
            _teardown(runtime, old_runtime)


# ═══════════════════════════════════════════════════════════════════
# Image validation details
# ═══════════════════════════════════════════════════════════════════


class TestImageValidation:
    def test_jpeg_accepted(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            resp = _upload_single(client, token, person["id"], "photo.jpg", _valid_jpeg_bytes(), "image/jpeg")
            assert resp.status_code == 201
        finally:
            _teardown(runtime, old_runtime)

    def test_png_accepted(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            resp = _upload_single(client, token, person["id"], "photo.png", _valid_png_bytes(), "image/png")
            assert resp.status_code == 201
        finally:
            _teardown(runtime, old_runtime)

    def test_oversized_file_rejected(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            # Create data larger than max_upload_bytes_per_image
            large_data = b"X" * (settings.max_upload_bytes_per_image + 1)
            resp = _upload_single(client, token, person["id"], data=large_data)
            assert resp.status_code == 422
        finally:
            _teardown(runtime, old_runtime)

    def test_mime_extension_mismatch(self, tmp_path: Path) -> None:
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            # PNG data with .jpg extension and image/png MIME should fail
            # Actually extension is .jpg, MIME is image/png -> mismatch
            resp = _upload_single(client, token, person["id"], "photo.jpg", _valid_png_bytes(), "image/png")
            assert resp.status_code == 422
        finally:
            _teardown(runtime, old_runtime)

    def test_storage_key_generated(self, tmp_path: Path) -> None:
        """Verify the storage key starts with human/reference_images/ and does not
        contain raw filename."""
        runtime, old_runtime, client = _setup_client(tmp_path)
        try:
            token = _admin_token(client)
            person = _create_person(client, token)
            resp = _upload_single(client, token, person["id"], "myphoto.jpg")
            body = resp.json()
            key = body["results"][0]["image"]["storage_key"]
            assert key.startswith("human/reference_images/")
            assert "myphoto" not in key  # UUID-based, not raw filename
        finally:
            _teardown(runtime, old_runtime)
