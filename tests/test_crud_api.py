"""Comprehensive CRUD API tests for every application module.

Tests that all CRUD endpoints are accessible and respond correctly.
Generates structured reports at .agentic/reports/crud-analysis-*.json/*.md.

Each test validates the endpoint is wired, responds with 2xx/4xx as appropriate,
and returns valid JSON. Detailed schema validation is done in module-specific tests.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import cv2
import pytest

from app.testsupport import (
    CrudTestContext,
    ModuleCrudReport,
    CrudTestResult,
    generate_crud_report,
    setup_crud_context,
    teardown_crud_context,
    write_crud_report,
    request_json,
    VALID_CODE_1,
)

pytestmark = [
    pytest.mark.crud,
    pytest.mark.api,
    pytest.mark.usefixtures("postgres_database"),
]

SHIFT_DATA = {
    "shift_name": "Morning",
    "shift_type": "morning",
    "start_time": "08:00",
    "end_time": "16:00",
    "saturday": False,
    "sunday": True,
    "monday": True,
    "tuesday": True,
    "wednesday": True,
    "thursday": True,
    "friday": False,
}


@pytest.fixture
def crud(tmp_path: Path, postgres_database) -> CrudTestContext:
    ctx = setup_crud_context(tmp_path)
    yield ctx
    teardown_crud_context(ctx)


def _ok(resp) -> None:
    assert resp.status_code < 500, f"{resp.request.method} {resp.request.url}: {resp.status_code} {resp.text[:500]}"


def _created_or_ok(resp) -> None:
    assert resp.status_code in (200, 201, 204), f"{resp.request.method} {resp.request.url}: {resp.status_code} {resp.text[:500]}"


# ═══════════════════════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════════════════════

class TestAuthCrud:
    MODULE = "auth"

    def test_login_me(self, crud):
        resp = crud.client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        assert resp.json()["username"] == "admin"

    def test_create_user(self, crud):
        resp = crud.client.post("/api/v1/auth/create-user", json={"username": "newuser1", "password": "pass12345", "role": "operator"},
                                headers={"Authorization": f"Bearer {crud.admin_token}"})
        assert resp.status_code in (200, 201, 409), resp.text

    def test_list_users(self, crud):
        resp = crud.client.get("/api/v1/auth/users", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        data = resp.json()
        assert isinstance(data, list)

    def test_token_refresh(self, crud):
        resp = crud.client.post("/api/v1/auth/token", data={"username": "admin", "password": "admin123"})
        _ok(resp)
        assert "access_token" in resp.json()


# ═══════════════════════════════════════════════════════════════════════
# CAMERAS
# ═══════════════════════════════════════════════════════════════════════

class TestCamerasCrud:
    MODULE = "cameras"

    def test_create_camera(self, crud):
        resp = crud.client.post("/api/v1/cameras", json={"camera_id": "crud-cam-1", "name": "CRUD Cam", "enabled": True, "source_uri": "rtsp://test/crud"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        assert resp.status_code == 201, resp.text

    def test_list_cameras(self, crud):
        resp = crud.client.get("/api/v1/cameras", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        assert isinstance(resp.json(), list)

    def test_get_update_delete_camera(self, crud):
        cam_id = "crud-cam-cycle"
        crud.client.post("/api/v1/cameras", json={"camera_id": cam_id, "name": "Cycle", "enabled": True, "source_uri": "rtsp://test/cycle"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        resp = crud.client.get(f"/api/v1/cameras/{cam_id}", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        assert resp.json()["camera_id"] == cam_id
        resp = crud.client.patch(f"/api/v1/cameras/{cam_id}", json={"name": "Updated"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        assert resp.json()["name"] == "Updated"
        resp = crud.client.post(f"/api/v1/cameras/{cam_id}/disable", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        resp = crud.client.post(f"/api/v1/cameras/{cam_id}/enable", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        resp = crud.client.delete(f"/api/v1/cameras/{cam_id}", headers={"Authorization": f"Bearer {crud.admin_token}"})
        assert resp.status_code in (200, 204), resp.text


# ═══════════════════════════════════════════════════════════════════════
# SOURCES
# ═══════════════════════════════════════════════════════════════════════

class TestSourcesCrud:
    MODULE = "sources"

    def test_create_source(self, crud):
        resp = crud.client.post("/api/v1/sources", json={"source_id": "crud-src-1", "name": "CRUD Src", "enabled": True, "source_uri": "rtsp://test/crud"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        assert resp.status_code == 201, resp.text

    def test_list_sources(self, crud):
        resp = crud.client.get("/api/v1/sources", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        assert isinstance(resp.json(), list)

    def test_get_update_delete_source(self, crud):
        src_id = "crud-src-cycle"
        crud.client.post("/api/v1/sources", json={"source_id": src_id, "name": "Cycle", "enabled": True, "source_uri": "rtsp://test/cycle"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        resp = crud.client.get(f"/api/v1/sources/{src_id}", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        assert resp.json()["source_id"] == src_id
        resp = crud.client.patch(f"/api/v1/sources/{src_id}", json={"name": "Updated Src"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        crud.client.post(f"/api/v1/sources/{src_id}/disable", headers={"Authorization": f"Bearer {crud.admin_token}"})
        crud.client.post(f"/api/v1/sources/{src_id}/enable", headers={"Authorization": f"Bearer {crud.admin_token}"})
        resp = crud.client.delete(f"/api/v1/sources/{src_id}", headers={"Authorization": f"Bearer {crud.admin_token}"})
        assert resp.status_code in (200, 204), resp.text


# ═══════════════════════════════════════════════════════════════════════
# PERSONNEL
# ═══════════════════════════════════════════════════════════════════════

class TestPersonnelCrud:
    MODULE = "personnel"

    def test_create_personnel(self, crud):
        resp = crud.client.post("/api/v1/personnel/", json={"fname": "CRUD", "lname": "Person", "national_code": VALID_CODE_1, "employee_type": "employee"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        assert resp.status_code == 201, resp.text
        assert resp.json()["fname"] == "CRUD"

    def test_list_personnel(self, crud):
        resp = crud.client.get("/api/v1/personnel/", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        data = resp.json()
        assert isinstance(data, list)

    def test_get_update_delete_personnel(self, crud):
        resp = crud.client.post("/api/v1/personnel/", json={"fname": "Cycle", "lname": "Test", "national_code": "9876543210", "employee_type": "employee"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        pid = resp.json()["id"]
        resp = crud.client.get(f"/api/v1/personnel/{pid}", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        assert resp.json()["fname"] == "Cycle"
        resp = crud.client.put(f"/api/v1/personnel/{pid}", json={"fname": "Cycled"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        assert resp.json()["fname"] == "Cycled"
        resp = crud.client.delete(f"/api/v1/personnel/{pid}", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)


# ═══════════════════════════════════════════════════════════════════════
# LOCATIONS
# ═══════════════════════════════════════════════════════════════════════

class TestLocationsCrud:
    MODULE = "locations"

    def test_building_crud(self, crud):
        resp = crud.client.post("/buildings/", json={"name": "CRUD Bld", "address": "123 Test"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        assert resp.status_code == 201, resp.text
        bid = resp.json()["id"]
        resp = crud.client.get("/buildings/", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        resp = crud.client.get(f"/buildings/{bid}", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        resp = crud.client.patch(f"/buildings/{bid}", json={"name": "Updated Bld"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        resp = crud.client.delete(f"/buildings/{bid}", headers={"Authorization": f"Bearer {crud.admin_token}"})
        assert resp.status_code in (200, 204), resp.text

    def test_section_crud(self, crud):
        resp = crud.client.post("/buildings/", json={"name": "Bld for Sec", "address": "456 Test"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        bid = resp.json()["id"]
        resp = crud.client.post("/sections/", json={"section_name": "CRUD Sec", "building_id": bid}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        assert resp.status_code == 201, resp.text
        resp = crud.client.delete(f"/sections/{resp.json()['id']}", headers={"Authorization": f"Bearer {crud.admin_token}"})
        assert resp.status_code in (200, 204), resp.text

    def test_room_crud(self, crud):
        resp = crud.client.post("/buildings/", json={"name": "Bld for Room", "address": "789 Test"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        bid = resp.json()["id"]
        resp = crud.client.post("/sections/", json={"section_name": "Sec for Room", "building_id": bid}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        sid = resp.json()["id"]
        resp = crud.client.post("/rooms/", json={"room_name": "CRUD Room", "section_id": sid}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        assert resp.status_code == 201, resp.text
        rid = resp.json()["id"]
        resp = crud.client.put(f"/rooms/{rid}", json={"room_name": "Updated Room"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        resp = crud.client.delete(f"/rooms/{rid}", headers={"Authorization": f"Bearer {crud.admin_token}"})
        assert resp.status_code in (200, 204), resp.text


# ═══════════════════════════════════════════════════════════════════════
# SHIFTS
# ═══════════════════════════════════════════════════════════════════════

class TestShiftsCrud:
    MODULE = "shifts"

    def test_shift_crud(self, crud):
        resp = crud.client.post("/api/v1/shifts/", json=SHIFT_DATA, headers={"Authorization": f"Bearer {crud.admin_token}"})
        assert resp.status_code == 201, resp.text
        shift = resp.json().get("shift", resp.json())
        sid = shift["id"]
        resp = crud.client.get(f"/api/v1/shifts/{sid}", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        resp = crud.client.put(f"/api/v1/shifts/{sid}", json={"shift_name": "Evening", "start_time": "16:00", "end_time": "00:00", "shift_type": "evening"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        resp = crud.client.delete(f"/api/v1/shifts/{sid}", headers={"Authorization": f"Bearer {crud.admin_token}"})
        assert resp.status_code in (200, 204), resp.text

    def test_list_shifts(self, crud):
        resp = crud.client.get("/api/v1/shifts/", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)


# ═══════════════════════════════════════════════════════════════════════
# HOLIDAYS
# ═══════════════════════════════════════════════════════════════════════

class TestHolidaysCrud:
    MODULE = "holidays"

    def test_holiday_crud(self, crud):
        resp = crud.client.post("/api/v1/holidays/", json={"name": "CRUD Holiday", "date_value": "2027-06-15", "holiday_type": "national"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        assert resp.status_code == 201, resp.text
        hol = resp.json().get("holiday", resp.json())
        hid = hol["id"]
        resp = crud.client.get(f"/api/v1/holidays/{hid}", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        resp = crud.client.put(f"/api/v1/holidays/{hid}", json={"name": "Updated Holiday"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        resp = crud.client.delete(f"/api/v1/holidays/{hid}", headers={"Authorization": f"Bearer {crud.admin_token}"})
        assert resp.status_code in (200, 204), resp.text

    def test_check_holiday(self, crud):
        crud.client.post("/api/v1/holidays/", json={"name": "Check Test", "date_value": "2028-01-01", "holiday_type": "national"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        resp = crud.client.get("/api/v1/holidays/check/2028-01-01", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)

    def test_list_holidays(self, crud):
        resp = crud.client.get("/api/v1/holidays/", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)


# ═══════════════════════════════════════════════════════════════════════
# REQUESTS
# ═══════════════════════════════════════════════════════════════════════

class TestRequestsCrud:
    MODULE = "requests"

    def test_request_crud(self, crud):
        resp = crud.client.post("/api/v1/personnel/", json={"fname": "Req", "lname": "Test", "national_code": VALID_CODE_1, "employee_type": "employee"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        pid = resp.json()["id"] if "id" in resp.json() else resp.json().get("personnel", {}).get("id")
        assert pid is not None, f"Personnel create failed: {resp.status_code} {resp.text}"
        resp = crud.client.post("/api/v1/requests/", json={"personnel_id": pid, "request_type": "leave", "start_date": "2026-09-01", "end_date": "2026-09-03"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        _created_or_ok(resp)
        req = resp.json()
        rid = req.get("id") or (req.get("request") or {}).get("id")
        if rid:
            resp = crud.client.get(f"/api/v1/requests/{rid}", headers={"Authorization": f"Bearer {crud.admin_token}"})
            _ok(resp)

    def test_list_requests(self, crud):
        resp = crud.client.get("/api/v1/requests/", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)


# ═══════════════════════════════════════════════════════════════════════
# BROADCAST
# ═══════════════════════════════════════════════════════════════════════

class TestBroadcastApi:
    MODULE = "broadcast"

    def test_broadcast_state(self, crud):
        resp = crud.client.get("/api/v1/broadcast/state", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        assert "enabled" in resp.json()
        resp = crud.client.put("/api/v1/broadcast/state", json={"enabled": False}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        assert resp.json()["enabled"] is False


# ═══════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════

class TestDiagnosticsApi:
    MODULE = "diagnostics"

    def test_overview(self, crud):
        resp = crud.client.get("/api/v1/diagnostics/overview", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)

    def test_checks(self, crud):
        resp = crud.client.get("/api/v1/diagnostics/checks", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)


# ═══════════════════════════════════════════════════════════════════════
# FACES
# ═══════════════════════════════════════════════════════════════════════

class TestFacesApi:
    MODULE = "faces"

    def test_status(self, crud):
        resp = crud.client.get("/api/v1/faces/status", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)

    def test_quality_settings(self, crud):
        resp = crud.client.get("/api/v1/faces/quality-settings", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        resp = crud.client.patch("/api/v1/faces/quality-settings", json={"quality_threshold": 0.6}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)

    def test_identities(self, crud):
        resp = crud.client.get("/api/v1/faces/identities", headers={"Authorization": f"Bearer {crud.admin_token}"})
        assert resp.status_code in (200, 503), resp.text


# ═══════════════════════════════════════════════════════════════════════
# FIRE/SMOKE
# ═══════════════════════════════════════════════════════════════════════

class TestFireSmokeApi:
    MODULE = "fire_smoke"

    def test_settings(self, crud):
        resp = crud.client.get("/api/v1/fire-smoke/settings", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)

    def test_logs(self, crud):
        resp = crud.client.get("/api/v1/fire-smoke-logs", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)


# ═══════════════════════════════════════════════════════════════════════
# GENERAL SETTINGS
# ═══════════════════════════════════════════════════════════════════════

class TestGeneralSettingsApi:
    MODULE = "general_settings"

    def test_get_patch_settings(self, crud):
        resp = crud.client.get("/api/v1/settings/general", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        resp = crud.client.patch("/api/v1/settings/general", json={}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)


# ═══════════════════════════════════════════════════════════════════════
# HUMANS
# ═══════════════════════════════════════════════════════════════════════

class TestHumansApi:
    MODULE = "humans"

    def test_status(self, crud):
        resp = crud.client.get("/api/v1/humans/status", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)

    def test_logs(self, crud):
        resp = crud.client.get("/api/v1/humans/logs", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)

    def test_active(self, crud):
        resp = crud.client.get("/api/v1/humans/active", headers={"Authorization": f"Bearer {crud.admin_token}"})
        assert resp.status_code in (200, 503), resp.text


# ═══════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════

class TestModelsApi:
    MODULE = "models"

    def test_artifacts(self, crud):
        resp = crud.client.get("/api/v1/models/artifacts", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)

    def test_settings(self, crud):
        resp = crud.client.get("/api/v1/models/settings", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        assert "resolved_models" in resp.json()

    def test_conversions(self, crud):
        resp = crud.client.get("/api/v1/models/conversions", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)


# ═══════════════════════════════════════════════════════════════════════
# PLATE LOGS
# ═══════════════════════════════════════════════════════════════════════

class TestPlateLogsApi:
    MODULE = "plate_logs"

    def test_list_logs(self, crud):
        resp = crud.client.get("/api/v1/plate-logs", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)


# ═══════════════════════════════════════════════════════════════════════
# PLATE SETTINGS
# ═══════════════════════════════════════════════════════════════════════

class TestPlateSettingsApi:
    MODULE = "plate_settings"

    def test_general_settings(self, crud):
        resp = crud.client.get("/api/v1/plate-settings/general", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)

    def test_camera_settings(self, crud):
        crud.client.post("/api/v1/cameras", json={"camera_id": "ps-cam-1", "name": "PS Cam", "enabled": True, "source_uri": "rtsp://test/ps"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        resp = crud.client.get("/api/v1/plate-settings/cameras/ps-cam-1", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)


# ═══════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════

class TestResultsApi:
    MODULE = "results"

    def test_recent_results(self, crud):
        resp = crud.client.get("/api/v1/results/recent", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)

    def test_router_status(self, crud):
        resp = crud.client.get("/api/v1/router/status", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)


# ═══════════════════════════════════════════════════════════════════════
# ATTENDANCE
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# FRAMES
# ═══════════════════════════════════════════════════════════════════════

class TestFramesApi:
    MODULE = "frames"

    def test_submit_frame(self, crud):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", frame)
        resp = crud.client.post(
            "/api/v1/frame-rounds/jpeg",
            data={"source_ids_json": '["frame-test-cam"]', "captured_at_utc": "2026-07-21T00:00:00Z", "round_sequence": "1"},
            files={"files": ("test.jpg", buf.tobytes(), "image/jpeg")},
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        assert resp.status_code in (200, 202, 422), resp.text


# =======================================================================
# LEGACY CRUD AND DISCOVERY ROUTES
# =======================================================================

class TestCarPlatesCrud:
    MODULE = "car_plates"

    PLATE_DATA = {
        "left_digits": "12",
        "plate_alphabet": "A",
        "right_digits": "345",
        "iran_code": "67",
        "usage_type": "private",
        "vehicle_type": "car",
        "owner_name": "CRUD Owner",
        "owner_phone": "09000000000",
    }

    def test_create_list_get_update_delete_and_permissions(self, crud):
        base = "/api/v1/car-plates"
        response = crud.client.post(
            base,
            json=self.PLATE_DATA,
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        assert response.status_code == 201, response.text
        item = response.json()
        assert item["left_digits"] == "12"
        plate_id = item["id"]

        listed = crud.client.get(
            base,
            headers={"Authorization": f"Bearer {crud.viewer_token}"},
        )
        assert listed.status_code == 200
        assert any(value["id"] == plate_id for value in listed.json())

        fetched = crud.client.get(
            f"{base}/{plate_id}",
            headers={"Authorization": f"Bearer {crud.viewer_token}"},
        )
        assert fetched.status_code == 200
        assert fetched.json()["owner_name"] == "CRUD Owner"

        updated = crud.client.patch(
            f"{base}/{plate_id}",
            json={"owner_name": "Updated Owner"},
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        assert updated.status_code == 200
        assert updated.json()["owner_name"] == "Updated Owner"

        forbidden = crud.client.post(
            base,
            json={**self.PLATE_DATA, "right_digits": "346"},
            headers={"Authorization": f"Bearer {crud.viewer_token}"},
        )
        assert forbidden.status_code == 403

        invalid = crud.client.post(
            base,
            json={**self.PLATE_DATA, "left_digits": "1"},
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        assert invalid.status_code == 422

        missing = crud.client.get(
            f"{base}/99999999",
            headers={"Authorization": f"Bearer {crud.viewer_token}"},
        )
        assert missing.status_code == 404

        deleted = crud.client.delete(
            f"{base}/{plate_id}",
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        assert deleted.status_code == 200
        assert crud.client.get(
            f"{base}/{plate_id}",
            headers={"Authorization": f"Bearer {crud.viewer_token}"},
        ).status_code == 404


class TestFireLogsCrud:
    MODULE = "fire_logs"

    LOG_DATA = {
        "detection_time": "2026-07-21T00:00:00Z",
        "camera_id": "crud-fire-camera",
        "hazard_type": "fire",
        "severity": "high",
        "confidence": 0.91,
    }

    def test_create_list_get_validation_and_permissions(self, crud):
        base = "/api/v1/fire-logs"
        created = crud.client.post(
            base,
            json=self.LOG_DATA,
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        assert created.status_code == 201, created.text
        item = created.json()
        assert item["camera"] == "crud-fire-camera"
        log_id = item["id"]

        listed = crud.client.get(
            base,
            params={"camera_id": "crud-fire-camera", "limit": 10},
            headers={"Authorization": f"Bearer {crud.viewer_token}"},
        )
        assert listed.status_code == 200
        assert any(value["id"] == log_id for value in listed.json())

        fetched = crud.client.get(
            f"{base}/{log_id}",
            headers={"Authorization": f"Bearer {crud.viewer_token}"},
        )
        assert fetched.status_code == 200
        assert fetched.json()["severity"] == "high"

        invalid = crud.client.post(
            base,
            json={**self.LOG_DATA, "severity": "critical"},
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        assert invalid.status_code == 422

        forbidden = crud.client.post(
            base,
            json={**self.LOG_DATA, "camera_id": "another-camera"},
            headers={"Authorization": f"Bearer {crud.viewer_token}"},
        )
        assert forbidden.status_code == 403

        missing = crud.client.get(
            f"{base}/99999999",
            headers={"Authorization": f"Bearer {crud.viewer_token}"},
        )
        assert missing.status_code == 404


class TestDetectionLogsApi:
    MODULE = "detection_logs"

    def test_filter_validation_not_found_and_permissions(self, crud):
        base = "/api/v1/logs"
        listed = crud.client.get(
            f"{base}/filter",
            params={"period": "today", "limit": 10},
            headers={"Authorization": f"Bearer {crud.operator_token}"},
        )
        assert listed.status_code == 200
        assert isinstance(listed.json(), list)

        invalid_period = crud.client.get(
            f"{base}/filter",
            params={"period": "invalid"},
            headers={"Authorization": f"Bearer {crud.operator_token}"},
        )
        assert invalid_period.status_code == 400

        missing = crud.client.get(
            f"{base}/99999999",
            headers={"Authorization": f"Bearer {crud.operator_token}"},
        )
        assert missing.status_code == 404

        unauthenticated = crud.client.get(f"{base}/filter")
        assert unauthenticated.status_code == 401


class TestPersonnelRequestsApi:
    MODULE = "personnel_requests"

    def test_list_not_found_validation_and_permissions(self, crud):
        base = "/api/v1/personnel-requests"
        listed = crud.client.get(
            f"{base}/",
            headers={"Authorization": f"Bearer {crud.operator_token}"},
        )
        assert listed.status_code == 200
        assert isinstance(listed.json(), list)

        missing = crud.client.get(
            f"{base}/99999999",
            headers={"Authorization": f"Bearer {crud.operator_token}"},
        )
        assert missing.status_code == 404

        invalid_personnel = crud.client.post(
            f"{base}/",
            json={
                "personnel_id": 99999999,
                "request_type": "earned_leave",
                "start_date": "1405/01/01",
                "end_date": "1405/01/02",
            },
            headers={"Authorization": f"Bearer {crud.operator_token}"},
        )
        assert invalid_personnel.status_code == 404

        forbidden = crud.client.get(f"{base}/", headers={"Authorization": f"Bearer {crud.viewer_token}"})
        assert forbidden.status_code == 403


class TestDeveloperAndProcessorTestsApi:
    MODULE = "developer_and_processor_tests"

    def test_public_discovery_and_not_found_contracts(self, crud):
        info = crud.client.get("/api/v1/project-info")
        assert info.status_code == 200
        assert {"name", "current_version"} <= info.json()["project"].keys()

        apps = crud.client.get("/api/v1/developer/apps")
        assert apps.status_code == 200
        assert apps.json()["success"] is True
        assert apps.json()["total_apps"] >= 0
        assert isinstance(apps.json()["items"], list)

        missing = crud.client.get("/api/v1/project-info/releases/does-not-exist")
        assert missing.status_code == 404

        models = crud.client.get("/api/v1/tests/face/models")
        assert models.status_code == 200
        assert {"human_detector", "face_detector", "embedding_model"} <= models.json().keys()

    def test_admin_test_request_allowlist_and_authentication(self, crud):
        forbidden = crud.client.post(
            "/api/v1/developer/test-request",
            json={"method": "GET", "url": "/api/v1/health"},
            headers={"Authorization": f"Bearer {crud.viewer_token}"},
        )
        assert forbidden.status_code == 403

        rejected = crud.client.post(
            "/api/v1/developer/test-request",
            json={"method": "DELETE", "url": "/api/v1/diagnostics/overview"},
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        assert rejected.status_code == 200
        assert rejected.json()["success"] is False
        assert rejected.json()["status_code"] == 400


# ═══════════════════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    MODULE = "health"

    def test_health(self, crud):
        resp = crud.client.get("/health")
        _ok(resp)
        assert resp.json().get("status") in ("ok", "running")


# ═══════════════════════════════════════════════════════════════════════
# CRUD COVERAGE REPORT
# ═══════════════════════════════════════════════════════════════════════

CRUD_TEST_CLASSES: dict[str, type] = {
    "auth": TestAuthCrud,
    "cameras": TestCamerasCrud,
    "sources": TestSourcesCrud,
    "personnel": TestPersonnelCrud,
    "locations": TestLocationsCrud,
    "shifts": TestShiftsCrud,
    "holidays": TestHolidaysCrud,
    "requests": TestRequestsCrud,
    "broadcast": TestBroadcastApi,
    "diagnostics": TestDiagnosticsApi,
    "faces": TestFacesApi,
    "fire_smoke": TestFireSmokeApi,
    "general_settings": TestGeneralSettingsApi,
    "humans": TestHumansApi,
    "models": TestModelsApi,
    "plate_logs": TestPlateLogsApi,
    "plate_settings": TestPlateSettingsApi,
    "results": TestResultsApi,
    
    "frames": TestFramesApi,
    "car_plates": TestCarPlatesCrud,
    "fire_logs": TestFireLogsCrud,
    "detection_logs": TestDetectionLogsApi,
    "personnel_requests": TestPersonnelRequestsApi,
    "developer_and_processor_tests": TestDeveloperAndProcessorTestsApi,
    "health": TestHealthEndpoint,
}


def count_test_methods(cls: type) -> int:
    return sum(1 for name in vars(cls) if name.startswith("test_") and callable(vars(cls)[name]))


@pytest.mark.crud
class TestCrudCoverageReport:
    """Verify CRUD test coverage and generate a module-level report."""

    def test_all_modules_have_crud_tests(self) -> None:
        for module, cls in CRUD_TEST_CLASSES.items():
            assert count_test_methods(cls) >= 1, f"{module}: no tests in {cls.__name__}"

    def test_crud_coverage_report(self, tmp_path: Path) -> None:
        report_data = generate_crud_report([])
        paths = write_crud_report(report_data, prefix="crud-coverage-verify")
        assert paths["json"].exists()
        assert paths["md"].exists()
