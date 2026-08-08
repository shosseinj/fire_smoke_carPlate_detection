"""Comprehensive CRUD API tests for every application module.

Tests that all CRUD endpoints are accessible and respond correctly.
Generates structured reports at .agentic/reports/crud-analysis-*.json/*.md.

Each test validates the endpoint is wired, responds with 2xx/4xx as appropriate,
and returns valid JSON. Detailed schema validation is done in module-specific tests.
"""
from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import cv2
import pytest

from app.config import settings
from app.api import detection_logs as detection_logs_api
from app.core.jalali_utils import parse_jalali_date

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


def _parse_iso(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


# ═══════════════════════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════════════════════

class TestAuthCrud:
    MODULE = "auth"

    def test_login_valid(self, crud):
        resp = crud.client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert "refresh_token" in body
        assert body["token_type"] == "bearer"
        assert body["role"] in ("admin", "superadmin")

    def test_login_invalid(self, crud):
        resp = crud.client.post("/api/v1/auth/login", json={"username": "admin", "password": "wrong"})
        assert resp.status_code == 401

    def test_token_oauth2(self, crud):
        resp = crud.client.post("/api/v1/auth/token", data={"username": "admin", "password": "admin123"})
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert "refresh_token" in body
        assert body["token_type"] == "bearer"

    def test_refresh_valid(self, crud):
        login_resp = crud.client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
        refresh_token = login_resp.json()["refresh_token"]
        resp = crud.client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_refresh_invalid(self, crud):
        resp = crud.client.post("/api/v1/auth/refresh", json={"refresh_token": "invalid-token"})
        assert resp.status_code == 401

    def test_logout(self, crud):
        login_resp = crud.client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
        access_token = login_resp.json()["access_token"]
        resp = crud.client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert resp.status_code == 200
        # The revoked access token no longer authenticates
        me = crud.client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {access_token}"})
        assert me.status_code == 401

    def test_me(self, crud):
        resp = crud.client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {crud.admin_token}"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["username"] == "admin"
        assert body["role"] in ("admin", "superadmin")
        assert body["is_active"] is True
        assert "id" in body

    def test_roles(self, crud):
        resp = crud.client.get("/api/v1/auth/roles")
        assert resp.status_code == 200
        data = resp.json()
        assert "roles" in data
        assert len(data["roles"]) >= 3

    def test_create_user(self, crud):
        resp = crud.client.post(
            "/api/v1/auth/create-user",
            json={"username": "cruduser1", "password": "StrongPass1!", "email": "crud1@test.local", "confirm_password": "StrongPass1!", "role": "user"},
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        assert resp.status_code in (200, 201)
        body = resp.json()
        assert body["username"] == "cruduser1"
        assert body["role"] == "user"

    def test_create_user_validation_error(self, crud):
        resp = crud.client.post(
            "/api/v1/auth/create-user",
            json={"username": "x", "password": "short", "email": "", "confirm_password": ""},
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        assert resp.status_code == 422

    def test_create_user_duplicate(self, crud):
        resp = crud.client.post(
            "/api/v1/auth/create-user",
            json={"username": "admin", "password": "StrongPass1!", "email": "dup@test.local", "confirm_password": "StrongPass1!", "role": "user"},
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        assert resp.status_code == 400

    def test_list_users(self, crud):
        resp = crud.client.get("/api/v1/auth/users", headers={"Authorization": f"Bearer {crud.admin_token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_list_users_forbidden(self, crud):
        resp = crud.client.get("/api/v1/auth/users", headers={"Authorization": f"Bearer {crud.viewer_token}"})
        assert resp.status_code == 403

    def test_change_password(self, crud):
        resp = crud.client.post(
            "/api/v1/auth/me/password",
            json={"current_password": "admin123", "new_password": "NewPass123!", "confirm_password": "NewPass123!"},
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
# CAMERAS
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# SOURCES
# ═══════════════════════════════════════════════════════════════════════

class TestSourcesCrud:
    MODULE = "sources"

    def test_create_source(self, crud):
        resp = crud.client.post(
            "/api/v1/sources",
            json={"source_uri": "crud-src-1", "name": "CRUD Src", "enabled": True},
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["source_uri"] == "crud-src-1"
        assert resp.json()["name"] == "CRUD Src"

    def test_list_sources(self, crud):
        resp = crud.client.get("/api/v1/sources", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        assert isinstance(resp.json(), list)

    def test_get_update_delete_source(self, crud):
        src_id = "crud-src-cycle"
        crud.client.post(
            "/api/v1/sources",
            json={"source_uri": src_id, "name": "Cycle", "enabled": True},
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        resp = crud.client.get(f"/api/v1/sources/{src_id}", headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        assert resp.json()["source_uri"] == src_id
        resp = crud.client.patch(f"/api/v1/sources/{src_id}", json={"name": "Updated Src"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
        _ok(resp)
        assert resp.json()["name"] == "Updated Src"
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
        cam = crud.client.post(
            "/api/v1/cams",
            json={
                "camera_name": "Room Cam",
                "camera_number": 1,
                "width": 640,
                "high": 480,
                "source_type": "rtsp",
                "section_id": sid,
                "url": "rtsp://test-room-cam.local/live",
            },
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        assert cam.status_code == 201, cam.text
        cam_id = cam.json()["id"]
        resp = crud.client.post("/rooms/", json={"room_name": "CRUD Room", "section_id": sid, "camera_id": cam_id}, headers={"Authorization": f"Bearer {crud.admin_token}"})
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
        resp = crud.client.post("/api/v1/holidays/", json={"name": "CRUD Holiday", "date": "2027-06-15", "holiday_type": "national"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
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
        crud.client.post("/api/v1/holidays/", json={"name": "Check Test", "date": "2028-01-01", "holiday_type": "national"}, headers={"Authorization": f"Bearer {crud.admin_token}"})
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
        resp = crud.client.patch(
            "/api/v1/faces/quality-settings",
            json={
                "quality_threshold": 0.6,
                "human_pose_min_keypoints": 5,
                "recognition_quality_weight": 0.75,
            },
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        _ok(resp)
        body = resp.json()
        assert body["human_pose_min_keypoints"] == 5
        assert body["recognition_quality_weight"] == 0.75

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
        crud.client.post(
            "/api/v1/sources",
            json={"source_uri": "ps-cam-1", "name": "PS Cam", "enabled": True},
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
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
        "plate_alphabet": "ب",
        "right_digits": "345",
        "iran_code": "67",
        "usage_type": "personal",
        "vehicle_type": "sedan",
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
        "fire_confidence": 0.91,
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
        assert item["camera_id"] == "crud-fire-camera"
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

    def test_manual_api_and_excel_import_force_confidence_to_one(self, crud):
        auth = {"Authorization": f"Bearer {crud.operator_token}"}
        api_response = crud.client.post(
            "/api/v1/logs/log",
            json={"person": "Unknown", "confidence": 0.23},
            headers=auth,
        )
        assert api_response.status_code == 200, api_response.text
        assert api_response.json()["confidence"] == 1.0

        _, national_code = self._first_personnel(crud)
        workbook = detection_logs_api.openpyxl.Workbook()
        worksheet = workbook.active
        worksheet.append(
            [
                "national_code",
                "year",
                "month",
                "day",
                "hour",
                "minute",
                "room_id",
                "access_granted",
                "counts_for_attendance",
            ]
        )
        worksheet.append([national_code, 1405, 2, 1, 10, 15, None, 1, 1])
        content = BytesIO()
        workbook.save(content)

        excel_response = crud.client.post(
            "/api/v1/logs/import-excel",
            files={
                "file": (
                    "detection-logs.xlsx",
                    content.getvalue(),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
            headers=auth,
        )
        assert excel_response.status_code == 202, excel_response.text
        status_url = excel_response.json()["status_url"]
        deadline = time.monotonic() + 5
        progress = None
        while time.monotonic() < deadline:
            progress_response = crud.client.get(status_url, headers=auth)
            assert progress_response.status_code == 200
            progress = progress_response.json()
            if progress["status"] not in {"queued", "running"}:
                break
            time.sleep(0.01)
        assert progress is not None and progress["status"] == "completed"
        assert progress["result"]["imported_rows"] == 1
        imported = [
            record
            for record in detection_logs_api.get_detection_log_store().list_all()
            if record.source_system == "excel_import"
        ]
        assert imported
        assert imported[-1].confidence == 1.0

    def _first_log_id(self, crud) -> int:
        resp = crud.client.post(
            "/api/v1/logs/log",
            json={"person": "Unknown"},
            headers={"Authorization": f"Bearer {crud.operator_token}"},
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["id"]

    def _first_personnel(self, crud) -> tuple[int, str]:
        resp = crud.client.get(
            "/api/v1/personnel/",
            params={"limit": 1},
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        assert resp.status_code == 200, resp.text
        items = resp.json()
        assert items, "No personnel found"
        return items[0]["id"], items[0]["national_code"]

    def test_filter_validation_not_found_and_permissions(self, crud):
        base = "/api/v1/logs"
        listed = crud.client.get(
            f"{base}/filter",
            params={"period": "all", "limit": 10},
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

    def test_generate_fake_does_not_expose_count_parameter(self, crud):
        operation = crud.client.get("/openapi.json").json()["paths"][
            "/api/v1/logs/generate-fake"
        ]["post"]
        parameter_names = {parameter["name"] for parameter in operation["parameters"]}
        assert "count" not in parameter_names

    def test_generate_fake_single_day_pair_window_and_removal(self, crud):
        base = "/api/v1/logs/generate-fake"
        auth = {"Authorization": f"Bearer {crud.admin_token}"}
        day = "1400-01-15"

        first = crud.client.post(
            base,
            params={
                "from_date": day,
                "to_date": day,
                "pair_logs": True,
                "remove_existing": True,
            },
            headers=auth,
        )
        assert first.status_code == 201

        local_tz = ZoneInfo(settings.business_timezone_name)
        expected_date = parse_jalali_date(day)
        personnel, _ = detection_logs_api.get_personnel_store().list(limit=1000)
        generated = [
            record
            for record in detection_logs_api.get_detection_log_store().list_all()
            if record.source_system == "generate_fake"
            and _parse_iso(record.detection_time).astimezone(local_tz).date()
            == expected_date
        ]
        assert first.json()["count"] == len(personnel) * 2
        assert len(generated) == len(personnel) * 2
        assert all(record.counts_for_attendance for record in generated)
        for person in personnel:
            person_logs = sorted(
                (record for record in generated if record.personnel_id == person.id),
                key=lambda record: record.detection_time,
            )
            assert len(person_logs) == 2
            first_time, second_time = (
                _parse_iso(record.detection_time).astimezone(local_tz)
                for record in person_logs
            )
            assert (6, 0) <= (first_time.hour, first_time.minute) <= (9, 0)
            assert (14, 0) <= (second_time.hour, second_time.minute) <= (18, 0)

        replacement = crud.client.post(
            base,
            params={
                "from_date": day,
                "to_date": day,
                "remove_existing": True,
                "pair_logs": True,
            },
            headers=auth,
        )
        assert replacement.status_code == 201
        assert replacement.json()["deleted_count"] == len(generated)
        assert replacement.json()["count"] == len(personnel) * 2

    def test_generate_fake_unpaired_creates_zero_to_five_per_person_per_day(
        self, crud, monkeypatch
    ):
        base = "/api/v1/logs/generate-fake"
        auth = {"Authorization": f"Bearer {crud.admin_token}"}
        day = "1400-02-20"

        original_randint = random.randint

        def deterministic_randint(start: int, end: int) -> int:
            if (start, end) == (0, 5):
                return 3
            return original_randint(start, end)

        monkeypatch.setattr(random, "randint", deterministic_randint)
        response = crud.client.post(
            base,
            params={
                "from_date": day,
                "to_date": day,
                "pair_logs": False,
                "remove_existing": True,
            },
            headers=auth,
        )
        assert response.status_code == 201

        local_tz = ZoneInfo(settings.business_timezone_name)
        expected_date = parse_jalali_date(day)
        personnel, _ = detection_logs_api.get_personnel_store().list(limit=1000)
        generated = [
            record
            for record in detection_logs_api.get_detection_log_store().list_all()
            if record.source_system == "generate_fake"
            and _parse_iso(record.detection_time).astimezone(local_tz).date()
            == expected_date
        ]
        assert response.json()["count"] == len(personnel) * 3
        assert all(record.counts_for_attendance for record in generated)
        for person in personnel:
            person_logs = [
                record for record in generated if record.personnel_id == person.id
            ]
            assert len(person_logs) == 3
            for record in person_logs:
                local_time = _parse_iso(record.detection_time).astimezone(local_tz)
                assert (7, 0) <= (local_time.hour, local_time.minute) <= (18, 0)

    def test_generate_fake_validates_date_arguments(self, crud):
        base = "/api/v1/logs/generate-fake"
        auth = {"Authorization": f"Bearer {crud.admin_token}"}
        cases = (
            {"from_date": "1400-01-01"},
            {"remove_existing": True},
            {"from_date": "1400-01-02", "to_date": "1400-01-01"},
        )
        for params in cases:
            response = crud.client.post(base, params=params, headers=auth)
            assert response.status_code == 400

    def test_patch_log_person_by_national_code(self, crud):
        base = "/api/v1/logs"
        log_id = self._first_log_id(crud)
        pid, national_code = self._first_personnel(crud)

        resp = crud.client.patch(
            f"{base}/{log_id}/person",
            json={"person": national_code},
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["personnel_id"] == pid

    def test_patch_log_person_requires_admin(self, crud):
        base = "/api/v1/logs"
        log_id = self._first_log_id(crud)

        resp = crud.client.patch(
            f"{base}/{log_id}/person",
            json={"person": "0311344119"},
            headers={"Authorization": f"Bearer {crud.viewer_token}"},
        )
        assert resp.status_code == 403

    def test_patch_log_person_not_found(self, crud):
        resp = crud.client.patch(
            "/api/v1/logs/99999999/person",
            json={"person": "0311344119"},
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        assert resp.status_code == 404

    def test_patch_log_attendance_route_removed(self, crud):
        base = "/api/v1/logs"
        log_id = self._first_log_id(crud)

        resp = crud.client.patch(
            f"{base}/{log_id}/attendance",
            json={"counts_for_attendance": False},
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        assert resp.status_code == 404

    def test_patch_log_person_requires_identity(self, crud):
        log_id = self._first_log_id(crud)
        resp = crud.client.patch(
            f"/api/v1/logs/{log_id}/person",
            json={"confidence": 0.8},
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        assert resp.status_code == 422

    def test_patch_log_detection_time_jalali(self, crud):
        base = "/api/v1/logs"
        log_id = self._first_log_id(crud)
        resp = crud.client.patch(
            f"{base}/{log_id}/person",
            json={"detection_time": "1404-05-17 08:30:00"},
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["detection_time"] == "1404-05-17 08:30"
        assert data["id"] == log_id

    def test_patch_log_detection_time_invalid(self, crud):
        log_id = self._first_log_id(crud)
        resp = crud.client.patch(
            f"/api/v1/logs/{log_id}/person",
            json={"detection_time": "not-a-date"},
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        assert resp.status_code == 400

    def test_patch_log_person_and_detection_time_together(self, crud):
        base = "/api/v1/logs"
        log_id = self._first_log_id(crud)
        pid, national_code = self._first_personnel(crud)
        resp = crud.client.patch(
            f"{base}/{log_id}/person",
            json={
                "person": national_code,
                "detection_time": "1404-05-17 14:45:00",
            },
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["personnel_id"] == pid
        assert data["detection_time"] == "1404-05-17 14:45"


class TestPersonnelRequestsApi:
    MODULE = "personnel_requests"

    def test_bulk_create_for_one_personnel(self, crud):
        personnel_response = crud.client.get(
            "/api/v1/personnel/",
            headers={"Authorization": f"Bearer {crud.admin_token}"},
        )
        assert personnel_response.status_code == 200, personnel_response.text
        personnel = personnel_response.json()[0]
        assignment = crud.runtime.shift_store.get_assignment_for_date(
            personnel["id"], parse_jalali_date("1405-06-01")
        )
        assert assignment is not None

        response = crud.client.post(
            "/api/v1/personnel-requests/bulk",
            json={
                "personnel_id": personnel["id"],
                "requests": [
                    {
                        "request_type": "earned_leave",
                        "duration_type": "daily",
                        "start_date": "1405/06/01",
                        "end_date": "1405/06/01",
                    },
                    {
                        "request_type": "mission",
                        "duration_type": "daily",
                        "start_date": "1405/06/02",
                        "end_date": "1405/06/02",
                    },
                ],
            },
            headers={"Authorization": f"Bearer {crud.operator_token}"},
        )

        assert response.status_code == 201, response.text
        payload = response.json()
        assert payload["personnel_id"] == personnel["id"]
        assert payload["count"] == 2
        assert [item["request_type"] for item in payload["requests"]] == [
            "earned_leave",
            "mission",
        ]
        assert all(
            item["personnel_id"] == personnel["id"]
            for item in payload["requests"]
        )

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
        assert forbidden.status_code == 200


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
