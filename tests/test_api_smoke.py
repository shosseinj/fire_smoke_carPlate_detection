from __future__ import annotations

import os

import pytest
from fastapi import FastAPI


def _collect_paths(routes):
    paths = set()
    for route in routes:
        if hasattr(route, "path"):
            paths.add(route.path)
        if hasattr(route, "original_router"):
            paths.update(_collect_paths(route.original_router.routes))
    return paths


def _test_routers():
    from app.api.auth import router as auth_router
    from app.api.locations import buildings_router, sections_router, rooms_router
    from app.api.personnel import router as personnel_router
    from app.api.shifts import router as shifts_router
    from app.api.holidays import router as holidays_router
    from app.api.personnel_requests import router as personnel_requests_router
    from app.api.detection_logs import router as detection_logs_router
    from app.api.general_settings import router as general_settings_router
    return [
        ("auth", auth_router),
        ("buildings", buildings_router),
        ("sections", sections_router),
        ("rooms", rooms_router),
        ("personnel", personnel_router),
        ("shifts", shifts_router),
        ("holidays", holidays_router),
        ("personnel_requests", personnel_requests_router),
        ("detection_logs", detection_logs_router),
        ("general_settings", general_settings_router),
    ]


def _build_test_app() -> FastAPI:
    app = FastAPI(title="API smoke test app")
    for _name, router in _test_routers():
        app.include_router(router)
    return app


@pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL must point to a running PostgreSQL")
def test_protected_list_endpoints_reject_unauthenticated_requests() -> None:
    from fastapi.testclient import TestClient

    app = _build_test_app()
    client = TestClient(app)

    protected_paths = [
        "/api/v1/personnel/",
        "/api/v1/rooms/",
        "/api/v1/buildings/",
        "/api/v1/sections/",
        "/api/v1/shifts/",
        "/api/v1/personnel-requests/",
        "/api/v1/holidays/",
        "/api/v1/general-settings/",
        "/api/v1/logs/filter",
    ]

    for path in protected_paths:
        response = client.get(path)
        assert response.status_code in {401, 403}, f"{path} returned {response.status_code}"


def test_health_endpoint_not_present_in_api_router() -> None:
    app = _build_test_app()
    paths = _collect_paths(app.routes)
    assert "/health" not in paths