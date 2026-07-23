from __future__ import annotations

import pytest


pytestmark = pytest.mark.usefixtures("postgres_database")


def test_legacy_http_routes_are_registered() -> None:
    from app.api.cameras import router as cameras_router
    from app.api.car_plates import router as car_plates_router
    from app.api.fire_logs import router as fire_logs_router
    from app.api.legacy_model_exports import router as model_exports_router
    from app.api.legacy_websocket import router as websocket_router

    routers = (
        cameras_router,
        websocket_router,
        car_plates_router,
        fire_logs_router,
        model_exports_router,
    )
    paths = {getattr(route, "path", "") for item in routers for route in item.routes}
    expected = {
        "/api/v1/cameras/active",
        "/api/v1/cameras/active/effective",
        "/api/v1/cameras/{camera_id}/effective-settings",
        "/api/v1/cameras/batch-active",
        "/api/v1/cameras/health-check",
        "/api/v1/ws/connections/status",
        "/api/v1/ws/video-page",
        "/api/v1/car-plates",
        "/api/v1/fire-logs",
        "/api/v1/model-exports",
    }
    assert expected <= paths


def test_legacy_live_websocket_route_is_registered() -> None:
    from app.api.legacy_websocket import router

    websocket_paths = {
        getattr(route, "path", "")
        for route in router.routes
        if getattr(route, "name", "") == "legacy_live_feed"
    }
    assert "/api/v1/ws/live/{camera_id}" in websocket_paths
