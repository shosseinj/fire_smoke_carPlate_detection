from __future__ import annotations

import pytest


pytestmark = pytest.mark.usefixtures("postgres_database")


def test_legacy_http_routes_are_registered() -> None:
    import app.main as main_module

    paths = {getattr(route, "path", "") for route in main_module.app.routes}
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
    import app.main as main_module

    websocket_paths = {
        getattr(route, "path", "")
        for route in main_module.app.routes
        if getattr(route, "name", "") == "legacy_live_feed"
    }
    assert "/api/v1/ws/live/{camera_id}" in websocket_paths
