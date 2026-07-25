from __future__ import annotations

import pytest


pytestmark = pytest.mark.usefixtures("postgres_database")


def test_legacy_http_routes_are_registered() -> None:
    from app.api.cameras import router as cameras_router
    from app.api.car_plates import router as car_plates_router
    from app.api.fire_logs import router as fire_logs_router

    routers = (
        cameras_router,
        car_plates_router,
        fire_logs_router,
    )
    paths = {getattr(route, "path", "") for item in routers for route in item.routes}
    expected = {
        "/api/v1/cameras/{camera_id}/effective-settings",
        "/api/v1/cameras/health-check",
        "/api/v1/car-plates",
        "/api/v1/fire-logs",
    }
    assert expected <= paths