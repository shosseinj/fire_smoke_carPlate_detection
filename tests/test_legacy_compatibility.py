from __future__ import annotations

import pytest


pytestmark = pytest.mark.usefixtures("postgres_database")


def test_legacy_http_routes_are_registered() -> None:
    from app.api.sources import router as sources_router
    from app.api.car_plates import router as car_plates_router
    from app.api.fire_logs import router as fire_logs_router

    routers = (
        sources_router,
        car_plates_router,
        fire_logs_router,
    )
    paths = {getattr(route, "path", "") for item in routers for route in item.routes}
    expected = {
        "/api/v1/sources",
        "/api/v1/sources/{id:path}",
        "/api/v1/sources/{id:path}/enable",
        "/api/v1/sources/{id:path}/disable",
        "/api/v1/sources/bulk/task-assignment",
        "/api/v1/car-plates",
        "/api/v1/fire-logs",
    }
    assert expected <= paths
    assert not any(path.startswith("/api/v1/cameras") for path in paths)
