from __future__ import annotations

from fastapi import FastAPI


def _collect_paths(routes):
    paths = set()
    for route in routes:
        if hasattr(route, "path"):
            paths.add(route.path)
        if hasattr(route, "original_router"):
            paths.update(_collect_paths(route.original_router.routes))
    return paths


def _build_test_app() -> FastAPI:
    from app.api.auth import router as auth_router
    from app.api.locations import buildings_router, sections_router, rooms_router
    from app.api.diagnostics import router as diagnostics_router
    from app.api.processor_tests import router as processor_tests_router
    from app.api.general_settings import router as general_settings_router
    from app.api.models import router as models_router
    from app.api.sources import router as sources_router
    from app.api.frames import router as frames_router
    from app.api.results import router as results_router
    from app.api.broadcast import router as broadcast_router
    from app.api.plate_logs import router as plate_logs_router
    from app.api.car_plates import router as car_plates_router
    from app.api.fire_smoke_logs import router as fire_smoke_logs_router
    from app.api.fire_logs import router as fire_logs_router
    from app.api.faces import router as faces_router
    from app.api.humans import router as humans_router
    from app.api.personnel import router as personnel_router
    from app.api.shifts import router as shifts_router
    from app.api.holidays import router as holidays_router
    from app.api.requests import router as requests_router
    from app.api.personnel_requests import router as personnel_requests_router
    from app.api.personnel_images import router as personnel_images_router
    from app.api.detection_logs import router as detection_logs_router
    from app.api.plate_settings import router as plate_settings_router
    from app.api.developer import router as developer_router
    from app.api.project_info import router as project_info_router

    app = FastAPI(title="Route registry test app")
    app.include_router(auth_router)
    app.include_router(buildings_router)
    app.include_router(sections_router)
    app.include_router(rooms_router)
    app.include_router(diagnostics_router)
    app.include_router(processor_tests_router)
    app.include_router(general_settings_router)
    app.include_router(models_router)
    app.include_router(sources_router)
    app.include_router(frames_router)
    app.include_router(results_router)
    app.include_router(broadcast_router)
    app.include_router(plate_logs_router)
    app.include_router(car_plates_router)
    app.include_router(fire_smoke_logs_router)
    app.include_router(fire_logs_router)
    app.include_router(faces_router)
    app.include_router(humans_router)
    app.include_router(personnel_router)
    app.include_router(shifts_router)
    app.include_router(holidays_router)
    app.include_router(requests_router)
    app.include_router(personnel_requests_router)
    app.include_router(personnel_images_router)
    app.include_router(detection_logs_router)
    app.include_router(plate_settings_router)
    app.include_router(developer_router)
    app.include_router(project_info_router)
    return app


def test_core_backend_routes_are_registered() -> None:
    app = _build_test_app()
    paths = _collect_paths(app.routes)

    expected_paths = {
        "/api/v1/auth/token",
        "/api/v1/auth/login",
        "/api/v1/personnel/search/{national_code}",
        "/api/v1/personnel/",
        "/api/v1/personnel-images/personnel/{personnel_id}",
        "/api/v1/logs/filter",
        "/api/v1/logs/{log_id}",
        "/api/v1/sources",
        "/api/v1/sources/{id:path}",
        "/api/v1/sources/{id:path}/enable",
        "/api/v1/sources/{id:path}/disable",
        "/api/v1/sources/bulk/task-assignment",
        "/rooms/",
        "/buildings/",
        "/sections/",
        "/api/v1/shifts/",
        "/api/v1/personnel-requests/",
        "/api/v1/holidays/",
        "/api/v1/settings/general",
        "/api/v1/developer/apps",
        "/api/v1/developer/apps/{app_key}/urls",
        "/api/v1/developer/urls",
        "/api/v1/project-info",
    }

    missing = expected_paths - paths
    assert missing == set(), f"Missing routes: {missing}"
    assert "/api/v1/cameras" not in paths

import pytest

pytestmark = pytest.mark.unit
