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
    from app.api.personnel import router as personnel_router
    from app.api.shifts import router as shifts_router
    from app.api.holidays import router as holidays_router
    from app.api.requests import router as requests_router
    from app.api.personnel_requests import router as personnel_requests_router
    from app.api.personnel_images import router as personnel_images_router
    from app.api.detection_logs import router as detection_logs_router
    from app.api.general_settings import router as general_settings_router
    from app.api.developer import router as developer_router
    from app.api.project_info import router as project_info_router

    app = FastAPI(title="Frontend contract test app")
    app.include_router(auth_router)
    app.include_router(buildings_router)
    app.include_router(sections_router)
    app.include_router(rooms_router)
    app.include_router(personnel_router)
    app.include_router(shifts_router)
    app.include_router(holidays_router)
    app.include_router(requests_router)
    app.include_router(personnel_requests_router)
    app.include_router(personnel_images_router)
    app.include_router(detection_logs_router)
    app.include_router(general_settings_router)
    app.include_router(developer_router)
    app.include_router(project_info_router)
    return app


def test_frontend_expected_backend_routes_are_registered() -> None:
    app = _build_test_app()
    paths = _collect_paths(app.routes)

    expected_frontend_paths = {
        "/api/v1/auth/login",
        "/api/v1/auth/me",
        "/buildings/",
        "/sections/",
        "/rooms/",
        "/api/v1/personnel/",
        "/api/v1/personnel/import-template",
        "/api/v1/personnel-images/personnel/{personnel_id}",
        "/api/v1/personnel-requests/",
        "/api/v1/shifts/",
        "/api/v1/logs/filter",
        "/api/v1/logs/{log_id}",
        "/api/v1/logs/{log_id}/face",
        "/api/v1/logs/{log_id}/body",
        "/api/v1/logs/{log_id}/snapshot",
        "/api/v1/logs/{log_id}/video",
        "/api/v1/logs/{log_id}/face-video",
        "/api/v1/logs/import-excel/template",
        "/api/v1/holidays/",
        "/api/v1/settings/general",
        "/api/v1/developer/apps",
        "/api/v1/developer/apps/{app_key}/urls",
        "/api/v1/developer/urls",
    }

    missing = expected_frontend_paths - paths
    assert missing == set(), f"Missing frontend routes: {missing}"