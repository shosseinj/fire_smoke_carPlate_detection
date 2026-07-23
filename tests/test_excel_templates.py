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
    from app.api.personnel import router as personnel_router
    from app.api.detection_logs import router as detection_logs_router
    from app.api.auth import router as auth_router

    app = FastAPI(title="Excel templates test app")
    app.include_router(auth_router)
    app.include_router(personnel_router)
    app.include_router(detection_logs_router)
    return app


def test_personnel_import_template_endpoint_registered() -> None:
    app = _build_test_app()
    paths = _collect_paths(app.routes)
    assert "/api/v1/personnel/import-template" in paths


def test_logs_import_template_endpoint_registered() -> None:
    app = _build_test_app()
    paths = _collect_paths(app.routes)
    assert "/api/v1/logs/import-excel/template" in paths