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
    from app.api.project_info import router as project_info_router

    app = FastAPI(title="Project info test app")
    app.include_router(project_info_router, prefix="/api/v1")
    return app


def test_project_info_endpoint_registered() -> None:
    app = _build_test_app()
    paths = _collect_paths(app.routes)
    assert "/api/v1/project-info" in paths


def test_project_info_current_endpoint_registered() -> None:
    app = _build_test_app()
    paths = _collect_paths(app.routes)
    assert "/api/v1/project-info/current" in paths


def test_project_info_releases_endpoint_registered() -> None:
    app = _build_test_app()
    paths = _collect_paths(app.routes)
    assert "/api/v1/project-info/releases" in paths
import pytest

pytestmark = pytest.mark.unit
