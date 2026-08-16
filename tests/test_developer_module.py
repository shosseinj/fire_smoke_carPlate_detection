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
    from app.api.developer import router as developer_router
    from app.api.auth import router as auth_router

    app = FastAPI(title="Developer test app")
    app.include_router(auth_router)
    app.include_router(developer_router)
    return app


def test_developer_apps_endpoint_registered() -> None:
    app = _build_test_app()
    paths = _collect_paths(app.routes)
    assert "/api/v1/developer/apps" in paths


def test_developer_app_urls_endpoint_registered() -> None:
    app = _build_test_app()
    paths = _collect_paths(app.routes)
    assert "/api/v1/developer/apps/{app_key}/urls" in paths


def test_developer_urls_endpoint_registered() -> None:
    app = _build_test_app()
    paths = _collect_paths(app.routes)
    assert "/api/v1/developer/urls" in paths


def test_developer_test_request_endpoint_registered() -> None:
    app = _build_test_app()
    paths = _collect_paths(app.routes)
    assert "/api/v1/developer/test-request" in paths
import pytest

pytestmark = pytest.mark.unit
