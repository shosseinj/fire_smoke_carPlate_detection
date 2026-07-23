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
    from app.api.general_settings import router as general_settings_router
    from app.api.auth import router as auth_router

    app = FastAPI(title="Settings test app")
    app.include_router(auth_router)
    app.include_router(general_settings_router)
    return app


def test_general_settings_defaults_endpoint_registered() -> None:
    app = _build_test_app()
    paths = _collect_paths(app.routes)
    assert "/api/v1/settings/general" in paths