from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from fastapi import FastAPI

import app.api.general_settings as general_settings_module
from app.api.general_settings import GeneralSettingsPatch


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


class _FakeGeneralSettingsStore:
    def __init__(self, force: bool = False) -> None:
        self.force = force
        self.update_calls = []

    def get(self):
        return SimpleNamespace(
            force=self.force,
            operational=SimpleNamespace(to_dict=lambda: {}),
            draw_box=True,
            draw_face=True,
            draw_skeleton=False,
            draw_zones=True,
            confirmation_threshold=0.6,
        )

    def update(self, changes, updated_by=None):
        self.update_calls.append((changes, updated_by))
        if "force" in changes:
            self.force = changes["force"]
        return self.get()


def test_general_settings_patch_persists_force(monkeypatch) -> None:
    store = _FakeGeneralSettingsStore()
    runtime = SimpleNamespace(general_settings=store)
    monkeypatch.setattr(
        general_settings_module,
        "_snapshot",
        lambda current_runtime: {"force": current_runtime.general_settings.get().force},
    )

    response = general_settings_module.update_general_settings(
        GeneralSettingsPatch(force=True),
        current_user=SimpleNamespace(id=7),
        runtime=runtime,
    )

    assert store.update_calls == [({"force": True}, 7)]
    assert response["force"] is True


@dataclass
class _FakeApplicationSettings:
    database_url: str | None = None
    face_qdrant_api_key: str | None = None
    jwt_secret_key: str | None = None
    auth_default_admin_password: str | None = None


def test_general_settings_snapshot_exposes_force() -> None:
    runtime = SimpleNamespace(
        settings=_FakeApplicationSettings(),
        general_settings=_FakeGeneralSettingsStore(force=True),
        models=SimpleNamespace(snapshot=lambda: {}),
        plate_settings=SimpleNamespace(general=lambda: {}),
        fire_smoke_logs=SimpleNamespace(settings=lambda: {}),
        face_quality_settings=SimpleNamespace(as_dict=lambda: {}),
    )

    assert general_settings_module._snapshot(runtime)["force"] is True

import pytest

pytestmark = pytest.mark.unit
