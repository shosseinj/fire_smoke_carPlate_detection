from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.models import ModelSettingsPatch
from app.core.plate_settings_store import PlateDetectionPolicy
from app.fire_core.policy import FireSmokePolicyConfig
from app.runtime import Runtime


router = APIRouter(prefix="/api/v1/settings", tags=["general-settings"])


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


class PlateDetectionPatch(BaseModel):
    vehicle_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    plate_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    min_vehicle_width_pixels: int | None = Field(default=None, ge=1, le=4096)
    min_vehicle_height_pixels: int | None = Field(default=None, ge=1, le=4096)
    min_vehicle_area_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    vehicle_crop_padding_ratio: float | None = Field(default=None, ge=0.0, le=0.5)


class FireSmokePolicyPatch(BaseModel):
    window_seconds: float | None = Field(default=None, ge=0.25, le=300.0)
    low_count: int | None = Field(default=None, ge=1, le=100000)
    medium_count: int | None = Field(default=None, ge=2, le=100000)
    high_count: int | None = Field(default=None, ge=3, le=100000)

class GeneralSettingsPatch(BaseModel):
    models: ModelSettingsPatch | None = None
    plate_detection: PlateDetectionPatch | None = None
    fire_smoke_detection: FireSmokePolicyPatch | None = None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def _snapshot(runtime: Runtime) -> dict[str, Any]:
    return {
        "models": runtime.models.snapshot(),
        "plate_detection": runtime.plate_settings.general(),
        "fire_smoke_detection": runtime.fire_smoke_logs.settings(),
        "application_config": {
            "source": "environment/startup defaults; use dynamic sections above for online changes",
            "values": _json_safe(asdict(runtime.settings)),
        },
        "camera_processing": {
            "allowed_tasks": ["fire_smoke", "plate_recognition"],
            "modes": {
                "play_only": [],
                "fire_smoke_only": ["fire_smoke"],
                "plate_only": ["plate_recognition"],
                "both": ["fire_smoke", "plate_recognition"],
            },
            "note": "An enabled camera with tasks=[] is decoded and broadcast without AI inference.",
        },
    }


@router.get(
    "/general",
    summary="Read all general configuration in one place",
    description=(
        "Combines model choices, dynamic detection policies, startup configuration, "
        "and the four supported camera processing modes."
    ),
)
def get_general_settings(runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    return _snapshot(runtime)


@router.patch(
    "/general",
    summary="Update online general settings",
    description="Only supplied dynamic sections and fields are changed.",
)
def update_general_settings(
    payload: GeneralSettingsPatch,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    try:
        if payload.models is not None:
            runtime.models.update(
                payload.models.model_dump(exclude_unset=True, exclude_none=True)
            )
        if payload.plate_detection is not None:
            current = runtime.plate_settings.resolve("__general__")
            changes = payload.plate_detection.model_dump(
                exclude_unset=True,
                exclude_none=True,
            )
            runtime.plate_settings.update_general(
                PlateDetectionPolicy(**{**asdict(current), **changes})
            )
        if payload.fire_smoke_detection is not None:
            _, current = runtime.fire_smoke_logs.policy_snapshot()
            changes = payload.fire_smoke_detection.model_dump(
                exclude_unset=True,
                exclude_none=True,
            )
            runtime.fire_smoke_logs.update_policy(
                FireSmokePolicyConfig(**{**asdict(current), **changes})
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _snapshot(runtime)
