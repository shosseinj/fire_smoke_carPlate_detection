from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.plate_settings_store import PlateDetectionPolicy
from app.runtime import Runtime

router = APIRouter(prefix="/api/v1/plate-settings", tags=["plate-settings"])


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


class GeneralPlateSettingsUpdate(BaseModel):
    vehicle_confidence: float = Field(ge=0.0, le=1.0)
    plate_confidence: float = Field(ge=0.0, le=1.0)
    ocr_confidence: float = Field(ge=0.0, le=1.0)
    min_vehicle_width_pixels: int = Field(ge=1, le=4096)
    min_vehicle_height_pixels: int = Field(ge=1, le=4096)
    min_vehicle_area_ratio: float = Field(ge=0.0, le=1.0)
    vehicle_crop_padding_ratio: float = Field(ge=0.0, le=0.5)


class CameraPlateSettingsPatch(BaseModel):
    vehicle_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    plate_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    min_vehicle_width_pixels: int | None = Field(default=None, ge=1, le=4096)
    min_vehicle_height_pixels: int | None = Field(default=None, ge=1, le=4096)
    min_vehicle_area_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    vehicle_crop_padding_ratio: float | None = Field(default=None, ge=0.0, le=0.5)


@router.get(
    "/general",
    summary="Read general plate-recognition settings",
    description="These values are inherited by every camera without an override.",
)
def get_general_plate_settings(
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    return runtime.plate_settings.general()


@router.put(
    "/general",
    summary="Replace general plate-recognition settings",
    description="Changes take effect online without restarting model workers.",
)
def update_general_plate_settings(
    payload: GeneralPlateSettingsUpdate,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    try:
        return runtime.plate_settings.update_general(
            PlateDetectionPolicy(**payload.model_dump())
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _require_camera(runtime: Runtime, camera_id: str) -> None:
    if runtime.registry.get(camera_id) is None:
        raise HTTPException(status_code=404, detail="Camera not found")


@router.get(
    "/cameras/{camera_id}",
    summary="Read effective settings for one camera",
    description="Returns explicit overrides, inherited fields, and final effective values.",
)
def get_camera_plate_settings(
    camera_id: str,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    _require_camera(runtime, camera_id)
    return runtime.plate_settings.camera(camera_id)


@router.patch(
    "/cameras/{camera_id}",
    summary="Set or clear per-camera overrides",
    description=(
        "Only submitted fields are changed. Send null for a field to remove its "
        "override and inherit the current general value."
    ),
)
def update_camera_plate_settings(
    camera_id: str,
    payload: CameraPlateSettingsPatch,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    _require_camera(runtime, camera_id)
    try:
        return runtime.plate_settings.update_camera(
            camera_id,
            payload.model_dump(exclude_unset=True),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete(
    "/cameras/{camera_id}",
    summary="Reset all camera overrides",
)
def reset_camera_plate_settings(
    camera_id: str,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    _require_camera(runtime, camera_id)
    removed = runtime.plate_settings.delete_camera(camera_id)
    return {"removed": removed, **runtime.plate_settings.camera(camera_id)}
