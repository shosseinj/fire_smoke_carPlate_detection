from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.engine import make_url

from app.api.models import ModelSettingsPatch
from app.core.auth import require_role
from app.core.auth_store import UserRecord
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

class OperationalSettingsPatch(BaseModel):
    video_ingest_fps: float | None = Field(default=None, gt=0, le=240)
    video_preview_fps: float | None = Field(default=None, gt=0, le=240)
    video_loop: bool | None = None
    rtsp_transport: str | None = None
    rtsp_open_timeout_ms: int | None = Field(default=None, gt=0)
    rtsp_read_timeout_ms: int | None = Field(default=None, gt=0)
    rtsp_reconnect_seconds: float | None = Field(default=None, ge=0.5)
    deepstream_rtsp_latency_ms: int | None = Field(default=None, gt=0)
    deepstream_rtsp_stall_timeout_seconds: int | None = Field(default=None, gt=0)
    broadcast_enabled: bool | None = None
    broadcast_jpeg_quality: int | None = Field(default=None, ge=1, le=100)
    broadcast_wall_jpeg_quality: int | None = Field(default=None, ge=1, le=100)
    broadcast_wall_max_width: int | None = Field(default=None, gt=0, le=4096)
    broadcast_wall_max_height: int | None = Field(default=None, gt=0, le=4096)
    fire_confidence: float | None = Field(default=None, ge=0, le=1)
    smoke_confidence: float | None = Field(default=None, ge=0, le=1)
    plate_confidence: float | None = Field(default=None, ge=0, le=1)
    plate_iou: float | None = Field(default=None, ge=0, le=1)
    vehicle_confidence: float | None = Field(default=None, ge=0, le=1)
    vehicle_iou: float | None = Field(default=None, ge=0, le=1)
    face_human_confidence: float | None = Field(default=None, ge=0, le=1)
    face_detection_confidence: float | None = Field(default=None, ge=0, le=1)
    face_recognition_threshold: float | None = Field(default=None, ge=0, le=1)


class DisplaySettingsPatch(BaseModel):
    draw_box: bool | None = None
    draw_face: bool | None = None
    draw_skeleton: bool | None = None
    draw_zones: bool | None = None

class GeneralSettingsPatch(BaseModel):
    models: ModelSettingsPatch | None = None
    plate_detection: PlateDetectionPatch | None = None
    fire_smoke_detection: FireSmokePolicyPatch | None = None
    operational: OperationalSettingsPatch | None = None
    display: DisplaySettingsPatch | None = None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def _snapshot(runtime: Runtime) -> dict[str, Any]:
    application_values = asdict(runtime.settings)
    if application_values.get("database_url"):
        application_values["database_url"] = make_url(
            str(application_values["database_url"])
        ).render_as_string(hide_password=True)
    for secret_name in (
        "face_qdrant_api_key",
        "jwt_secret_key",
        "auth_default_admin_password",
    ):
        if application_values.get(secret_name):
            application_values[secret_name] = "***"
    gs = runtime.general_settings.get()
    return {
        "operational": gs.operational.to_dict(),
        "models": runtime.models.snapshot(),
        "plate_detection": runtime.plate_settings.general(),
        "fire_smoke_detection": runtime.fire_smoke_logs.settings(),
        "face_recognition_quality": runtime.face_quality_settings.as_dict(),
        "application_config": {
            "source": "environment/startup defaults; use dynamic sections above for online changes",
            "values": _json_safe(application_values),
        },
        "camera_processing": {
            "allowed_tasks": ["fire_smoke", "plate_recognition", "face_recognition"],
            "modes": {
                "play_only": [],
                "fire_smoke_only": ["fire_smoke"],
                "plate_only": ["plate_recognition"],
                "face_recognition_only": ["face_recognition"],
                "both": ["fire_smoke", "plate_recognition"],
                "all": ["fire_smoke", "plate_recognition", "face_recognition"],
            },
            "note": "An enabled camera with tasks=[] is decoded and broadcast without AI inference.",
        },
        "display": {
            "draw_box": gs.draw_box,
            "draw_face": gs.draw_face,
            "draw_skeleton": gs.draw_skeleton,
            "draw_zones": gs.draw_zones,
        },
    }


@router.get(
    "/general",
    summary="Read all general configuration in one place",
    description=(
        "Combines model choices, dynamic detection policies, startup configuration, "
        "and the supported camera processing modes."
    ),
)
def get_general_settings(runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    return _snapshot(runtime)


@router.patch(
    "/general",
    summary="بروزرسانی تنظیمات عمومی آنلاین",
    description="فقط بخش‌ها و فیلدهای ارسال‌شده تغییر می‌کنند.",
)
def update_general_settings(
    payload: GeneralSettingsPatch,
    current_user: UserRecord = Depends(require_role("admin")),
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
        if payload.display is not None:
            display_changes = payload.display.model_dump(exclude_unset=True, exclude_none=True)
            if display_changes:
                runtime.general_settings.update(display_changes, updated_by=current_user.id)
                if "draw_zones" in display_changes:
                    runtime.broadcast.set_draw_zones(display_changes["draw_zones"])
        if payload.operational is not None:
            changes = payload.operational.model_dump(exclude_unset=True)
            if changes:
                runtime.general_settings.update(changes, updated_by=current_user.id)
                runtime.apply_operational_settings()
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


@router.post(
    "/general/reset",
    summary="بازنشانی تنظیمات عمومی به مقادیر پیش‌فرض",
)
def reset_general_settings(
    current_user: UserRecord = Depends(require_role("admin")),
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    runtime.general_settings.reset(updated_by=current_user.id)
    return _snapshot(runtime)
