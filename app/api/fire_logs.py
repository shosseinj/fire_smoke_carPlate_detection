from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.auth import get_current_user, require_role
from app.core.auth_store import UserRecord
from app.runtime import Runtime


router = APIRouter(prefix="/api/v1/fire-logs", tags=["fire-logs"])


HazardType = Literal["fire", "smoke", "fire_smoke"]
Severity = Literal["low", "medium", "high"]


class FireLogResponse(BaseModel):
    id: int
    detection_time: datetime
    camera_id: str
    incident_id: str | None = None
    hazard_type: HazardType | None = None
    severity: Severity
    fire_count: int
    smoke_count: int
    fire_confidence: float
    smoke_confidence: float
    window_seconds: float
    snapshot_url: str
    video_url: str
    details: dict[str, Any]


def _require_camera_id(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("camera_id must not be empty")
    return value


def _date_range_values(
    detected_from: datetime | None,
    detected_to: datetime | None,
) -> tuple[str | None, str | None]:
    if detected_from is not None and detected_from.tzinfo is None:
        detected_from = detected_from.replace(tzinfo=timezone.utc)
    if detected_to is not None and detected_to.tzinfo is None:
        detected_to = detected_to.replace(tzinfo=timezone.utc)
    if detected_from is not None and detected_to is not None and detected_from > detected_to:
        raise ValueError("detected_from must not be later than detected_to")
    return (
        detected_from.isoformat() if detected_from is not None else None,
        detected_to.isoformat() if detected_to is not None else None,
    )


def _protected_media_response(value: dict[str, Any]) -> dict[str, Any]:
    response = dict(value)
    log_id = response.get("id")
    if log_id is not None:
        if response.get("snapshot_url"):
            response["snapshot_url"] = f"/api/v1/fire-logs/{log_id}/snapshot"
        if response.get("video_url"):
            response["video_url"] = f"/api/v1/fire-logs/{log_id}/video"
    return response


class FireLogCreate(BaseModel):
    detection_time: datetime
    camera_id: str = Field(min_length=1, description="شناسه دوربین")
    hazard_type: HazardType = Field(description="نوع خطر")
    severity: Severity = Field(description="شدت")
    fire_confidence: float | None = Field(default=None, ge=0, le=1, description="اطمینان حریق")
    smoke_confidence: float | None = Field(default=None, ge=0, le=1, description="اطمینان دود")
    snapshot_url: str | None = Field(default=None, description="آدرس تصویر لحظه‌ای")
    video_url: str | None = Field(default=None, description="آدرس ویدیو")

    _camera_id_not_empty = field_validator("camera_id")(_require_camera_id)

    @model_validator(mode="after")
    def require_confidence(self) -> "FireLogCreate":
        if self.fire_confidence is None and self.smoke_confidence is None:
            raise ValueError("at least one confidence value is required")
        return self


class FireLogUpdate(BaseModel):
    detection_time: datetime | None = Field(default=None, description="زمان تشخیص")
    camera_id: str | None = Field(default=None, description="شناسه دوربین")
    hazard_type: HazardType | None = Field(default=None, description="نوع خطر")
    severity: Severity | None = Field(default=None, description="شدت")
    fire_confidence: float | None = Field(default=None, ge=0, le=1, description="اطمینان حریق")
    smoke_confidence: float | None = Field(default=None, ge=0, le=1, description="اطمینان دود")
    snapshot_url: str | None = Field(default=None, description="آدرس تصویر لحظه‌ای")
    video_url: str | None = Field(default=None, description="آدرس ویدیو")

    _camera_id_not_empty = field_validator("camera_id")(_require_camera_id)


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


@router.get("", response_model=list[FireLogResponse])
@router.get("/", response_model=list[FireLogResponse])
def list_fire_logs(
    camera_id: str | None = None,
    severity: Severity | None = Query(default=None),
    hazard_type: HazardType | None = Query(default=None),
    detected_from: datetime | None = Query(default=None),
    detected_to: datetime | None = Query(default=None),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    _: UserRecord = Depends(get_current_user),
    runtime: Runtime = Depends(get_runtime),
) -> list[dict[str, Any]]:
    try:
        detected_from_value, detected_to_value = _date_range_values(detected_from, detected_to)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    rows = runtime.fire_smoke_logs.list(
        camera=camera_id,
        severity=severity,
        hazard_type=hazard_type,
        detected_from=detected_from_value,
        detected_to=detected_to_value,
        limit=skip + limit,
    )[skip:]
    return [_protected_media_response(row) for row in rows]


@router.get("/{log_id}", response_model=FireLogResponse)
def get_fire_log(log_id: int, _: UserRecord = Depends(get_current_user), runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    value = runtime.fire_smoke_logs.get(log_id)
    if value is None:
        raise HTTPException(status_code=404, detail="لاگ حریق یا دود یافت نشد")
    return _protected_media_response(value)


def _fire_log_media_path(log_id: int, media_kind: str, runtime: Runtime) -> Path:
    value = runtime.fire_smoke_logs.get(log_id)
    if value is None:
        raise HTTPException(status_code=404, detail="لاگ حریق یا دود یافت نشد")
    media_url = value.get(f"{media_kind}_url")
    if not media_url:
        raise HTTPException(status_code=404, detail="فایل رسانه‌ای یافت نشد")
    media_root = runtime.settings.saved_media_path.resolve()
    relative_path = str(media_url).removeprefix("/media/")
    candidate = (media_root / relative_path).resolve()
    if media_root not in candidate.parents or not candidate.is_file():
        raise HTTPException(status_code=404, detail="فایل رسانه‌ای یافت نشد")
    return candidate


@router.get("/{log_id}/snapshot", response_class=FileResponse)
def get_fire_log_snapshot(
    log_id: int,
    _: UserRecord = Depends(get_current_user),
    runtime: Runtime = Depends(get_runtime),
) -> FileResponse:
    return FileResponse(_fire_log_media_path(log_id, "snapshot", runtime))


@router.get("/{log_id}/video", response_class=FileResponse)
def get_fire_log_video(
    log_id: int,
    _: UserRecord = Depends(get_current_user),
    runtime: Runtime = Depends(get_runtime),
) -> FileResponse:
    return FileResponse(_fire_log_media_path(log_id, "video", runtime))


@router.post("", status_code=status.HTTP_201_CREATED, response_model=FireLogResponse)
@router.post("/", status_code=status.HTTP_201_CREATED, response_model=FireLogResponse)
def create_fire_log(payload: FireLogCreate, _: UserRecord = Depends(require_role("admin")), runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    return _protected_media_response(runtime.fire_smoke_logs.create_manual(payload.model_dump()))


@router.patch("/{log_id}", response_model=FireLogResponse)
def update_fire_log(log_id: int, payload: FireLogUpdate, _: UserRecord = Depends(require_role("superadmin")), runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    value = runtime.fire_smoke_logs.update_manual(log_id, payload.model_dump(exclude_unset=True))
    if value is None:
        raise HTTPException(status_code=404, detail="لاگ حریق یا دود یافت نشد")
    return _protected_media_response(value)


@router.delete("/{log_id}")
def delete_fire_log(log_id: int, _: UserRecord = Depends(require_role("superadmin")), runtime: Runtime = Depends(get_runtime)) -> dict[str, str]:
    if not runtime.fire_smoke_logs.delete(log_id):
        raise HTTPException(status_code=404, detail="لاگ حریق یا دود یافت نشد")
    return {"message": "لاگ حریق یا دود با موفقیت حذف شد"}
