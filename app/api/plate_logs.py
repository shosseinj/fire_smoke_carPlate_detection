from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator

from app.core.auth import get_current_user, require_role
from app.core.auth_store import UserRecord
from app.core.plate_constants import (
    PlateLogDirection,
    PlateLogSourceType,
    normalize_persian_text,
    normalize_plate_full_number,
)
from app.runtime import Runtime
from app.time_utils import utc_now


router = APIRouter(prefix="/api/v1/plate-logs", tags=["plate-logs"])


class PlateLogCreate(BaseModel):
    plate_id: Optional[int] = Field(default=None, ge=1)
    plate_full_number: str = Field(..., max_length=32)
    raw_plate_text: Optional[str] = Field(default=None, max_length=64)
    detection_time: datetime = Field()
    camera_id: str = Field(..., min_length=1, max_length=200)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    direction: PlateLogDirection = Field(default=PlateLogDirection.UNKNOWN)
    source_type: PlateLogSourceType = Field(default=PlateLogSourceType.CAMERA)
    snapshot_path: Optional[str] = Field(default=None, max_length=512)
    plate_crop_path: Optional[str] = Field(default=None, max_length=512)
    notes: Optional[str] = None

    @field_validator("plate_full_number", mode="before")
    @classmethod
    def normalize_plate_number(cls, value: object) -> object:
        return normalize_plate_full_number(value)

    @field_validator("raw_plate_text", "notes", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        return normalize_persian_text(value)

    @field_validator("detection_time")
    @classmethod
    def validate_detection_time_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("زمان تشخیص باید همراه با منطقه زمانی ارسال شود")
        return value


class PlateLogUpdate(BaseModel):
    plate_id: Optional[int] = Field(default=None, ge=1)
    plate_full_number: Optional[str] = Field(default=None, max_length=32)
    raw_plate_text: Optional[str] = Field(default=None, max_length=64)
    detection_time: Optional[datetime] = None
    camera_id: Optional[str] = Field(default=None, min_length=1, max_length=200)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    direction: Optional[PlateLogDirection] = None
    snapshot_path: Optional[str] = Field(default=None, max_length=512)
    plate_crop_path: Optional[str] = Field(default=None, max_length=512)
    is_verified: Optional[bool] = None
    notes: Optional[str] = None

    @field_validator("plate_full_number", mode="before")
    @classmethod
    def normalize_plate_number(cls, value: object) -> object:
        if value is None:
            return None
        return normalize_plate_full_number(value)

    @field_validator("raw_plate_text", "notes", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        return normalize_persian_text(value)

    @field_validator("detection_time")
    @classmethod
    def validate_detection_time_timezone(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("زمان تشخیص باید همراه با منطقه زمانی ارسال شود")
        return value


class PlateLogResponse(BaseModel):
    id: int
    plate_id: Optional[int] = None
    plate_full_number: str
    raw_plate_text: Optional[str] = None
    detection_time: Optional[datetime] = None
    camera_id: Optional[str] = None
    confidence: Optional[float] = None
    direction: Optional[str] = None
    source_type: Optional[str] = None
    snapshot_path: Optional[str] = None
    plate_crop_path: Optional[str] = None
    is_verified: bool = False
    created_by_user_id: Optional[int] = None
    verified_by_user_id: Optional[int] = None
    verified_at: Optional[datetime] = None
    notes: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


def _get_log_or_404(log_id: int, runtime: Runtime) -> dict[str, Any]:
    value = runtime.plate_logs.get_log(log_id)
    if value is None:
        raise HTTPException(status_code=404, detail="لاگ تشخیص پلاک یافت نشد")
    return value


def _validate_references(runtime: Runtime, plate_id: int | None, camera_id: str) -> None:
    if plate_id is not None:
        plate = runtime.car_plates.get(plate_id)
        if plate is None:
            raise HTTPException(status_code=404, detail="پلاک ثبت‌شده انتخاب‌شده یافت نشد")


@router.get("", response_model=list[PlateLogResponse])
def list_plate_logs(
    plate_id: Optional[int] = Query(default=None, ge=1),
    plate_full_number: Optional[str] = Query(default=None, max_length=32),
    camera_id: Optional[str] = Query(default=None, max_length=200),
    direction: Optional[PlateLogDirection] = None,
    source_type: Optional[PlateLogSourceType] = None,
    is_verified: Optional[bool] = None,
    detected_from: Optional[str] = Query(default=None),
    detected_to: Optional[str] = Query(default=None),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    _: UserRecord = Depends(get_current_user),
    runtime: Runtime = Depends(get_runtime),
) -> list[dict[str, Any]]:
    try:
        if detected_from:
            datetime.fromisoformat(detected_from.replace("Z", "+00:00"))
        if detected_to:
            datetime.fromisoformat(detected_to.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="بازه زمانی باید با قالب معتبر ISO 8601 و همراه با منطقه زمانی ارسال شود",
        )

    return runtime.plate_logs.list_logs(
        plate_id=plate_id,
        plate_full_number=plate_full_number,
        camera_id=camera_id,
        direction=direction.value if direction else None,
        source_type=source_type.value if source_type else None,
        is_verified=is_verified,
        detected_from=detected_from,
        detected_to=detected_to,
        skip=skip,
        limit=limit,
    )


@router.get("/{log_id}", response_model=PlateLogResponse)
def get_plate_log(
    log_id: int,
    _: UserRecord = Depends(get_current_user),
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    return _get_log_or_404(log_id, runtime)


@router.post("", response_model=PlateLogResponse, status_code=status.HTTP_201_CREATED)
@router.post("/", response_model=PlateLogResponse, status_code=status.HTTP_201_CREATED)
def create_plate_log(
    log_data: PlateLogCreate,
    admin_user: UserRecord = Depends(require_role("admin")),
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    payload = log_data.model_dump(mode="python")
    source_type = payload["source_type"]
    created_by_user_id: int | None = admin_user.id if source_type == PlateLogSourceType.MANUAL.value else None
    _validate_references(runtime, payload.get("plate_id"), payload["camera_id"])

    data = dict(payload)
    data["created_by_user_id"] = created_by_user_id
    data["plate_full_number"] = normalize_plate_full_number(payload["plate_full_number"])
    try:
        return runtime.plate_logs.create_log(data)
    except Exception as exc:
        if "UNIQUE" in str(exc).upper():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="تشخیص مشابه این پلاک برای همین دوربین و جهت، در بازه یک دقیقه‌ای قبلاً ثبت شده است",
            ) from exc
        raise


@router.patch("/{log_id}", response_model=PlateLogResponse)
def update_plate_log(
    log_id: int,
    log_data: PlateLogUpdate,
    superuser: UserRecord = Depends(require_role("superuser")),
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    plate_log = _get_log_or_404(log_id, runtime)
    updates = log_data.model_dump(exclude_unset=True, mode="python")

    final_plate_id = updates.get("plate_id", plate_log.get("plate_id"))
    final_camera_id = updates.get("camera_id", plate_log.get("camera_id", ""))
    _validate_references(runtime, final_plate_id, final_camera_id)

    requested_verification = updates.pop("is_verified", None)
    if requested_verification is True:
        updates["is_verified"] = True
        updates["verified_by_user_id"] = superuser.id
        updates["verified_at"] = utc_now().isoformat()
    elif requested_verification is False:
        updates["is_verified"] = False
        updates["verified_by_user_id"] = None
        updates["verified_at"] = None

    result = runtime.plate_logs.update_log(log_id, updates)
    if result is None:
        raise HTTPException(status_code=404, detail="لاگ تشخیص پلاک یافت نشد")
    return result


@router.delete("/{log_id}")
def delete_plate_log(
    log_id: int,
    superuser: UserRecord = Depends(require_role("superuser")),
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, str]:
    _get_log_or_404(log_id, runtime)
    if not runtime.plate_logs.delete_log(log_id):
        raise HTTPException(status_code=404, detail="لاگ تشخیص پلاک یافت نشد")
    return {"message": "لاگ تشخیص پلاک با موفقیت حذف شد"}
