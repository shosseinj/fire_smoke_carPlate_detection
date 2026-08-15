from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.auth import get_current_user, require_permission
from app.core.auth_store import UserRecord
from app.core.jalali_utils import utc_iso_to_jalali_datetime
from app.core.detection_media import DetectionMediaStorage, InvalidMediaKey
from app.core.plate_constants import (
    PlateLogSourceType,
    normalize_persian_text,
    normalize_plate_full_number,
)
from app.runtime import Runtime


router = APIRouter(prefix="/api/v1/plate-logs", tags=["plate-logs"])


class PlateLogCreate(BaseModel):
    plate_id: int | None = Field(default=None, ge=1)
    plate_number: str = Field(max_length=32)
    raw_plate_text: str | None = Field(default=None, max_length=64)
    confidence: float | None = Field(default=None, ge=0, le=1)
    detection_time: datetime
    source_type: PlateLogSourceType = PlateLogSourceType.MANUAL
    snapshot_key: str | None = Field(default=None, max_length=512)
    video_key: str | None = Field(default=None, max_length=512)
    notes: str | None = None

    @field_validator("plate_number", mode="before")
    @classmethod
    def normalize_plate_number(cls, value: object) -> object:
        return normalize_plate_full_number(value)

    @field_validator("raw_plate_text", "notes", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        return normalize_persian_text(value)

    @field_validator("detection_time")
    @classmethod
    def validate_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("زمان تشخیص باید همراه با منطقه زمانی ارسال شود")
        return value

    @model_validator(mode="after")
    def manual_only(self) -> "PlateLogCreate":
        if self.source_type != PlateLogSourceType.MANUAL:
            raise ValueError("ثبت API فقط برای لاگ دستی مجاز است")
        return self


class PlateLogUpdate(BaseModel):
    plate_id: int | None = Field(default=None, ge=1)
    plate_number: str | None = Field(default=None, max_length=32)
    raw_plate_text: str | None = Field(default=None, max_length=64)
    confidence: float | None = Field(default=None, ge=0, le=1)
    detection_time: datetime | None = None
    snapshot_key: str | None = Field(default=None, max_length=512)
    video_key: str | None = Field(default=None, max_length=512)
    notes: str | None = None

    @field_validator("plate_number", mode="before")
    @classmethod
    def normalize_plate_number(cls, value: object) -> object:
        return None if value is None else normalize_plate_full_number(value)

    @field_validator("raw_plate_text", "notes", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        return normalize_persian_text(value)

    @field_validator("detection_time")
    @classmethod
    def validate_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("زمان تشخیص باید همراه با منطقه زمانی ارسال شود")
        return value


class PlateLogResponse(BaseModel):
    id: int
    source_type: PlateLogSourceType
    source_uri: str | None = None
    static_video_id: int | None = None
    created_by_user_id: int | None = None
    updated_by_user_id: int | None = None
    plate_id: int | None = None
    is_registered: bool
    plate_number: str | None = None
    raw_plate_text: str | None = None
    confidence: float | None = None
    detection_time: datetime
    detection_time_local: str
    detection_time_jalali: str
    snapshot_thumbnail: str | None = None
    snap_shot_url: str | None = None
    video_url: str | None = None
    notes: str | None = None
    created_at: datetime
    updated_at: datetime


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


def _time_text(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _media_storage(runtime: Runtime) -> DetectionMediaStorage:
    return DetectionMediaStorage(runtime.plate_logs.media_root)


def _response(
    value: dict[str, Any],
    media: DetectionMediaStorage,
    *,
    check_media: bool = True,
) -> dict[str, Any]:
    result = dict(value)
    detected = result["detection_time"]
    if not isinstance(detected, datetime):
        detected = datetime.fromisoformat(str(detected).replace("Z", "+00:00"))
    result["detection_time_local"] = detected.astimezone(
        ZoneInfo("Asia/Tehran")
    ).isoformat()
    result["detection_time_jalali"] = (
        utc_iso_to_jalali_datetime(_time_text(detected)) or ""
    )
    snapshot_key = result.pop("snapshot_key", None)
    video_key = result.pop("video_key", None)
    result.pop("snapshot_url", None)
    snapshot_ready = bool(snapshot_key) and (
        not check_media or media.exists(snapshot_key)
    )
    video_ready = bool(video_key) and (
        not check_media or media.exists(video_key)
    )
    result["snapshot_thumbnail"] = (
        media.thumbnail_data_uri(None, snapshot_key) if snapshot_key else None
    )
    result["snap_shot_url"] = (
        f"/api/v1/plate-logs/{result['id']}/snapshot" if snapshot_ready else None
    )
    result["video_url"] = (
        f"/api/v1/plate-logs/{result['id']}/video" if video_ready else None
    )
    return result


def _media_path(
    log_id: int,
    kind: str,
    runtime: Runtime,
) -> tuple[Path, DetectionMediaStorage]:
    record = _get_log_or_404(log_id, runtime)
    key = record.get("snapshot_key" if kind == "snapshot" else "video_key")
    media = _media_storage(runtime)
    try:
        path = media.resolve(key, require_file=True)
    except InvalidMediaKey as exc:
        raise HTTPException(status_code=404, detail="پرونده رسانه یافت نشد") from exc
    if path is None or path.stat().st_size <= 0:
        raise HTTPException(status_code=404, detail="پرونده رسانه یافت نشد")
    return path, media


def _serve_media(path: Path, media: DetectionMediaStorage, download: bool) -> FileResponse:
    disposition = "attachment" if download else "inline"
    return FileResponse(
        str(path),
        media_type=media.content_type(path),
        headers={
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'{disposition}; filename="{path.name}"',
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "private, no-store",
        },
    )


def _get_log_or_404(log_id: int, runtime: Runtime) -> dict[str, Any]:
    value = runtime.plate_logs.get_log(log_id)
    if value is None:
        raise HTTPException(status_code=404, detail="لاگ تشخیص پلاک یافت نشد")
    return value


def _validate_plate(runtime: Runtime, plate_id: int | None) -> None:
    if plate_id is not None and runtime.car_plates.get(plate_id) is None:
        raise HTTPException(status_code=404, detail="پلاک ثبت‌شده یافت نشد")


@router.get("", response_model=list[PlateLogResponse])
def list_plate_logs(
    plate_id: int | None = Query(default=None, ge=1),
    plate_number: str | None = Query(default=None, max_length=32),
    source_uri: str | None = Query(default=None, max_length=500),
    static_video_id: int | None = Query(default=None, ge=1),
    source_type: PlateLogSourceType | None = None,
    detected_from: str | None = None,
    detected_to: str | None = None,
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
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="بازه زمانی ISO 8601 نامعتبر است") from exc
    media = _media_storage(runtime)
    return [
        _response(item, media, check_media=False)
        for item in runtime.plate_logs.list_logs(
            plate_id=plate_id,
            plate_number=plate_number,
            source_uri=source_uri,
            static_video_id=static_video_id,
            source_type=source_type.value if source_type else None,
            detected_from=detected_from,
            detected_to=detected_to,
            skip=skip,
            limit=limit,
        )
    ]


@router.get("/{log_id}", response_model=PlateLogResponse)
def get_plate_log(
    log_id: int,
    _: UserRecord = Depends(get_current_user),
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    return _response(_get_log_or_404(log_id, runtime), _media_storage(runtime))


@router.post("", response_model=PlateLogResponse, status_code=status.HTTP_201_CREATED)
def create_plate_log(
    log_data: PlateLogCreate,
    admin_user: UserRecord = Depends(require_permission("plate_logs.edit")),
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    payload = log_data.model_dump(mode="python")
    _validate_plate(runtime, payload.get("plate_id"))
    payload.update(
        source_type="manual",
        source_uri=None,
        static_video_id=None,
        created_by_user_id=admin_user.id,
    )
    return _response(
        runtime.plate_logs.create_log(payload), _media_storage(runtime)
    )


@router.patch("/{log_id}", response_model=PlateLogResponse)
def update_plate_log(
    log_id: int,
    log_data: PlateLogUpdate,
    user: UserRecord = Depends(require_permission("plate_logs.edit")),
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    _get_log_or_404(log_id, runtime)
    updates = log_data.model_dump(exclude_unset=True, mode="python")
    _validate_plate(runtime, updates.get("plate_id"))
    updates["updated_by_user_id"] = user.id
    result = runtime.plate_logs.update_log(log_id, updates)
    if result is None:
        raise HTTPException(status_code=404, detail="لاگ تشخیص پلاک یافت نشد")
    return _response(result, _media_storage(runtime))


@router.get("/{log_id}/snapshot")
def get_plate_snapshot(
    log_id: int,
    request: Request,
    download: bool = Query(default=False),
    _: UserRecord = Depends(get_current_user),
    runtime: Runtime = Depends(get_runtime),
) -> FileResponse:
    del request  # FileResponse handles range requests in the active Starlette runtime.
    path, media = _media_path(log_id, "snapshot", runtime)
    return _serve_media(path, media, download)


@router.get("/{log_id}/thumbnail")
def get_plate_thumbnail(
    log_id: int,
    _: UserRecord = Depends(get_current_user),
    runtime: Runtime = Depends(get_runtime),
) -> Response:
    record = _get_log_or_404(log_id, runtime)
    media = _media_storage(runtime)
    data = media.thumbnail_bytes(None, record.get("snapshot_key"))
    if not data:
        raise HTTPException(status_code=404, detail="تصویر بندانگشتی یافت نشد")
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "private, max-age=300",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/{log_id}/video")
def get_plate_video(
    log_id: int,
    request: Request,
    download: bool = Query(default=False),
    _: UserRecord = Depends(get_current_user),
    runtime: Runtime = Depends(get_runtime),
) -> FileResponse:
    del request  # FileResponse handles range requests in the active Starlette runtime.
    path, media = _media_path(log_id, "video", runtime)
    return _serve_media(path, media, download)


@router.delete("/{log_id}")
def delete_plate_log(
    log_id: int,
    _: UserRecord = Depends(require_permission("plate_logs.delete")),
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, str]:
    _get_log_or_404(log_id, runtime)
    if not runtime.plate_logs.delete_log(log_id):
        raise HTTPException(status_code=404, detail="لاگ تشخیص پلاک یافت نشد")
    return {"message": "لاگ تشخیص پلاک با موفقیت حذف شد"}
