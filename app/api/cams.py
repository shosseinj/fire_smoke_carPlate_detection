from __future__ import annotations

import base64
from typing import Literal

import cv2
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field, field_validator

from app.core.auth import accessible_scope_ids, enforce_scoped_permission, get_current_user
from app.core.auth_store import UserRecord
from app.core.cam_store import CamRecord
from app.core.common_schemas import UserBrief, resolve_user_brief
from app.core.jalali_utils import utc_iso_to_jalali_datetime
from app.runtime import Runtime

router = APIRouter(prefix="/api/v1/cams", tags=["Cameras"])


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


CamSourceType = Literal["usb", "rtsp", "other"]


class CamBase(BaseModel):
    camera_name: str = Field(min_length=1, max_length=512)
    camera_number: int = Field(ge=1)
    width: int = Field(ge=1, le=16384)
    high: int = Field(ge=1, le=16384)
    source_type: CamSourceType
    section_id: int = Field(ge=1)
    url: str = Field(min_length=1, max_length=2048)

    @field_validator("camera_name", "url")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("مقدار نمی‌تواند خالی باشد")
        return value


class CamCreate(CamBase):
    pass


class CamUpdate(BaseModel):
    camera_name: str | None = Field(default=None, min_length=1, max_length=512)
    camera_number: int | None = Field(default=None, ge=1)
    width: int | None = Field(default=None, ge=1, le=16384)
    high: int | None = Field(default=None, ge=1, le=16384)
    source_type: CamSourceType | None = None
    section_id: int | None = Field(default=None, ge=1)
    url: str | None = Field(default=None, min_length=1, max_length=2048)

    @field_validator("camera_name", "url")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("مقدار نمی‌تواند خالی باشد")
        return value


class CamResponse(CamBase):
    id: int
    created_at_utc: str
    updated_at_utc: str
    created_at_jalali: str = ""
    updated_at_jalali: str | None = None
    created_by: UserBrief | None = None
    updated_by: UserBrief | None = None


class CamListResponse(BaseModel):
    items: list[CamResponse]
    total: int
    skip: int
    limit: int


class CamHealthCheckRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)

    @field_validator("url")
    @classmethod
    def strip_url(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("آدرس نمی‌تواند خالی باشد")
        return value


class CamHealthCheckResponse(BaseModel):
    status: Literal["healthy", "unhealthy"]
    message: str
    snapshot: str | None = None


def _response(record: CamRecord, db=None) -> CamResponse:
    c = u = None
    if db is not None:
        c = resolve_user_brief(record.created_by, db)
        u = resolve_user_brief(record.updated_by, db)
    return CamResponse(
        id=record.id,
        camera_name=record.camera_name,
        camera_number=record.camera_number,
        width=record.width,
        high=record.high,
        source_type=record.source_type,
        section_id=record.section_id,
        url=record.url,
        created_at_utc=record.created_at_utc,
        updated_at_utc=record.updated_at_utc,
        created_at_jalali=utc_iso_to_jalali_datetime(record.created_at_utc) or "",
        updated_at_jalali=utc_iso_to_jalali_datetime(record.updated_at_utc),
        created_by=c,
        updated_by=u,
    )


def _open_capture(url: str):
    source: str | int = url
    if url.isdigit():
        source = int(url)
    elif url.lower().startswith("usb://") and url[6:].isdigit():
        source = int(url[6:])

    capture = cv2.VideoCapture()
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    capture.open(source)
    return capture


@router.post(
    "/health-check",
    response_model=CamHealthCheckResponse,
    summary="Check a camera URL and return one JPEG snapshot",
)
def check_cam_health(
    payload: CamHealthCheckRequest,
) -> CamHealthCheckResponse:
    capture = None
    try:
        capture = _open_capture(payload.url)
        if not capture.isOpened():
            return CamHealthCheckResponse(
                status="unhealthy",
                message="اتصال به دوربین امکان‌پذیر نیست",
                snapshot=None,
            )
        ok, frame = capture.read()
        if not ok or frame is None:
            return CamHealthCheckResponse(
                status="unhealthy",
                message="اتصال برقرار شد، اما هیچ فریمی خوانده نشد",
                snapshot=None,
            )
        encoded, buffer = cv2.imencode(".jpg", frame)
        if not encoded:
            return CamHealthCheckResponse(
                status="unhealthy",
                message="تبدیل فریم به تصویر JPEG امکان‌پذیر نیست",
                snapshot=None,
            )
        image_base64 = base64.b64encode(buffer).decode("ascii")
        return CamHealthCheckResponse(
            status="healthy",
            message="دوربین در دسترس است",
            snapshot=f"data:image/jpeg;base64,{image_base64}",
        )
    except Exception as exc:
        return CamHealthCheckResponse(
            status="unhealthy",
            message="بررسی سلامت دوربین با خطا مواجه شد",
            snapshot=None,
        )
    finally:
        if capture is not None:
            capture.release()


@router.post("", response_model=CamResponse, status_code=status.HTTP_201_CREATED)
def create_cam(
    payload: CamCreate,
    runtime: Runtime = Depends(get_runtime),
    current_user: UserRecord = Depends(get_current_user),
) -> CamResponse:
    enforce_scoped_permission(current_user, "cameras.create", "section", payload.section_id)
    try:
        record = runtime.cam_store.create(**payload.model_dump(), created_by=current_user.id)
    except ValueError as exc:
        detail = str(exc)
        code = (
            status.HTTP_404_NOT_FOUND
            if "not found" in detail.lower()
            else status.HTTP_409_CONFLICT
        )
        raise HTTPException(status_code=code, detail=detail) from exc
    with runtime.cam_store.database.connection() as conn:
        return _response(record, conn)


@router.get("", response_model=CamListResponse)
def list_cams(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    section_id: int | None = Query(default=None, ge=1),
    source_type: CamSourceType | None = Query(default=None),
    runtime: Runtime = Depends(get_runtime),
    current_user: UserRecord = Depends(get_current_user),
) -> CamListResponse:
    allowed_ids = accessible_scope_ids(current_user, "cameras.read", "camera")
    try:
        records, total = runtime.cam_store.list(
            offset=skip,
            limit=limit,
            section_id=section_id,
            source_type=source_type,
            allowed_ids=allowed_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    with runtime.cam_store.database.connection() as conn:
        items = [_response(record, conn) for record in records]
    return CamListResponse(
        items=items,
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get("/{cam_id}", response_model=CamResponse)
def get_cam(
    cam_id: int,
    runtime: Runtime = Depends(get_runtime),
    current_user: UserRecord = Depends(get_current_user),
) -> CamResponse:
    record = runtime.cam_store.get(cam_id)
    if record is None:
        raise HTTPException(status_code=404, detail="دوربین یافت نشد")
    enforce_scoped_permission(current_user, "cameras.read", "camera", cam_id)
    with runtime.cam_store.database.connection() as conn:
        return _response(record, conn)


@router.patch("/{cam_id}", response_model=CamResponse)
def update_cam(
    cam_id: int,
    payload: CamUpdate,
    runtime: Runtime = Depends(get_runtime),
    current_user: UserRecord = Depends(get_current_user),
) -> CamResponse:
    existing = runtime.cam_store.get(cam_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="دوربین یافت نشد")
    enforce_scoped_permission(current_user, "cameras.edit", "camera", cam_id)
    changes = payload.model_dump(exclude_unset=True)
    if changes.get("section_id") not in (None, existing.section_id):
        enforce_scoped_permission(current_user, "cameras.edit", "section", changes["section_id"])
    if not changes:
        raise HTTPException(status_code=422, detail="هیچ فیلدی برای به‌روزرسانی وارد نشده است")
    if any(value is None for value in changes.values()):
        raise HTTPException(status_code=422, detail="فیلدهای دوربین نمی‌توانند خالی باشند")
    changes["updated_by"] = current_user.id
    try:
        record = runtime.cam_store.update(cam_id, **changes)
    except ValueError as exc:
        detail = str(exc)
        code = (
            status.HTTP_404_NOT_FOUND
            if "not found" in detail.lower()
            else status.HTTP_409_CONFLICT
        )
        raise HTTPException(status_code=code, detail=detail) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="دوربین یافت نشد")
    with runtime.cam_store.database.connection() as conn:
        return _response(record, conn)


@router.delete("/{cam_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_cam(
    cam_id: int,
    runtime: Runtime = Depends(get_runtime),
    current_user: UserRecord = Depends(get_current_user),
) -> Response:
    if runtime.cam_store.get(cam_id) is None:
        raise HTTPException(status_code=404, detail="دوربین یافت نشد")
    enforce_scoped_permission(current_user, "cameras.delete", "camera", cam_id)
    try:
        deleted = runtime.cam_store.delete(cam_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="دوربین یافت نشد")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
