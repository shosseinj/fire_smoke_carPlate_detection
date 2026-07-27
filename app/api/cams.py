from __future__ import annotations

import base64
from typing import Literal

import cv2
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field, field_validator

from app.core.auth import require_role
from app.core.auth_store import UserRecord
from app.core.cam_store import CamRecord
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
            raise ValueError("value cannot be blank")
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
            raise ValueError("value cannot be blank")
        return value


class CamResponse(CamBase):
    id: int
    created_at_utc: str
    updated_at_utc: str
    created_at_jalali: str = ""
    updated_at_jalali: str | None = None


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
            raise ValueError("url cannot be blank")
        return value


class CamHealthCheckResponse(BaseModel):
    status: Literal["healthy", "unhealthy"]
    message: str
    snapshot: str | None = None


def _response(record: CamRecord) -> CamResponse:
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
                message="Unable to connect to the camera",
                snapshot=None,
            )
        ok, frame = capture.read()
        if not ok or frame is None:
            return CamHealthCheckResponse(
                status="unhealthy",
                message="Connected, but no frame could be read",
                snapshot=None,
            )
        encoded, buffer = cv2.imencode(".jpg", frame)
        if not encoded:
            return CamHealthCheckResponse(
                status="unhealthy",
                message="The frame could not be encoded as JPEG",
                snapshot=None,
            )
        image_base64 = base64.b64encode(buffer).decode("ascii")
        return CamHealthCheckResponse(
            status="healthy",
            message="Camera is reachable",
            snapshot=f"data:image/jpeg;base64,{image_base64}",
        )
    except Exception as exc:
        return CamHealthCheckResponse(
            status="unhealthy",
            message=f"Camera health check failed: {exc}",
            snapshot=None,
        )
    finally:
        if capture is not None:
            capture.release()


@router.post("", response_model=CamResponse, status_code=status.HTTP_201_CREATED)
def create_cam(
    payload: CamCreate,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> CamResponse:
    try:
        record = runtime.cam_store.create(**payload.model_dump())
    except ValueError as exc:
        detail = str(exc)
        code = (
            status.HTTP_404_NOT_FOUND
            if "not found" in detail.lower()
            else status.HTTP_409_CONFLICT
        )
        raise HTTPException(status_code=code, detail=detail) from exc
    return _response(record)


@router.get("", response_model=CamListResponse)
def list_cams(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    section_id: int | None = Query(default=None, ge=1),
    source_type: CamSourceType | None = Query(default=None),
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> CamListResponse:
    try:
        records, total = runtime.cam_store.list(
            offset=skip,
            limit=limit,
            section_id=section_id,
            source_type=source_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return CamListResponse(
        items=[_response(record) for record in records],
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get("/{cam_id}", response_model=CamResponse)
def get_cam(
    cam_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> CamResponse:
    record = runtime.cam_store.get(cam_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Cam not found")
    return _response(record)


@router.patch("/{cam_id}", response_model=CamResponse)
def update_cam(
    cam_id: int,
    payload: CamUpdate,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> CamResponse:
    changes = payload.model_dump(exclude_unset=True)
    if not changes:
        raise HTTPException(status_code=422, detail="No fields to update")
    if any(value is None for value in changes.values()):
        raise HTTPException(status_code=422, detail="Cam fields cannot be null")
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
        raise HTTPException(status_code=404, detail="Cam not found")
    return _response(record)


@router.delete("/{cam_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_cam(
    cam_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("superuser")),
) -> Response:
    try:
        deleted = runtime.cam_store.delete(cam_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Cam not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
