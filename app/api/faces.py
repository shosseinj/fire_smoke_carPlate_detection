from __future__ import annotations

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.processors.face_recognition import FaceRecognitionProcessor
from app.runtime import Runtime


router = APIRouter(prefix="/api/v1/faces", tags=["face-recognition"])


class FaceQualitySettingsPatch(BaseModel):
    quality_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    blur_threshold: float | None = Field(default=None, ge=0.0)
    min_face_width: int | None = Field(default=None, ge=1, le=4096)
    min_face_height: int | None = Field(default=None, ge=1, le=4096)
    min_eye_distance: float | None = Field(default=None, ge=0.0)
    max_abs_yaw: float | None = Field(default=None, gt=0.0, le=90.0)
    max_abs_pitch: float | None = Field(default=None, gt=0.0, le=90.0)
    max_abs_roll: float | None = Field(default=None, gt=0.0, le=90.0)
    require_landmarks: bool | None = None
    human_pose_enabled: bool | None = None
    human_pose_min_keypoints: int | None = Field(default=None, ge=1, le=17)
    human_pose_keypoint_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    recognition_quality_weight: float | None = Field(default=None, ge=0.0, le=1.0)


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


def _processor(runtime: Runtime) -> FaceRecognitionProcessor:
    processor = runtime.face_processor
    if not isinstance(processor, FaceRecognitionProcessor):
        raise HTTPException(
            status_code=503,
            detail="ثبت چهره در حالت PROCESSOR_MODE=mock غیرفعال است",
        )
    return processor


def _service_unavailable(exc: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail=f"تشخیص چهره آماده نیست: {exc}")


@router.get("/status", summary="Face model, tracker, batch, and Qdrant status")
def status(runtime: Runtime = Depends(get_runtime)) -> dict:
    return runtime.face_processor.status()


@router.get(
    "/quality-settings",
    summary="Read persistent landmark, blur, pose, and quality recognition gates",
)
def quality_settings(runtime: Runtime = Depends(get_runtime)) -> dict:
    return runtime.face_quality_settings.as_dict()


@router.patch(
    "/quality-settings",
    summary="Update best-face selection and recognition quality gates online",
)
def update_quality_settings(
    payload: FaceQualitySettingsPatch,
    runtime: Runtime = Depends(get_runtime),
) -> dict:
    changes = payload.model_dump(exclude_unset=True, exclude_none=True)
    try:
        runtime.face_quality_settings.update(changes)
        if isinstance(runtime.face_processor, FaceRecognitionProcessor):
            runtime.face_processor.update_quality_settings(changes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return runtime.face_quality_settings.as_dict()


@router.post("/enroll", summary="Enroll one face image in Qdrant")
async def enroll(
    runtime: Runtime = Depends(get_runtime),
    person: str = Form(..., min_length=1),
    ref_img_id: str | None = Form(default=None),
    file: UploadFile = File(...),
) -> dict:
    raw = await file.read()
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail="فایل باید تصویر JPEG یا PNG معتبر باشد")
    try:
        return await run_in_threadpool(
            _processor(runtime).enroll,
            image,
            person=person,
            ref_img_id=ref_img_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError, ImportError) as exc:
        raise _service_unavailable(exc) from exc


@router.get("/identities", summary="List enrolled Qdrant identities")
async def identities(
    runtime: Runtime = Depends(get_runtime),
    limit: int = Query(default=1000, ge=1, le=10000),
) -> dict:
    try:
        items = await run_in_threadpool(_processor(runtime).identities, limit)
    except (FileNotFoundError, RuntimeError, ImportError) as exc:
        raise _service_unavailable(exc) from exc
    return {"items": items, "count": len(items)}


@router.delete("/identities/{person}", summary="Delete all embeddings for a person")
async def delete_identity(
    person: str,
    runtime: Runtime = Depends(get_runtime),
) -> dict:
    if not person.strip():
        raise HTTPException(status_code=422, detail="نام شخص نمی‌تواند خالی باشد")
    try:
        deleted = await run_in_threadpool(_processor(runtime).delete_person, person)
    except (FileNotFoundError, RuntimeError, ImportError) as exc:
        raise _service_unavailable(exc) from exc
    return {"person": person, "deleted_embeddings": deleted}
