from __future__ import annotations

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from app.processors.face_recognition import FaceRecognitionProcessor
from app.runtime import Runtime


router = APIRouter(prefix="/api/v1/faces", tags=["face-recognition"])


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


def _processor(runtime: Runtime) -> FaceRecognitionProcessor:
    processor = runtime.face_processor
    if not isinstance(processor, FaceRecognitionProcessor):
        raise HTTPException(
            status_code=503,
            detail="Face enrollment is unavailable while PROCESSOR_MODE=mock",
        )
    return processor


def _service_unavailable(exc: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail=f"Face recognition is not ready: {exc}")


@router.get("/status", summary="Face model, tracker, batch, and Qdrant status")
def status(runtime: Runtime = Depends(get_runtime)) -> dict:
    return runtime.face_processor.status()


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
        raise HTTPException(status_code=422, detail="file must be a valid JPEG or PNG image")
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
        raise HTTPException(status_code=422, detail="person cannot be blank")
    try:
        deleted = await run_in_threadpool(_processor(runtime).delete_person, person)
    except (FileNotFoundError, RuntimeError, ImportError) as exc:
        raise _service_unavailable(exc) from exc
    return {"person": person, "deleted_embeddings": deleted}
