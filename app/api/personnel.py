from __future__ import annotations

import logging

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.core.auth import require_role
from app.core.personnel_store import (
    PersonnelImageRecord,
    PersonnelRecord,
    PersonnelStore,
)
from app.processors.face_recognition import FaceRecognitionProcessor
from app.runtime import Runtime
from app.core.auth_store import UserRecord

LOGGER = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/api/v1/personnel", tags=["personnel"])


def get_runtime() -> Runtime:
    from app.main import runtime
    return runtime


def _store(runtime: Runtime) -> PersonnelStore:
    return runtime.personnel_store


def _face_processor(runtime: Runtime) -> FaceRecognitionProcessor | None:
    processor = runtime.face_processor
    if isinstance(processor, FaceRecognitionProcessor):
        return processor
    return None


# ── Pydantic models ─────────────────────────────────────────────────


class PersonnelCreateRequest(BaseModel):
    fname: str = Field(..., min_length=1, max_length=200)
    lname: str = Field(..., min_length=1, max_length=200)
    national_code: str = Field(..., min_length=1, max_length=20)
    employee_type: str = Field(default="unknown")
    degree: str | None = Field(default=None, max_length=200)


class PersonnelUpdateRequest(BaseModel):
    fname: str | None = Field(default=None, min_length=1, max_length=200)
    lname: str | None = Field(default=None, min_length=1, max_length=200)
    national_code: str | None = Field(default=None, min_length=1, max_length=20)
    employee_type: str | None = Field(default=None)
    degree: str | None = Field(default=None, max_length=200)


class PersonnelResponse(BaseModel):
    id: int
    fname: str
    lname: str
    national_code: str
    employee_type: str
    degree: str | None = None
    last_seen: str | None = None
    created_at_utc: str
    updated_at_utc: str


class PersonnelWithImagesResponse(PersonnelResponse):
    images: list[dict] = Field(default_factory=list)


class PersonnelImageResponse(BaseModel):
    id: int
    personnel_id: int
    storage_key: str
    description: str | None = None
    is_primary: bool
    uploaded_at_utc: str
    embedding_id: str | None = None
    url: str = ""
    face_status: int = 0
    cropped_face_key: str | None = None


def _personnel_to_response(p: PersonnelRecord) -> PersonnelResponse:
    return PersonnelResponse(
        id=p.id,
        fname=p.fname,
        lname=p.lname,
        national_code=p.national_code,
        employee_type=p.employee_type,
        degree=p.degree,
        last_seen=p.last_seen,
        created_at_utc=p.created_at_utc,
        updated_at_utc=p.updated_at_utc,
    )


def _image_to_response(img: PersonnelImageRecord) -> PersonnelImageResponse:
    from app.config import settings
    base_url = getattr(settings, "external_base_url", "")
    url = f"{base_url}/media/{img.storage_key}"
    return PersonnelImageResponse(
        id=img.id,
        personnel_id=img.personnel_id,
        storage_key=img.storage_key,
        description=img.description,
        is_primary=img.is_primary,
        uploaded_at_utc=img.uploaded_at_utc,
        embedding_id=img.embedding_id,
        url=url,
    )


# ── Personnel CRUD ──────────────────────────────────────────────────


@router.get("/", summary="List all personnel records")
def list_personnel(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=1000),
    employee_type: str | None = Query(default=None),
    search: str | None = Query(default=None, description="Search by fname, lname, or national_code"),
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> dict:
    records, total = _store(runtime).list(
        offset=offset,
        limit=limit,
        employee_type=employee_type,
        search=search,
    )
    return {
        "items": [_personnel_to_response(r) for r in records],
        "count": len(records),
        "total": total,
    }


@router.post("/", summary="Create a new personnel record", status_code=status.HTTP_201_CREATED)
def create_personnel(
    payload: PersonnelCreateRequest,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> PersonnelResponse:
    try:
        record = _store(runtime).create(
            fname=payload.fname,
            lname=payload.lname,
            national_code=payload.national_code,
            employee_type=payload.employee_type,
            degree=payload.degree,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return _personnel_to_response(record)


@router.get("/search/{national_code}", summary="Search personnel by national code")
def search_personnel(
    national_code: str,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> PersonnelResponse:
    record = _store(runtime).get_by_national_code(national_code)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Personnel not found")
    return _personnel_to_response(record)


@router.get("/with-images", summary="List personnel records with their images")
def list_personnel_with_images(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> dict:
    items, total = _store(runtime).list_with_images(offset=offset, limit=limit)
    # Convert dataclass dicts to response dicts with URL enrichment
    from app.config import settings
    base_url = getattr(settings, "external_base_url", "")
    enriched: list[dict] = []
    for item in items:
        for img in item.get("images", []):
            if "storage_key" in img:
                img["url"] = f"{base_url}/media/{img['storage_key']}"
        enriched.append(item)
    return {"items": enriched, "count": len(enriched), "total": total}


# ── Import / Export / Bulk ───────────────────────────────────────────
# NOTE: these routes must be defined BEFORE /{personnel_id} to avoid
# FastAPI matching static names like "import-template" as personnel_id


@router.post(
    "/import-excel",
    summary="Import personnel records from an Excel file",
)
async def import_excel(
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
    file: UploadFile = File(...),
) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Empty file")
    try:
        result = await run_in_threadpool(_store(runtime).import_from_excel, raw)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return result


@router.get(
    "/import-template",
    summary="Download an Excel import template",
)
def import_template(
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> Response:
    data = _store(runtime).generate_import_template()
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=personnel_import_template.xlsx"},
    )


@router.post(
    "/upload-personnel-zip",
    summary="Upload a ZIP file with personnel data and images",
)
async def upload_personnel_zip(
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
    file: UploadFile = File(...),
) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Empty file")
    try:
        store = _store(runtime)
        fp = _face_processor(runtime)
        result = await run_in_threadpool(store.upload_personnel_zip, raw, fp)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return result


# ── Personnel CRUD by ID ─────────────────────────────────────────────
# NOTE: /{personnel_id} routes must be defined AFTER all static paths


@router.get("/{personnel_id}", summary="Get a personnel record by ID")
def get_personnel(
    personnel_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> PersonnelResponse:
    record = _store(runtime).get(personnel_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Personnel not found")
    return _personnel_to_response(record)


@router.put("/{personnel_id}", summary="Update a personnel record")
def update_personnel(
    personnel_id: int,
    payload: PersonnelUpdateRequest,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> PersonnelResponse:
    changes = payload.model_dump(exclude_unset=True, exclude_none=True)
    if not changes:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No fields to update")
    try:
        record = _store(runtime).update(
            personnel_id=personnel_id,
            **changes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Personnel not found")
    return _personnel_to_response(record)


@router.delete("/{personnel_id}", summary="Delete a personnel record")
def delete_personnel(
    personnel_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> dict:
    deleted = _store(runtime).delete(personnel_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Personnel not found")
    return {"deleted": True, "personnel_id": personnel_id}


# ── Personnel Image endpoints ───────────────────────────────────────


@router.get(
    "/{personnel_id}/images",
    summary="List all images for a personnel record",
)
def list_personnel_images(
    personnel_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> dict:
    store = _store(runtime)
    # Verify personnel exists
    if store.get(personnel_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Personnel not found")
    images = store.list_images(personnel_id)
    return {
        "items": [_image_to_response(img) for img in images],
        "count": len(images),
    }


@router.post(
    "/{personnel_id}/images",
    summary="Upload a face image for a personnel record",
    status_code=status.HTTP_201_CREATED,
)
async def upload_personnel_image(
    personnel_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
    file: UploadFile = File(...),
    description: str | None = Form(default=None),
) -> PersonnelImageResponse:
    store = _store(runtime)
    # Verify personnel exists
    person = store.get(personnel_id)
    if person is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Personnel not found")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Empty file")

    # Always save snapshot first
    storage_key = store._save_image_file(personnel_id, raw, file.filename or "image.jpg")

    embedding_id: str | None = None
    face_status = 0
    cropped_face_key: str | None = None
    face_processor = _face_processor(runtime)

    if face_processor is not None:
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            try:
                raw_face_count = await run_in_threadpool(face_processor.count_faces, image)
            except Exception:
                raw_face_count = 0

            if raw_face_count == 0:
                face_status = 0
            elif raw_face_count >= 2:
                face_status = 2
            else:
                # Exactly one raw face — try enrollment and save cropped face
                try:
                    enroll_result = await run_in_threadpool(
                        face_processor.enroll,
                        image,
                        person=person.national_code,
                        ref_img_id=f"{personnel_id}",
                    )
                    embedding_id = enroll_result.get("point_id")
                    face_status = 1

                    # Save aligned cropped face
                    success, aligned = await run_in_threadpool(
                        face_processor.get_aligned_face, image
                    )
                    if success and aligned is not None:
                        success_enc, encoded = cv2.imencode(".jpg", aligned)
                        if success_enc:
                            cropped_face_key = store._save_cropped_face_file(
                                personnel_id, encoded.tobytes(), file.filename or "face.jpg"
                            )
                except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
                    LOGGER.warning(
                        "Face enrollment failed for personnel %s: %s", personnel_id, exc
                    )
                    face_status = 0
        else:
            LOGGER.warning("Could not decode uploaded image for face enrollment")

    # Create image record
    try:
        img_record = store.create_image(
            personnel_id=personnel_id,
            storage_key=storage_key,
            description=description,
            embedding_id=embedding_id,
        )
    except ValueError as exc:
        # Rollback file save
        store._delete_storage_file(storage_key)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    resp = _image_to_response(img_record)
    resp.face_status = face_status
    resp.cropped_face_key = cropped_face_key
    return resp


# ── Personnel Images (standalone) ────────────────────────────────────


@router.get(
    "/images/{image_id}",
    summary="Get an image record by ID",
    tags=["personnel-images"],
)
def get_personnel_image(
    image_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> PersonnelImageResponse:
    img = _store(runtime).get_image(image_id)
    if img is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    return _image_to_response(img)


@router.delete(
    "/images/{image_id}",
    summary="Delete a personnel image",
    tags=["personnel-images"],
)
def delete_personnel_image(
    image_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> dict:
    deleted = _store(runtime).delete_image(image_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    return {"deleted": True, "image_id": image_id}


@router.put(
    "/images/{image_id}/set-primary",
    summary="Set an image as the primary image for its personnel",
    tags=["personnel-images"],
)
def set_primary_image(
    image_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> PersonnelImageResponse:
    img = _store(runtime).set_primary_image(image_id)
    if img is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    return _image_to_response(img)
