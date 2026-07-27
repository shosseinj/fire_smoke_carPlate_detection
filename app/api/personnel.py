from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.core.auth import require_role
from app.core.common_schemas import UserBrief, resolve_user_brief
from app.core.personnel_store import (
    PersonnelImageRecord,
    PersonnelRecord,
    PersonnelStore,
    dataclass_to_dict,
)
from app.core.personnel_image_service import (
    PersonnelImageProcessor,
    validate_uploaded_image,
)
from app.processors.face_recognition import FaceRecognitionProcessor
from app.runtime import Runtime
from app.core.auth_store import UserRecord
from typing import Optional

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


def _image_processor(runtime: Runtime) -> PersonnelImageProcessor:
    return PersonnelImageProcessor(_face_processor(runtime))


# ── Pydantic models ─────────────────────────────────────────────────

class PersonnelCreateRequest(BaseModel):
    fname: str = Field(min_length=1, max_length=200)
    lname: str = Field(min_length=1, max_length=200)
    national_code: str = Field(min_length=1, max_length=20)
    employee_type: Optional[str] = None
    degree: Optional[str] = Field(default=None, max_length=200)
    department_id: Optional[int] = None
    shift_id: Optional[int] = None


class PersonnelUpdateRequest(BaseModel):
    fname: Optional[str] = Field(default=None, min_length=1, max_length=200)
    lname: Optional[str] = Field(default=None, min_length=1, max_length=200)
    national_code: Optional[str] = Field(default=None, min_length=1, max_length=20)
    employee_type: Optional[str] = None
    degree: Optional[str] = Field(default=None, max_length=200)
    department_id: Optional[int] = None
    shift_id: Optional[int] = None


class SimplePersonnelResponse(BaseModel):
    id: int
    fname: str
    lname: str
    national_code: str
    employee_type: Optional[str] = None
    department_name: Optional[str] = None
    shift_name: Optional[str] = None
    degree: Optional[str] = None
    created_at: datetime
    created_at_jalali: str = ""
    created_by: UserBrief | None = None
    updated_by: UserBrief | None = None


class PersonnelImageResponse(BaseModel):
    id: int
    image_base64: Optional[str] = None
    personnel_id: int
    is_primary: bool
    uploaded_at: Optional[datetime] = None


# ── Converters ───────────────────────────────────────────────────

def _personnel_simple(p: PersonnelRecord, store: PersonnelStore) -> SimplePersonnelResponse:
    from app.core.jalali_utils import utc_iso_to_jalali_datetime
    c = u = None
    if p.created_by is not None or p.updated_by is not None:
        with store.database.connection() as conn:
            c = resolve_user_brief(p.created_by, conn)
            u = resolve_user_brief(p.updated_by, conn)
    return SimplePersonnelResponse(
        id=p.id,
        fname=p.fname,
        lname=p.lname,
        national_code=p.national_code,
        employee_type=p.employee_type,
        department_name=store._resolve_department_name(p.department_id),
        shift_name=store._resolve_shift_name(p.shift_id),
        degree=p.degree,
        created_at=p.created_at_utc,
        created_at_jalali=utc_iso_to_jalali_datetime(p.created_at_utc) or "",
        created_by=c,
        updated_by=u,
    )


def _image_to_base64_response(img: PersonnelImageRecord, store: PersonnelStore) -> PersonnelImageResponse:
    return PersonnelImageResponse(
        id=img.id,
        image_base64=store.read_image_base64(img.storage_key),
        personnel_id=img.personnel_id,
        is_primary=img.is_primary,
        uploaded_at=img.uploaded_at_utc,
    )


# ── Personnel CRUD ──────────────────────────────────────────────────


@router.get("/", summary="List all personnel records")
def list_personnel(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    employee_type: str | None = Query(default=None),
    search: str | None = Query(default=None, description="Search by fname, lname, or national_code"),
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> list:
    store = _store(runtime)
    records, _ = store.list(offset=skip, limit=limit, employee_type=employee_type, search=search)
    return [_personnel_simple(r, store) for r in records]


@router.post("/", summary="Create a new personnel record", status_code=status.HTTP_201_CREATED)
def create_personnel(
    payload: PersonnelCreateRequest,
    runtime: Runtime = Depends(get_runtime),
    current_user: UserRecord = Depends(require_role("admin")),
) -> SimplePersonnelResponse:
    store = _store(runtime)
    try:
        record = store.create(
            fname=payload.fname,
            lname=payload.lname,
            national_code=payload.national_code,
            employee_type=payload.employee_type,
            degree=payload.degree,
            shift_id=payload.shift_id,
            department_id=payload.department_id,
            created_by=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return _personnel_simple(record, store)


@router.get("/search/{national_code}", summary="Search personnel by national code")
def search_personnel(
    national_code: str,
    runtime: Runtime = Depends(get_runtime),
) -> SimplePersonnelResponse | None:
    store = _store(runtime)
    record = store.get_by_national_code(national_code)
    if record is None:
        return None
    return _personnel_simple(record, store)


# ── Import / Export / Bulk ───────────────────────────────────────────
# NOTE: these routes must be defined BEFORE /{personnel_id} to avoid
# FastAPI matching static names like "import-template" as personnel_id


@router.post("/with-images", summary="Create personnel with images (legacy)", status_code=status.HTTP_201_CREATED)
async def create_personnel_with_images(
    runtime: Runtime = Depends(get_runtime),
    fname: str = Form(...),
    lname: str = Form(...),
    national_code: str = Form(...),
    employee_type: str = Form(default="unknown"),
    degree: str | None = Form(default=None),
    departmen_id: int | None = Form(default=None),
    department_id: int | None = Form(default=None),
    images: list[UploadFile] = File(...),
    enable_cropping: bool = Form(default=False),
) -> Any:
    store = _store(runtime)
    dept_id = department_id if department_id is not None else departmen_id
    try:
        person = store.create(
            fname=fname,
            lname=lname,
            national_code=national_code,
            employee_type=employee_type,
            degree=degree,
            department_id=dept_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    processor = _image_processor(runtime)
    person_name = f"{person.fname} {person.lname}"
    saved_images: list[PersonnelImageRecord] = []
    errors: list[str] = []

    for img_file in images:
        raw = await img_file.read()
        if not raw:
            errors.append(f"{img_file.filename}: empty file")
            continue
        validation = validate_uploaded_image(raw, img_file.filename or "image.jpg", img_file.content_type)
        if not validation.valid:
            errors.append(f"{img_file.filename}: {validation.failure_message}")
            continue
        storage_key = store._save_image_file(person.id, raw, img_file.filename or "image.jpg", person.national_code)
        img_record = store.create_image(
            personnel_id=person.id,
            storage_key=storage_key,
            embedding_id=None,
            is_primary=len(saved_images) == 0,
        )
        embedding_id: str | None = None
        process_result = processor.process_image(
            raw,
            person_name=person.national_code,
            ref_img_id=str(img_record.id),
            enable_cropping=enable_cropping,
        )
        if process_result.success:
            embedding_id = process_result.vector_point_id
            store.update_image_embedding(img_record.id, embedding_id)
        else:
            errors.append(f"{img_file.filename}: {process_result.failure_message}")
        saved_images.append(img_record)

    if not saved_images and not errors:
        store.delete(person.id)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="All images failed processing")

    imgs = [_image_to_base64_response(img, store).model_dump() for img in saved_images]
    primary_img = next((img for img in saved_images if img.is_primary), None)
    return {
        **_personnel_simple(person, store).model_dump(),
        "rooms": [],
        "images": imgs,
        "primary_image": _image_to_base64_response(primary_img, store).model_dump() if primary_img else None,
    }


@router.post(
    "/import-excel",
    summary="Import personnel records from an Excel file",
)
async def import_excel(
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
    file: UploadFile = File(...),
    update_existing: bool = Form(default=False),
    skip_invalid_rows: bool = Form(default=True),
) -> dict:
    store = _store(runtime)
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Empty file")
    try:
        result = await run_in_threadpool(
            store.import_from_excel,
            raw,
            update_existing=update_existing,
            skip_invalid_rows=skip_invalid_rows,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    total = result["created"] + result["skipped"] + len(result["errors"])
    return {
        "summary": {
            "total_rows": total,
            "successful": result["created"],
            "failed": len(result["errors"]),
            "skipped": result["skipped"],
            "created": result["created"],
            "updated": 0,
        },
        "successful_rows": [],
        "failed_rows": [{"row": e["row"], "error": e["error"]} for e in result["errors"]],
        "skipped_rows": [],
    }


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
    file: UploadFile = File(...),
    skip_invalid_national_codes: bool = Form(default=True),
    enable_cropping: bool = Form(default=False),
) -> Any:
    store = _store(runtime)
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Empty file")
    try:
        fp = _face_processor(runtime)
        result = await run_in_threadpool(store.upload_personnel_zip, raw, fp, enable_cropping)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    summary = {
        "total_images_in_zip": result.get("total_images_in_zip", 0),
        "total_persons": result.get("created_personnel", 0),
        "total_images_saved": result.get("created_images", 0),
        "total_errors": len(result.get("errors", [])),
        "total_failed": result.get("total_failed", 0),
        "qdrant_enrolled_count": result.get("qdrant_enrolled", 0),
        "face_stats": result.get("face_counts", {}),
    }
    return {
        "success": len(result.get("errors", [])) == 0,
        "filename": file.filename or "file.zip",
        "message": (
            f"Processed {summary['total_images_in_zip']} images: "
            f"{summary['total_images_saved']} saved, "
            f"{summary['qdrant_enrolled_count']} enrolled in Qdrant, "
            f"{summary['total_failed']} failed"
        ),
        "summary": summary,
        "details": result.get("image_details", []),
        "errors": result.get("errors", []),
    }


# ── Personnel CRUD by ID ─────────────────────────────────────────────
# NOTE: /{personnel_id} routes must be defined AFTER all static paths


@router.get("/{personnel_id}", summary="Get a personnel record by ID")
def get_personnel(
    personnel_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> SimplePersonnelResponse:
    store = _store(runtime)
    record = store.get(personnel_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Personnel not found")
    return _personnel_simple(record, store)


@router.put("/{personnel_id}", summary="Update a personnel record")
def update_personnel(
    personnel_id: int,
    payload: PersonnelUpdateRequest,
    runtime: Runtime = Depends(get_runtime),
    current_user: UserRecord = Depends(require_role("admin")),
) -> SimplePersonnelResponse:
    store = _store(runtime)
    changes = payload.model_dump(exclude_unset=True, exclude_none=True)
    if not changes:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No fields to update")
    changes["updated_by"] = current_user.id
    try:
        record = store.update(
            personnel_id=personnel_id,
            **changes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Personnel not found")
    return _personnel_simple(record, store)


@router.delete("/{personnel_id}", summary="Delete a personnel record", status_code=status.HTTP_204_NO_CONTENT)
def delete_personnel(
    personnel_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("superuser")),
) -> Response:
    store = _store(runtime)
    personnel = store.get(personnel_id)
    if personnel is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Personnel not found")
    images = store.list_images(personnel_id)
    embedding_ids = [img.embedding_id for img in images if img.embedding_id]
    if embedding_ids:
        processor = _face_processor(runtime)
        if processor is not None:
            processor.delete_points(embedding_ids)
            processor.delete_person(personnel.national_code)
            processor.delete_person(f"{personnel.fname} {personnel.lname}")
    deleted = store.delete(personnel_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Personnel not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ── Personnel Image endpoints ───────────────────────────────────────


@router.get(
    "/{personnel_id}/images",
    summary="List all images for a personnel record",
)
def list_personnel_images(
    personnel_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> list:
    store = _store(runtime)
    if store.get(personnel_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Personnel not found")
    images = store.list_images(personnel_id)
    return [_image_to_base64_response(img, store) for img in images]


@router.post(
    "/{personnel_id}/images",
    summary="Upload face image(s) for a personnel record",
    status_code=status.HTTP_201_CREATED,
)
async def upload_personnel_images(
    personnel_id: int,
    files: Annotated[
        list[UploadFile] | None,
        File(
            description="JPEG, PNG, or BMP image files",
            media_type="image/*",
            json_schema_extra={
                "items": {
                    "type": "string",
                    "format": "binary",
                    "contentMediaType": "image/*",
                }
            },
        ),
    ] = None,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
    enable_cropping: bool = Form(default=False),
    is_primary: str | None = Form(default=None),
) -> dict:
    store = _store(runtime)
    person = store.get(personnel_id)
    if person is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Personnel not found")

    upload_files: list[UploadFile] = []
    if files is not None:
        upload_files = files if isinstance(files, list) else [files]

    if not upload_files:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No files provided")

    face_processor = _face_processor(runtime)
    processor = _image_processor(runtime)
    results: list[dict] = []
    saved_images: list[PersonnelImageRecord] = []
    all_failed = True
    is_primary_val: bool | None = None
    if is_primary is not None:
        is_primary_val = is_primary.strip().lower() in ("true", "1", "yes")

    for img_file in upload_files:
        raw = await img_file.read()
        if not raw:
            results.append({"success": False, "failure_code": "empty_file", "failure_message": "File is empty", "image": None})
            continue

        validation = validate_uploaded_image(raw, img_file.filename or "image.jpg", img_file.content_type)
        if not validation.valid:
            results.append({"success": False, "failure_code": validation.failure_code, "failure_message": validation.failure_message, "image": None})
            continue

        storage_key = store._save_image_file(personnel_id, raw, img_file.filename or "image.jpg", person.national_code)
        embedding_id: str | None = None
        face_status = 0
        cropped_face_key: str | None = None

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
                    process_result = processor.process_image(raw, person_name=person.national_code, ref_img_id=f"{personnel_id}", enable_cropping=enable_cropping)
                    if process_result.success:
                        embedding_id = process_result.vector_point_id
                        face_status = 1
                        if enable_cropping:
                            try:
                                success, aligned = await run_in_threadpool(face_processor.get_aligned_face, image)
                                if success and aligned is not None:
                                    success_enc, encoded = cv2.imencode(".jpg", aligned)
                                    if success_enc:
                                        cropped_face_key = store._save_cropped_face_file(personnel_id, encoded.tobytes(), img_file.filename or "face.jpg", person.national_code)
                            except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
                                LOGGER.warning("Cropped face save failed for personnel %s: %s", personnel_id, exc)
                    else:
                        face_status = 0
            else:
                LOGGER.warning("Could not decode uploaded image for face enrollment")
        else:
            process_result = processor.process_image(raw, person_name=person.national_code, ref_img_id=f"{personnel_id}", enable_cropping=enable_cropping)
            if process_result.success:
                embedding_id = process_result.vector_point_id

        try:
            img_record = store.create_image(personnel_id=personnel_id, storage_key=storage_key, description=None, embedding_id=embedding_id, is_primary=is_primary_val)
            saved_images.append(img_record)
            all_failed = False
            img_dict = dataclass_to_dict(img_record)
            img_dict["face_status"] = face_status
            img_dict["cropped_face_key"] = cropped_face_key
            results.append({"success": True, "failure_code": None, "failure_message": None, "image": img_dict})
        except ValueError as exc:
            store._delete_storage_file(storage_key)
            results.append({"success": False, "failure_code": "storage_error", "failure_message": str(exc), "image": None})

    if all_failed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"total_success": 0, "total_failed": len(upload_files), "results": results, "personnel": dataclass_to_dict(person)},
        )

    person_dict = dataclass_to_dict(person)
    person_dict["images"] = []
    for img in saved_images:
        img_dict = dataclass_to_dict(img)
        for r in results:
            if r.get("image") and r["image"].get("id") == img.id:
                img_dict["face_status"] = r["image"].get("face_status", 0)
                img_dict["cropped_face_key"] = r["image"].get("cropped_face_key")
                break
        else:
            img_dict["face_status"] = 0
            img_dict["cropped_face_key"] = None
        person_dict["images"].append(img_dict)
    return {
        "total_success": sum(1 for r in results if r["success"]),
        "total_failed": sum(1 for r in results if not r["success"]),
        "results": results,
        "personnel": person_dict,
    }
