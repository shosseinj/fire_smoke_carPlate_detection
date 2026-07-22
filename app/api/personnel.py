from __future__ import annotations

import asyncio
import io
import logging
import re
from pathlib import Path
from typing import Any
from typing import Annotated

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
    dataclass_to_dict,
)
from app.core.personnel_image_service import (
    PersonnelImageProcessor,
    validate_uploaded_image,
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


def _image_processor(runtime: Runtime) -> PersonnelImageProcessor:
    return PersonnelImageProcessor(_face_processor(runtime))


# ── Shared helpers ───────────────────────────────────────────────

def _contract_mode(
    skip: int | None = None,
    offset: int | None = None,
    search: str | None = None,
    employee_type: str | None = None,
    contract: str | None = None,
) -> str:
    """Determine response contract mode based on query parameters.

    Returns 'legacy' for old bare-array format or 'current' for wrapper format.
    """
    if contract == "current":
        return "current"
    if contract == "legacy":
        return "legacy"
    if skip is not None:
        return "legacy"
    if offset is not None or search is not None or employee_type is not None:
        return "current"
    return "legacy"


# ── Pydantic models ─────────────────────────────────────────────────

class PersonnelCreateRequest(BaseModel):
    fname: str = Field(..., min_length=1, max_length=200)
    lname: str = Field(..., min_length=1, max_length=200)
    national_code: str = Field(..., min_length=1, max_length=20)
    employee_type: str = Field(default="unknown")
    degree: str | None = Field(default=None, max_length=200)
    department_id: int | None = Field(default=None)
    shift_id: int | None = Field(default=None)


class PersonnelUpdateRequest(BaseModel):
    fname: str | None = Field(default=None, min_length=1, max_length=200)
    lname: str | None = Field(default=None, min_length=1, max_length=200)
    national_code: str | None = Field(default=None, min_length=1, max_length=20)
    employee_type: str | None = Field(default=None)
    degree: str | None = Field(default=None, max_length=200)
    department_id: int | None = Field(default=None)
    shift_id: int | None = Field(default=None)


class PersonnelResponse(BaseModel):
    id: int
    fname: str
    lname: str
    national_code: str
    employee_type: str
    degree: str | None = None
    department_name: str | None = None
    shift_name: str | None = None
    last_seen: str | None = None
    created_at_utc: str
    updated_at_utc: str


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


class LegacyPersonnelResponse(BaseModel):
    id: int
    fname: str
    lname: str
    national_code: str
    employee_type: str | None
    department_name: str | None = None
    shift_name: str | None = None
    degree: str | None = None
    created_at: str


class LegacyPersonnelImageResponse(BaseModel):
    id: int
    image_base64: str | None = None
    personnel_id: int
    is_primary: bool
    uploaded_at: str | None = None


class LegacyPersonnelWithImagesResponse(BaseModel):
    id: int
    fname: str
    lname: str
    national_code: str
    employee_type: str | None
    department_name: str | None = None
    shift_name: str | None = None
    degree: str | None = None
    created_at: str
    rooms: list = Field(default_factory=list)
    images: list[LegacyPersonnelImageResponse] = Field(default_factory=list)
    primary_image: LegacyPersonnelImageResponse | None = None


class BatchImageResult(BaseModel):
    success: bool
    failure_code: str | None = None
    failure_message: str | None = None
    image: dict | None = None


class BatchUploadResponse(BaseModel):
    total_success: int
    total_failed: int
    results: list[BatchImageResult]
    personnel: dict


# ── Converters ───────────────────────────────────────────────────

def _personnel_to_response(p: PersonnelRecord, store: PersonnelStore | None = None) -> PersonnelResponse:
    dept_name: str | None = None
    shift_name: str | None = None
    if store is not None:
        dept_name = store._resolve_department_name(p.department_id)
        shift_name = store._resolve_shift_name(p.shift_id)
    return PersonnelResponse(
        id=p.id,
        fname=p.fname,
        lname=p.lname,
        national_code=p.national_code,
        employee_type=p.employee_type,
        degree=p.degree,
        department_name=dept_name,
        shift_name=shift_name,
        last_seen=p.last_seen,
        created_at_utc=p.created_at_utc,
        updated_at_utc=p.updated_at_utc,
    )


def _legacy_personnel_response(p: PersonnelRecord, store: PersonnelStore | None = None) -> LegacyPersonnelResponse:
    dept_name: str | None = None
    shift_name: str | None = None
    if store is not None:
        dept_name = store._resolve_department_name(p.department_id)
        shift_name = store._resolve_shift_name(p.shift_id)
    return LegacyPersonnelResponse(
        id=p.id,
        fname=p.fname,
        lname=p.lname,
        national_code=p.national_code,
        employee_type=p.employee_type,
        department_name=dept_name,
        shift_name=shift_name,
        degree=p.degree,
        created_at=p.created_at_utc,
    )


def _image_to_response(img: PersonnelImageRecord, store: PersonnelStore | None = None) -> PersonnelImageResponse:
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


def _legacy_image_response(img: PersonnelImageRecord, store: PersonnelStore | None = None) -> LegacyPersonnelImageResponse:
    base64_str: str | None = None
    if store is not None:
        base64_str = store.read_image_base64(img.storage_key)
    return LegacyPersonnelImageResponse(
        id=img.id,
        image_base64=base64_str,
        personnel_id=img.personnel_id,
        is_primary=img.is_primary,
        uploaded_at=img.uploaded_at_utc,
    )


def _enrich_url(item: dict) -> dict:
    from app.config import settings
    base_url = getattr(settings, "external_base_url", "")
    for img in item.get("images", []):
        if "storage_key" in img:
            img["url"] = f"{base_url}/media/{img['storage_key']}"
    return item


# ── Personnel CRUD ──────────────────────────────────────────────────


@router.get("/", summary="List all personnel records")
def list_personnel(
    skip: int | None = Query(default=None, ge=0, description="Legacy offset (skip)"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int | None = Query(default=None, ge=0, description="Current offset"),
    employee_type: str | None = Query(default=None),
    search: str | None = Query(default=None, description="Search by fname, lname, or national_code"),
    contract: str | None = Query(default=None, description="'current' for wrapper, 'legacy' for bare array"),
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> Any:
    store = _store(runtime)
    mode = _contract_mode(skip=skip, offset=offset, search=search, employee_type=employee_type, contract=contract)
    actual_offset = skip if skip is not None else (offset if offset is not None else 0)
    records, total = store.list(
        offset=actual_offset,
        limit=limit,
        employee_type=employee_type,
        search=search,
    )
    if mode == "legacy":
        return [_legacy_personnel_response(r, store) for r in records]
    return {
        "items": [_personnel_to_response(r, store) for r in records],
        "count": len(records),
        "total": total,
    }


@router.post("/", summary="Create a new personnel record", status_code=status.HTTP_201_CREATED)
def create_personnel(
    payload: PersonnelCreateRequest,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> Any:
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
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return _legacy_personnel_response(record, store)


@router.get("/search/{national_code}", summary="Search personnel by national code")
def search_personnel(
    national_code: str,
    contract: str | None = Query(default=None, description="'current' for 404 on not found"),
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> Any:
    store = _store(runtime)
    record = store.get_by_national_code(national_code)
    if record is None:
        if contract == "current":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Personnel not found")
        return None
    if contract == "current":
        return _personnel_to_response(record, store)
    return _legacy_personnel_response(record, store)


@router.get("/with-images", summary="List personnel records with their images")
def list_personnel_with_images(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> dict:
    store = _store(runtime)
    items, total = store.list_with_images(offset=offset, limit=limit)
    enriched = [_enrich_url(item) for item in items]
    return {"items": enriched, "count": len(enriched), "total": total}


# ── Import / Export / Bulk ───────────────────────────────────────────
# NOTE: these routes must be defined BEFORE /{personnel_id} to avoid
# FastAPI matching static names like "import-template" as personnel_id


@router.post("/with-images", summary="Create personnel with images (legacy)", status_code=status.HTTP_201_CREATED)
async def create_personnel_with_images(
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
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
        storage_key = store._save_image_file(person.id, raw, img_file.filename or "image.jpg")
        embedding_id: str | None = None
        process_result = processor.process_image(
            raw,
            person_name=person_name,
            ref_img_id=f"personnel_{person.id}",
            enable_cropping=enable_cropping,
        )
        if process_result.success:
            embedding_id = process_result.vector_point_id
        else:
            errors.append(f"{img_file.filename}: {process_result.failure_message}")
        try:
            img_record = store.create_image(
                personnel_id=person.id,
                storage_key=storage_key,
                embedding_id=embedding_id,
                is_primary=len(saved_images) == 0,
            )
            saved_images.append(img_record)
        except ValueError as exc:
            store._delete_storage_file(storage_key)
            errors.append(f"{img_file.filename}: {exc}")

    if not saved_images and not errors:
        store.delete(person.id)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="All images failed processing")

    legacy_imgs = [_legacy_image_response(img, store) for img in saved_images]
    primary_img = next((img for img in saved_images if img.is_primary), None)
    legacy_person = _legacy_personnel_response(person, store)
    return {
        **legacy_person.model_dump(),
        "rooms": [],
        "images": [img.model_dump() for img in legacy_imgs],
        "primary_image": _legacy_image_response(primary_img, store).model_dump() if primary_img else None,
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
    contract: str | None = Query(default=None),
) -> Any:
    store = _store(runtime)
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Empty file")
    try:
        result = await run_in_threadpool(store.import_from_excel, raw)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    if contract == "current":
        return result
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
    _: UserRecord = Depends(require_role("admin")),
    file: UploadFile = File(...),
    skip_invalid_national_codes: bool = Form(default=True),
) -> Any:
    store = _store(runtime)
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Empty file")
    try:
        fp = _face_processor(runtime)
        result = await run_in_threadpool(store.upload_personnel_zip, raw, fp)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return {
        "success": len(result.get("errors", [])) == 0,
        "filename": file.filename or "file.zip",
        "message": f"Processed {result.get('created_personnel', 0)} persons, {result.get('created_images', 0)} images",
        "summary": {
            "total_processed": result.get("created_personnel", 0) + result.get("created_images", 0),
            "total_errors": len(result.get("errors", [])),
            "total_persons": result.get("created_personnel", 0),
            "total_images_saved": result.get("created_images", 0),
            "skipped_folders": 0,
        },
        "details": [],
        "skipped_folders": None,
    }


# ── Personnel CRUD by ID ─────────────────────────────────────────────
# NOTE: /{personnel_id} routes must be defined AFTER all static paths


@router.get("/{personnel_id}", summary="Get a personnel record by ID")
def get_personnel(
    personnel_id: int,
    contract: str | None = Query(default=None),
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> Any:
    store = _store(runtime)
    record = store.get(personnel_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Personnel not found")
    if contract == "current":
        return _personnel_to_response(record, store)
    return _legacy_personnel_response(record, store)


@router.put("/{personnel_id}", summary="Update a personnel record")
def update_personnel(
    personnel_id: int,
    payload: PersonnelUpdateRequest,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> Any:
    store = _store(runtime)
    changes = payload.model_dump(exclude_unset=True, exclude_none=True)
    if not changes:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No fields to update")
    try:
        record = store.update(
            personnel_id=personnel_id,
            **changes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Personnel not found")
    return _legacy_personnel_response(record, store)


@router.delete("/{personnel_id}", summary="Delete a personnel record", status_code=status.HTTP_204_NO_CONTENT)
def delete_personnel(
    personnel_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
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
    contract: str | None = Query(default=None),
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> Any:
    store = _store(runtime)
    if store.get(personnel_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Personnel not found")
    images = store.list_images(personnel_id)
    if contract == "legacy" or contract is None:
        return [_legacy_image_response(img, store) for img in images]
    return {
        "items": [_image_to_response(img, store) for img in images],
        "count": len(images),
    }


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
) -> Any:
    store = _store(runtime)
    person = store.get(personnel_id)
    if person is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Personnel not found")

    upload_files: list[UploadFile] = []
    if files is not None:
        upload_files = files if isinstance(files, list) else [files]

    if not upload_files:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No files provided")

    from app.config import settings as app_settings
    max_files = getattr(app_settings, "max_images_per_request", 10)
    if len(upload_files) > max_files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Maximum {max_files} files per request",
        )

    face_processor = _face_processor(runtime)
    processor = _image_processor(runtime)
    person_name = f"{person.fname} {person.lname}"
    results: list[dict] = []
    saved_images: list[PersonnelImageRecord] = []
    all_failed = True
    is_primary_val: bool | None = None
    if is_primary is not None:
        is_primary_val = is_primary.strip().lower() in ("true", "1", "yes")

    for img_file in upload_files:
        raw = await img_file.read()
        if not raw:
            results.append({
                "success": False,
                "failure_code": "empty_file",
                "failure_message": "File is empty",
                "image": None,
            })
            continue

        validation = validate_uploaded_image(raw, img_file.filename or "image.jpg", img_file.content_type)
        if not validation.valid:
            results.append({
                "success": False,
                "failure_code": validation.failure_code,
                "failure_message": validation.failure_message,
                "image": None,
            })
            continue

        storage_key = store._save_image_file(personnel_id, raw, img_file.filename or "image.jpg")
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
                    process_result = processor.process_image(
                        raw,
                        person_name=person_name,
                        ref_img_id=f"personnel_{personnel_id}",
                        enable_cropping=enable_cropping,
                    )
                    if process_result.success:
                        embedding_id = process_result.vector_point_id
                        face_status = 1
                        if enable_cropping:
                            try:
                                success, aligned = await run_in_threadpool(
                                    face_processor.get_aligned_face, image
                                )
                                if success and aligned is not None:
                                    success_enc, encoded = cv2.imencode(".jpg", aligned)
                                    if success_enc:
                                        cropped_face_key = store._save_cropped_face_file(
                                            personnel_id, encoded.tobytes(), img_file.filename or "face.jpg"
                                        )
                            except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
                                LOGGER.warning(
                                    "Cropped face save failed for personnel %s: %s", personnel_id, exc
                                )
                    else:
                        face_status = 0
            else:
                LOGGER.warning("Could not decode uploaded image for face enrollment")
        else:
            process_result = processor.process_image(
                raw,
                person_name=person_name,
                ref_img_id=f"personnel_{personnel_id}",
                enable_cropping=enable_cropping,
            )
            if process_result.success:
                embedding_id = process_result.vector_point_id

        try:
            img_record = store.create_image(
                personnel_id=personnel_id,
                storage_key=storage_key,
                description=None,
                embedding_id=embedding_id,
                is_primary=is_primary_val,
            )
            saved_images.append(img_record)
            all_failed = False
            img_dict = dataclass_to_dict(img_record)
            img_dict["face_status"] = face_status
            img_dict["cropped_face_key"] = cropped_face_key
            results.append({
                "success": True,
                "failure_code": None,
                "failure_message": None,
                "image": img_dict,
            })
        except ValueError as exc:
            store._delete_storage_file(storage_key)
            results.append({
                "success": False,
                "failure_code": "storage_error",
                "failure_message": str(exc),
                "image": None,
            })

    if all_failed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "total_success": 0,
                "total_failed": len(upload_files),
                "results": results,
                "personnel": dataclass_to_dict(person),
            },
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
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_personnel_image(
    image_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> Response:
    store = _store(runtime)
    img = store.get_image(image_id)
    if img is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    if img.embedding_id:
        processor = _face_processor(runtime)
        if processor is not None:
            processor.delete_points([img.embedding_id])
    deleted = store.delete_image(image_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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
