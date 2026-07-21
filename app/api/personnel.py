from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.core.auth import require_role
from app.core.personnel_image_service import (
    ImageProcessResult,
    PersonnelImageProcessor,
    validate_uploaded_image,
)
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


class PersonnelImageUploadResult(BaseModel):
    """Result of processing one uploaded image."""

    filename: str
    success: bool
    image: PersonnelImageResponse | None = None
    failure_code: str | None = None
    failure_reason: str | None = None


class PersonnelBatchUploadResponse(BaseModel):
    """Response for the batch image upload endpoint."""

    personnel: PersonnelWithImagesResponse
    results: list[PersonnelImageUploadResult]
    total_success: int
    total_failed: int


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


def _personnel_with_images_response(
    p: PersonnelRecord,
    images: list[PersonnelImageRecord],
) -> PersonnelWithImagesResponse:
    base_url = getattr(settings, "external_base_url", "")
    enriched_images: list[dict] = []
    for img in images:
        img_dict = {
            "id": img.id,
            "personnel_id": img.personnel_id,
            "storage_key": img.storage_key,
            "description": img.description,
            "is_primary": img.is_primary,
            "uploaded_at_utc": img.uploaded_at_utc,
            "embedding_id": img.embedding_id,
            "url": f"{base_url}/media/{img.storage_key}",
        }
        enriched_images.append(img_dict)
    return PersonnelWithImagesResponse(
        id=p.id,
        fname=p.fname,
        lname=p.lname,
        national_code=p.national_code,
        employee_type=p.employee_type,
        degree=p.degree,
        last_seen=p.last_seen,
        created_at_utc=p.created_at_utc,
        updated_at_utc=p.updated_at_utc,
        images=enriched_images,
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
        result = await run_in_threadpool(_store(runtime).upload_personnel_zip, raw)
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


@router.delete(
    "/{personnel_id}",
    summary="Delete a personnel record",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_personnel(
    personnel_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> Response:
    store = _store(runtime)

    # 1. Collect image info for vector cleanup before deleting
    image_info_list = store.get_all_image_info(personnel_id)
    embedding_ids: list[str] = []
    for info in image_info_list:
        eid = info.get("embedding_id")
        if eid is not None:
            embedding_ids.append(str(eid))

    # 2. Preserve detection records by setting personnel_id to NULL
    store.set_null_personnel_id_for_detections(personnel_id)

    # 3. Delete personnel record (cascade deletes images and storage files)
    deleted = store.delete(personnel_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Personnel not found")

    # 4. Clean up vector store entries
    if embedding_ids:
        face_processor = _face_processor(runtime)
        if face_processor is not None:
            for point_id in embedding_ids:
                try:
                    face_processor._vector_store.delete_point(point_id)
                except Exception as exc:
                    LOGGER.warning("Vector cleanup failed for point %s: %s", point_id, exc)

    return Response(status_code=status.HTTP_204_NO_CONTENT)


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
    summary="Upload one or more face images for a personnel record",
    status_code=status.HTTP_201_CREATED,
)
async def upload_personnel_image(
    personnel_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
    files: list[UploadFile] = File(...),
    enable_cropping: bool = Form(default=False),
    is_primary: bool | None = Form(default=None),
) -> PersonnelBatchUploadResponse:
    store = _store(runtime)

    # 1. Verify personnel exists
    person = store.get(personnel_id)
    if person is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Personnel not found")

    # 2. Validate upload count
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one image file is required",
        )
    if len(files) > settings.max_images_per_request:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Maximum {settings.max_images_per_request} images per request",
        )

    face_processor = _face_processor(runtime)
    image_processor = PersonnelImageProcessor(face_processor)

    # 3. Track cleanup data using scalar values (not ORM objects)
    cleanup_data: list[dict] = []  # each: {storage_key, vector_point_id, image_id, filename}
    successful_records: list[PersonnelImageRecord] = []
    upload_results: list[PersonnelImageUploadResult] = []
    has_unrecoverable_error = False

    # 4. Process each file
    for file in files:
        raw = await file.read()
        safe_name = file.filename or "image"

        # 4a. Validate the uploaded image
        validation = validate_uploaded_image(raw, safe_name, file.content_type)
        if not validation.valid:
            upload_results.append(PersonnelImageUploadResult(
                filename=safe_name,
                success=False,
                failure_code=validation.failure_code,
                failure_reason=validation.failure_message,
            ))
            continue

        # 4b. Save to storage (get storage_key before DB record)
        try:
            storage_key = store._save_image_file(personnel_id, validation.data, safe_name)
        except OSError as exc:
            LOGGER.error("Storage write failed for %s: %s", safe_name, exc)
            upload_results.append(PersonnelImageUploadResult(
                filename=safe_name,
                success=False,
                failure_code="storage_error",
                failure_reason="Failed to save image file",
            ))
            continue
        cleanup_entry: dict = {"storage_key": storage_key, "vector_point_id": None, "image_id": None, "filename": safe_name}

        # 4c. Process face (detection, cropping, embedding, vector enroll)
        process_result: ImageProcessResult | None = None
        try:
            from starlette.concurrency import run_in_threadpool
            process_result = await run_in_threadpool(
                image_processor.process_image,
                validation.data,
                person_name=f"{person.fname} {person.lname}",
                ref_img_id=f"personnel_{personnel_id}",
                enable_cropping=enable_cropping,
            )
        except Exception as exc:
            LOGGER.error("Face processing error for %s: %s", safe_name, exc)
            process_result = ImageProcessResult(
                success=False,
                failure_code="processing_error",
                failure_message=f"Unexpected error: {type(exc).__name__}",
            )

        if process_result is None or not process_result.success:
            # Clean up storage file, skip vector cleanup (no vector was written)
            cleanup_entry["storage_key"] = storage_key  # already set
            store._delete_storage_file(storage_key)
            upload_results.append(PersonnelImageUploadResult(
                filename=safe_name,
                success=False,
                failure_code=process_result.failure_code if process_result else "unknown",
                failure_reason=process_result.failure_message if process_result else "Unknown processing error",
            ))
            continue

        # 4d. Determine which bytes to store (original vs cropped)
        if enable_cropping and process_result.processed_bytes is not None:
            stored_bytes = process_result.processed_bytes
        else:
            stored_bytes = validation.data

        # Re-save with processed bytes if cropping changed them
        if stored_bytes is not validation.data:
            # Delete original, save processed
            store._delete_storage_file(storage_key)
            try:
                storage_key = store._save_image_file(personnel_id, stored_bytes, safe_name)
                cleanup_entry["storage_key"] = storage_key
            except OSError as exc:
                LOGGER.error("Storage write for cropped image failed: %s", exc)
                store._delete_storage_file(storage_key)
                upload_results.append(PersonnelImageUploadResult(
                    filename=safe_name,
                    success=False,
                    failure_code="storage_error",
                    failure_reason="Failed to save cropped image",
                ))
                continue

        vector_point_id = process_result.vector_point_id
        cleanup_entry["vector_point_id"] = vector_point_id

        # 4e. Create the image record (without commit — we batch below)
        try:
            img_record = store.create_image(
                personnel_id=personnel_id,
                storage_key=storage_key,
                description=None,
                embedding_id=vector_point_id,
                is_primary=is_primary,
            )
        except ValueError as exc:
            LOGGER.warning("DB create_image failed for %s: %s", safe_name, exc)
            # Cleanup: storage + vector
            store._delete_storage_file(storage_key)
            if vector_point_id is not None and face_processor is not None:
                try:
                    face_processor._vector_store.delete_point(vector_point_id)
                except Exception as exc2:
                    LOGGER.warning("Vector cleanup failed for point %s: %s", vector_point_id, exc2)
            upload_results.append(PersonnelImageUploadResult(
                filename=safe_name,
                success=False,
                failure_code="db_error",
                failure_reason="Failed to create image record",
            ))
            continue

        cleanup_entry["image_id"] = img_record.id
        cleanup_data.append(cleanup_entry)
        successful_records.append(img_record)
        upload_results.append(PersonnelImageUploadResult(
            filename=safe_name,
            success=True,
            image=_image_to_response(img_record),
        ))

    # 5. Handle all-failed case
    if not successful_records:
        # Clean up any remaining cleanup data (should be empty here because we
        # cleaned up per-image failures already)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "All uploaded files failed processing",
                "results": [
                    {"filename": r.filename, "failure_code": r.failure_code, "failure_reason": r.failure_reason}
                    for r in upload_results
                ],
            },
        )

    # 6. Reload personnel with all images
    all_images = store.list_images(personnel_id)
    person_reloaded = store.get(personnel_id)
    if person_reloaded is None:
        # Should not happen, but handle defensively
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Personnel record disappeared")

    personnel_resp = _personnel_with_images_response(person_reloaded, all_images)
    total_success = sum(1 for r in upload_results if r.success)
    total_failed = len(upload_results) - total_success

    return PersonnelBatchUploadResponse(
        personnel=personnel_resp,
        results=upload_results,
        total_success=total_success,
        total_failed=total_failed,
    )


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
    cleanup_info = store.delete_image(image_id)
    if cleanup_info is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    # Clean up vector store if an embedding exists
    embedding_id = cleanup_info.get("embedding_id")
    if embedding_id is not None:
        face_processor = _face_processor(runtime)
        if face_processor is not None:
            try:
                face_processor._vector_store.delete_point(embedding_id)
            except Exception as exc:
                LOGGER.warning("Vector cleanup failed for point %s: %s", embedding_id, exc)
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
