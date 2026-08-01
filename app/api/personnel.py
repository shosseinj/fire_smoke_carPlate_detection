from __future__ import annotations

import asyncio
import io
import logging
import queue
import zipfile
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
from app.core.frontend_messages import LocalizedJSONRoute
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

router = APIRouter(
    prefix="/api/v1/personnel",
    tags=["personnel"],
    route_class=LocalizedJSONRoute,
)


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
        with store._connection() as conn:
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


@router.get("/", summary="دریافت لیست همه پرسنل")
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


@router.post("/", summary="ایجاد پرسنل جدید", status_code=status.HTTP_201_CREATED)
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


@router.get("/search/{national_code}", summary="جستجوی پرسنل با کد ملی")
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


@router.post("/with-images", summary="ایجاد پرسنل با تصاویر", status_code=status.HTTP_201_CREATED)
async def create_personnel_with_images(
    runtime: Runtime = Depends(get_runtime),
    fname: str = Form(...),
    lname: str = Form(...),
    national_code: str = Form(...),
    employee_type: str = Form(default="unknown"),
    degree: str | None = Form(default=None),
    department_id: int | None = Form(default=None),
    shift_id: int | None = Form(default=None),
    images: list[UploadFile] = File(
        description="تصاویر JPEG، PNG یا BMP",
        media_type="image/*",
        json_schema_extra={
            "items": {
                "type": "string",
                "format": "binary",
                "contentMediaType": "image/*",
            }
        },
    ),
    enable_cropping: bool = Form(default=False),
) -> Any:
    store = _store(runtime)
    try:
        person = store.create(
            fname=fname,
            lname=lname,
            national_code=national_code,
            employee_type=employee_type,
            degree=degree,
            department_id=department_id,
            shift_id=shift_id,
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
            errors.append(f"{img_file.filename}: فایل خالی است")
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
        if process_result.status == 1:
            embedding_id = process_result.vector_point_id
            try:
                store.update_image_embedding(img_record.id, embedding_id)
            except Exception:
                processor.delete_vector(embedding_id)
                store.delete_image(img_record.id)
                errors.append(f"{img_file.filename}: failed to persist vector reference")
                continue
        else:
            errors.append(f"{img_file.filename}: {process_result.failure_message}")
            store.delete_image(img_record.id)
            continue
        saved_images.append(store.get_image(img_record.id) or img_record)

    if not saved_images and not errors:
        store.delete(person.id)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="همه تصاویر پردازش نشدند")

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
    summary="ورود اطلاعات پرسنل از فایل اکسل",
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
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="فایل خالی است")
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
    updated_count = sum(1 for r in result.get("successful_rows", []) if r["action"] == "updated")
    return {
        "summary": {
            "total_rows": total,
            "successful": result["created"],
            "failed": len(result["errors"]),
            "skipped": result["skipped"],
            "created": result["created"],
            "updated": updated_count,
        },
        "successful_rows": result.get("successful_rows", []),
        "failed_rows": [
            {
                "row": e["row"],
                "national_code": e.get("national_code", ""),
                "field_errors": e.get("field_errors", []),
            }
            for e in result["errors"]
        ],
        "skipped_rows": [
            {
                "row": s["row"],
                "national_code": s.get("national_code", ""),
                "field_errors": s.get("field_errors", []),
            }
            for s in result.get("skipped_rows", [])
        ],
    }


@router.get(
    "/import-template",
    summary="دانلود قالب اکسل ورود اطلاعات",
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
    status_code=status.HTTP_202_ACCEPTED,
    summary="شروع ورود پس‌زمینه‌ای فایل ZIP پرسنل با امکان نمایش پیشرفت",
)
async def start_personnel_zip_import(
    runtime: Runtime = Depends(get_runtime),
    file: UploadFile = File(...),
    enable_cropping: bool = Form(default=True),
    current_user: UserRecord = Depends(require_role("admin")),
) -> dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="فایل ZIP خالی است.")
    if not zipfile.is_zipfile(io.BytesIO(raw)):
        raise HTTPException(status_code=422, detail="فایل ارسال‌شده یک ZIP معتبر نیست.")
    try:
        record = runtime.personnel_zip_imports.submit(
            raw,
            file.filename or "file.zip",
            enable_cropping,
            current_user.id,
        )
    except queue.Full:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="صف ورود فایل‌های پرسنل پر است؛ پس از پایان یکی از پردازش‌ها دوباره تلاش کنید.",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    return {
        "job_id": record.id,
        "progress_id": record.id,
        "status": record.status,
        "status_url": f"/api/v1/import-progress/{record.id}",
        "message": "فایل دریافت شد و پردازش آن در صف قرار گرفت.",
    }


# ── Personnel CRUD by ID ─────────────────────────────────────────────
# NOTE: /{personnel_id} routes must be defined AFTER all static paths


@router.get("/{personnel_id}", summary="دریافت یک پرسنل با شناسه")
def get_personnel(
    personnel_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> SimplePersonnelResponse:
    store = _store(runtime)
    record = store.get(personnel_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="پرسنل یافت نشد")
    return _personnel_simple(record, store)


@router.put("/{personnel_id}", summary="به‌روزرسانی یک پرسنل")
def update_personnel(
    personnel_id: int,
    payload: PersonnelUpdateRequest,
    runtime: Runtime = Depends(get_runtime),
    current_user: UserRecord = Depends(require_role("admin")),
) -> SimplePersonnelResponse:
    store = _store(runtime)
    changes = payload.model_dump(exclude_unset=True, exclude_none=True)
    if not changes:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="هیچ فیلدی برای به‌روزرسانی وارد نشده است")
    changes["updated_by"] = current_user.id
    try:
        record = store.update(
            personnel_id=personnel_id,
            **changes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="پرسنل یافت نشد")
    return _personnel_simple(record, store)


@router.delete("/{personnel_id}", summary="حذف یک پرسنل", status_code=status.HTTP_204_NO_CONTENT)
def delete_personnel(
    personnel_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("superuser")),
) -> Response:
    store = _store(runtime)
    personnel = store.get(personnel_id)
    if personnel is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="پرسنل یافت نشد")
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="پرسنل یافت نشد")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ── Personnel Image endpoints ───────────────────────────────────────


@router.get(
    "/{personnel_id}/images",
    summary="دریافت همه تصاویر یک پرسنل",
)
def list_personnel_images(
    personnel_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> list:
    store = _store(runtime)
    if store.get(personnel_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="پرسنل یافت نشد")
    images = store.list_images(personnel_id)
    return [_image_to_base64_response(img, store) for img in images]


@router.post(
    "/{personnel_id}/images",
    summary="آپلود تصاویر چهره برای یک پرسنل",
    status_code=status.HTTP_201_CREATED,
)
async def upload_personnel_images(
    personnel_id: int,
    files: Annotated[
        list[UploadFile] | None,
        File(
            description="تصاویر JPEG، PNG یا BMP",
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="پرسنل یافت نشد")

    upload_files: list[UploadFile] = []
    if files is not None:
        upload_files = files if isinstance(files, list) else [files]

    if not upload_files:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="هیچ فایلی ارسال نشده است")

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
            results.append({"success": False, "failure_code": "empty_file", "failure_message": "فایل خالی است", "image": None})
            continue

        validation = validate_uploaded_image(raw, img_file.filename or "image.jpg", img_file.content_type)
        if not validation.valid:
            results.append({"success": False, "failure_code": validation.failure_code, "failure_message": validation.failure_message, "image": None})
            continue

        storage_key = store._save_image_file(personnel_id, raw, img_file.filename or "image.jpg", person.national_code)
        try:
            img_record = store.create_image(
                personnel_id=personnel_id,
                storage_key=storage_key,
                description=None,
                embedding_id=None,
                is_primary=is_primary_val,
            )
        except ValueError as exc:
            store._delete_storage_file(storage_key)
            results.append({"success": False, "failure_code": "storage_error", "failure_message": str(exc), "image": None})
            continue

        embedding_id: str | None = None
        face_status = 0
        cropped_face_key: str | None = None
        failure_code = "face_enrollment_failed"
        failure_message = "ثبت چهره و ایجاد بردار انجام نشد."
        failure_details: dict[str, Any] | None = None

        if face_processor is not None:
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                raw_face_count: int | None = None
                try:
                    raw_face_count = await run_in_threadpool(face_processor.count_faces, image)
                except Exception as exc:
                    LOGGER.exception("Face detection failed for personnel %s", personnel_id)
                    failure_code = "face_detection_error"
                    failure_message = (
                        "سرویس تشخیص چهره هنگام پردازش تصویر با خطا مواجه شد؛ "
                        "وضعیت مدل‌های تشخیص چهره را بررسی کنید و دوباره تلاش کنید."
                    )
                    failure_details = {"error_type": type(exc).__name__}

                if raw_face_count == 0:
                    face_status = 0
                    failure_code = "no_face_detected"
                    failure_message = (
                        "هیچ چهره‌ای در تصویر شناسایی نشد؛ تصویر واضح، با نور کافی "
                        "و شامل یک چهره روبه‌دوربین ارسال کنید."
                    )
                    failure_details = {"detected_faces": 0}
                elif raw_face_count is not None and raw_face_count >= 2:
                    face_status = 2
                    failure_code = "multiple_faces_detected"
                    failure_message = (
                        f"در تصویر {raw_face_count} چهره شناسایی شد؛ "
                        "تصویر باید فقط شامل یک نفر باشد."
                    )
                    failure_details = {"detected_faces": raw_face_count}
                elif raw_face_count == 1:
                    process_result = processor.process_image(raw, person_name=person.national_code, ref_img_id=str(img_record.id), enable_cropping=enable_cropping)
                    if process_result.status == 1:
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
                        failure_code = process_result.failure_code or failure_code
                        failure_message = process_result.failure_message or failure_message
                        failure_details = process_result.failure_details
            else:
                failure_code = "image_decode_failed"
                failure_message = "فایل تصویر خراب یا نامعتبر است و قابل خواندن نیست."
        else:
            process_result = processor.process_image(raw, person_name=person.national_code, ref_img_id=str(img_record.id), enable_cropping=enable_cropping)
            if process_result.status == 1:
                embedding_id = process_result.vector_point_id
            else:
                failure_code = process_result.failure_code or failure_code
                failure_message = process_result.failure_message or failure_message
                failure_details = process_result.failure_details

        if not embedding_id:
            store.delete_image(img_record.id)
            results.append({
                "success": False,
                "failure_code": failure_code,
                "failure_message": failure_message,
                "failure_details": failure_details,
                "image": None,
            })
            continue

        try:
            store.update_image_embedding(img_record.id, embedding_id)
        except Exception as exc:
            processor.delete_vector(embedding_id)
            store.delete_image(img_record.id)
            if cropped_face_key:
                store._delete_storage_file(cropped_face_key)
            results.append({
                "success": False,
                "failure_code": "vector_reference_persistence_failed",
                "failure_message": "بردار چهره ایجاد شد، اما اتصال آن به تصویر مرجع ذخیره نشد؛ عملیات بازگردانی شد.",
                "failure_details": {"error_type": type(exc).__name__},
                "image": None,
            })
            continue

        img_record = store.get_image(img_record.id) or img_record
        saved_images.append(img_record)
        all_failed = False
        img_dict = dataclass_to_dict(img_record)
        img_dict["face_status"] = face_status
        img_dict["cropped_face_key"] = cropped_face_key
        results.append({"success": True, "failure_code": None, "failure_message": None, "image": img_dict})

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
