from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response

from app.core.auth import require_permission
from app.core.jalali_utils import to_jalali_local_string
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

router = APIRouter(prefix="/api/v1/personnel-images", tags=["personnel-images"])


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


def _image_response(img: PersonnelImageRecord, store: PersonnelStore) -> dict:
    return {
        "id": img.id,
        "image_base64": store.read_image_base64(img.storage_key),
        "personnel_id": img.personnel_id,
        "is_primary": img.is_primary,
        "uploaded_at": to_jalali_local_string(img.uploaded_at_utc),
    }


@router.get(
    "/personnel/{personnel_id}",
    summary="List all images for a personnel (legacy alias)",
)
def list_personnel_images(
    personnel_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_permission("personnel_images.read")),
) -> list:
    store = _store(runtime)
    if store.get(personnel_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,         detail="پرسنل یافت نشد")
    images = store.list_images(personnel_id)
    return [_image_response(img, store) for img in images]


@router.post(
    "/personnel/{personnel_id}",
    summary="Upload images for a personnel (legacy alias)",
    status_code=status.HTTP_201_CREATED,
)
async def upload_personnel_images(
    personnel_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_permission("personnel_images.create")),
    images: list[UploadFile] = File(...),
    enable_cropping: bool = Form(default=False),
    is_primary: bool = Form(default=False),
) -> list:
    store = _store(runtime)
    person = store.get(personnel_id)
    if person is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,         detail="پرسنل یافت نشد")

    processor = _image_processor(runtime)
    saved_images: list[PersonnelImageRecord] = []

    for idx, img_file in enumerate(images):
        raw = await img_file.read()
        if not raw:
            continue
        validation = validate_uploaded_image(raw, img_file.filename or "image.jpg", img_file.content_type)
        if not validation.valid:
            continue
        storage_key = store._save_image_file(personnel_id, raw, img_file.filename or "image.jpg", person.national_code)
        try:
            img_record = store.create_image(
                personnel_id=personnel_id,
                storage_key=storage_key,
                embedding_id=None,
                is_primary=is_primary if idx == 0 else None,
            )
        except ValueError:
            store._delete_storage_file(storage_key)
            continue

        process_result = processor.process_image(
            raw,
            person_name=person.national_code,
            ref_img_id=str(img_record.id),
            enable_cropping=enable_cropping,
        )
        if process_result.status != 1:
            store.delete_image(img_record.id)
            continue
        try:
            store.update_image_embedding(img_record.id, process_result.vector_point_id)
        except Exception:
            processor.delete_vector(process_result.vector_point_id)
            store.delete_image(img_record.id)
            continue
        saved_images.append(store.get_image(img_record.id) or img_record)

    if not saved_images:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="همه تصاویر پردازش نشدند")

    return [_image_response(img, store) for img in saved_images]


@router.delete(
    "/{image_id}",
    summary="Delete a personnel image (legacy alias)",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_personnel_image(
    image_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_permission("personnel_images.create")),
) -> Response:
    deleted = _store(runtime).delete_image(image_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="تصویر یافت نشد")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
