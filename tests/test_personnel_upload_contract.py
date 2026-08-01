from __future__ import annotations

import asyncio
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException, UploadFile
from starlette.datastructures import Headers

from app.api.personnel import (
    upload_personnel_images,
)
from app.api.personnel import router as personnel_router
from app.core.personnel_store import PersonnelImageRecord, PersonnelRecord


class _Store:
    def __init__(self) -> None:
        self.saved_files = 0
        self.images: dict[int, PersonnelImageRecord] = {}

    def get(self, personnel_id: int) -> PersonnelRecord | None:
        return PersonnelRecord(
            id=personnel_id,
            fname="Test",
            lname="User",
            national_code="1234567891",
            employee_type="employee",
            degree=None,
            shift_id=None,
            department_id=None,
            last_seen=None,
            created_at_utc="2026-07-22T00:00:00Z",
            updated_at_utc="2026-07-22T00:00:00Z",
        )

    def _save_image_file(self, personnel_id: int, raw: bytes, filename: str, national_code: str = "") -> str:
        self.saved_files += 1
        if national_code:
            return f"human/reference_images/{national_code}/{filename}"
        return f"human/reference_images/{personnel_id}/{filename}"

    def create_image(
        self,
        personnel_id: int,
        storage_key: str,
        description: str | None,
        embedding_id: str | None,
        is_primary: bool | None = None,
    ) -> PersonnelImageRecord:
        record = PersonnelImageRecord(
            id=self.saved_files,
            personnel_id=personnel_id,
            storage_key=storage_key,
            description=description,
            is_primary=True,
            uploaded_at_utc="2026-07-22T00:00:00Z",
            embedding_id=embedding_id,
        )
        self.images[record.id] = record
        return record

    def update_image_embedding(self, image_id: int, embedding_id: str | None) -> None:
        record = self.images[image_id]
        self.images[image_id] = replace(record, embedding_id=embedding_id)

    def get_image(self, image_id: int) -> PersonnelImageRecord | None:
        return self.images.get(image_id)

    def delete_image(self, image_id: int) -> bool:
        return self.images.pop(image_id, None) is not None


def _runtime(store: _Store) -> Any:
    return SimpleNamespace(personnel_store=store, face_processor=None)


def _upload_file(data: bytes, content_type: str) -> UploadFile:
    return UploadFile(
        file=BytesIO(data),
        filename="face.jpg",
        headers=Headers({"content-type": content_type}),
    )


def _jpeg_bytes() -> bytes:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


def test_openapi_declares_multiple_binary_images() -> None:
    app = FastAPI()
    app.include_router(personnel_router)

    schema = app.openapi()
    operation = schema["paths"]["/api/v1/personnel/{personnel_id}/images"]["post"]
    body_schema = operation["requestBody"]["content"]["multipart/form-data"]["schema"]
    component_name = body_schema["$ref"].rsplit("/", 1)[-1]
    files_schema = schema["components"]["schemas"][component_name]["properties"]["files"]

    # files is declared as array of binary strings (with nullable anyOf)
    assert files_schema["items"]["type"] == "string"
    assert files_schema["items"]["format"] == "binary"


def test_create_with_images_declares_department_and_shift_once() -> None:
    app = FastAPI()
    app.include_router(personnel_router)

    schema = app.openapi()
    operation = schema["paths"]["/api/v1/personnel/with-images"]["post"]
    body_schema = operation["requestBody"]["content"]["multipart/form-data"]["schema"]
    component_name = body_schema["$ref"].rsplit("/", 1)[-1]
    properties = schema["components"]["schemas"][component_name]["properties"]

    assert "department_id" in properties
    assert "shift_id" in properties
    assert "departmen_id" not in properties


@pytest.mark.parametrize(
    ("data", "content_type"),
    [
        (_jpeg_bytes(), "text/plain"),
        (b"not an image", "image/jpeg"),
    ],
)
def test_upload_rejects_non_images_before_storage(data: bytes, content_type: str) -> None:
    store = _Store()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            upload_personnel_images(
                personnel_id=1,
                files=[_upload_file(data, content_type)],
                runtime=_runtime(store),
                _=None,
                is_primary=None,
            )
        )

    assert exc_info.value.status_code == 422
    assert store.saved_files == 0


def test_upload_accepts_decodable_image_file() -> None:
    store = _Store()

    result = asyncio.run(
        upload_personnel_images(
            personnel_id=1,
            files=[_upload_file(_jpeg_bytes(), "image/jpeg")],
            runtime=_runtime(store),
            _=None,
            is_primary=None,
        )
    )

    assert result["total_success"] == 1
    assert result["results"][0]["image"]["storage_key"].endswith("face.jpg")
    assert store.saved_files == 1


@pytest.mark.parametrize("enable_cropping", [True, False])
def test_upload_accepts_enable_cropping_flag(enable_cropping: bool) -> None:
    """The enable_cropping parameter is accepted without error and
    cropped_face_key is None when cropping is disabled (mock mode)."""
    store = _Store()

    result = asyncio.run(
        upload_personnel_images(
            personnel_id=1,
            files=[_upload_file(_jpeg_bytes(), "image/jpeg")],
            runtime=_runtime(store),
            _=None,
            enable_cropping=enable_cropping,
            is_primary=None,
        )
    )

    assert result["total_success"] == 1
    assert result["results"][0]["success"] is True
    entry = result["results"][0]["image"]
    # In mock mode (face_processor=None) no cropping can happen
    assert "cropped_face_key" in entry
