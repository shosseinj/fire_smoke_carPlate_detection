from __future__ import annotations

import asyncio
from io import BytesIO
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException, UploadFile
from starlette.datastructures import Headers

from app.api.personnel import upload_personnel_images
from app.api.personnel import router as personnel_router
from app.core.personnel_store import PersonnelImageRecord


class _Store:
    def __init__(self) -> None:
        self.saved_files = 0

    def get(self, personnel_id: int) -> SimpleNamespace:
        return SimpleNamespace(id=personnel_id, national_code="1234567891")

    def _save_image_file(self, personnel_id: int, raw: bytes, filename: str) -> str:
        self.saved_files += 1
        return f"personnel_snapshots/{personnel_id}/{filename}"

    def create_image(
        self,
        personnel_id: int,
        storage_key: str,
        description: str | None,
        embedding_id: str | None,
    ) -> PersonnelImageRecord:
        return PersonnelImageRecord(
            id=1,
            personnel_id=personnel_id,
            storage_key=storage_key,
            description=description,
            is_primary=True,
            uploaded_at_utc="2026-07-22T00:00:00Z",
            embedding_id=embedding_id,
        )


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

    assert files_schema["type"] == "array"
    assert files_schema["items"] == {
        "type": "string",
        "format": "binary",
        "contentMediaType": "image/*",
    }


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
                description=None,
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
            description=None,
        )
    )

    assert len(result) == 1
    assert result[0].storage_key.endswith("face.jpg")
    assert store.saved_files == 1
