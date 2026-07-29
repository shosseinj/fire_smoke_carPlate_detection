from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.detection_media import (
    DetectionMediaStorage,
    InvalidMediaKey,
    RestrictedMediaStaticFiles,
)


def test_canonical_storage_keys_and_containment(tmp_path: Path) -> None:
    storage = DetectionMediaStorage(tmp_path)
    inside = tmp_path / "human" / "detected_faces" / "face.jpg"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"jpeg")

    assert storage.canonical_key("human/detected_faces/face.jpg") == "human/detected_faces/face.jpg"
    assert storage.canonical_key("/media/human/detected_faces/face.jpg") == "human/detected_faces/face.jpg"
    assert storage.canonical_key(inside) == "human/detected_faces/face.jpg"
    assert storage.resolve("human/detected_faces/face.jpg", require_file=True) == inside.resolve()

    with pytest.raises(InvalidMediaKey):
        storage.canonical_key("../outside.jpg")
    with pytest.raises(InvalidMediaKey):
        storage.canonical_key(tmp_path.parent / "outside.jpg")
    with pytest.raises(InvalidMediaKey):
        storage.canonical_key("https://example.test/file.jpg")


def test_thumbnail_is_small_jpeg_and_embeddable(tmp_path: Path) -> None:
    storage = DetectionMediaStorage(tmp_path)
    image = np.zeros((900, 600, 3), dtype=np.uint8)
    image[:, :, 1] = 180

    face_key, _ = storage.save_jpeg(
        image,
        directory="human/detected_faces",
        filename="face.jpg",
        quality=92,
    )
    thumb_key, thumb_path = storage.save_jpeg(
        image,
        directory="human/face_thumbnails",
        filename="face_thumb.jpg",
        quality=72,
        max_size=224,
    )

    decoded = cv2.imread(str(thumb_path))
    assert decoded is not None
    assert max(decoded.shape[:2]) <= 224
    assert thumb_path.stat().st_size < (tmp_path / face_key).stat().st_size

    data_uri = storage.thumbnail_data_uri(thumb_key, face_key)
    assert data_uri is not None
    prefix, payload = data_uri.split(",", 1)
    assert prefix == "data:image/jpeg;base64"
    embedded = base64.b64decode(payload)
    embedded_image = cv2.imdecode(
        np.frombuffer(embedded, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert embedded_image is not None
    assert max(embedded_image.shape[:2]) <= 224


def test_create_face_thumbnail_from_original(tmp_path: Path) -> None:
    storage = DetectionMediaStorage(tmp_path)
    image = np.full((720, 480, 3), 200, dtype=np.uint8)
    face_key, _ = storage.save_jpeg(
        image, directory="human/detected_faces", filename="manual_face.jpg", quality=92
    )

    thumbnail_key = storage.create_face_thumbnail(
        face_key, filename="manual_thumbnail.jpg"
    )
    assert thumbnail_key == "human/face_thumbnails/manual_thumbnail.jpg"
    thumbnail_path = storage.resolve(thumbnail_key, require_file=True)
    assert thumbnail_path is not None
    decoded = cv2.imread(str(thumbnail_path))
    assert decoded is not None
    assert max(decoded.shape[:2]) <= 224


def test_thumbnail_falls_back_to_face_for_historical_rows(tmp_path: Path) -> None:
    storage = DetectionMediaStorage(tmp_path)
    image = np.full((480, 640, 3), 127, dtype=np.uint8)
    face_key, _ = storage.save_jpeg(
        image,
        directory="detected_faces",
        filename="legacy_face.jpg",
        quality=92,
    )

    generated = storage.thumbnail_bytes(None, face_key)
    assert generated
    assert storage.thumbnail_bytes("../invalid-thumbnail.jpg", face_key)
    decoded = cv2.imdecode(np.frombuffer(generated, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert max(decoded.shape[:2]) <= 224


def test_delete_many_deduplicates_and_stays_inside_root(tmp_path: Path) -> None:
    storage = DetectionMediaStorage(tmp_path)
    file_path = tmp_path / "human_snapshots" / "one.jpg"
    file_path.parent.mkdir(parents=True)
    file_path.write_bytes(b"content")
    outside = tmp_path.parent / "outside.txt"
    outside.write_bytes(b"keep")

    deleted = storage.delete_many(
        ["human_snapshots/one.jpg", "/media/human_snapshots/one.jpg", outside]
    )
    assert deleted == 1
    assert not file_path.exists()
    assert outside.read_bytes() == b"keep"


def test_static_compatibility_mount_blocks_private_person_media(tmp_path: Path) -> None:
    private_directories = (
        "human",
        "detected_faces",
        "body_images",
        "full_frame_images",
        "reference_images",
        "human_snapshots",
        "whole_snapshots",
        "personnel_snapshots",
    )
    for directory in private_directories:
        private = tmp_path / directory / "secret.jpg"
        private.parent.mkdir(parents=True)
        private.write_bytes(b"secret")
    public = tmp_path / "plate" / "snapshots" / "plate.jpg"
    public.parent.mkdir(parents=True)
    public.write_bytes(b"plate")

    app = FastAPI()
    app.mount("/media", RestrictedMediaStaticFiles(directory=tmp_path), name="media")
    client = TestClient(app)

    for directory in private_directories:
        assert client.get(f"/media/{directory}/secret.jpg").status_code == 404
    response = client.get("/media/plate/snapshots/plate.jpg")
    assert response.status_code == 200
    assert response.content == b"plate"


def test_video_status_requires_a_readable_finalized_file(tmp_path: Path) -> None:
    storage = DetectionMediaStorage(tmp_path)
    assert storage.finalized_video_status(None) == "missing"

    invalid = tmp_path / "human_videos" / "invalid.mp4"
    invalid.parent.mkdir(parents=True)
    invalid.write_bytes(b"not-an-mp4")
    assert storage.finalized_video_status("human_videos/invalid.mp4") == "failed"
