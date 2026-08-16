from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

# The uploaded project depends on jdatetime in production. The isolated test
# environment used for this archive does not include it, and these tests do not
# execute Jalali conversion code.
jdatetime_stub = types.ModuleType("jdatetime")
class _JalaliDate:
    def __init__(self, year=1405, month=5, day=6):
        self.year = year
        self.month = month
        self.day = day

    @classmethod
    def fromgregorian(cls, date):
        return cls(1405, date.month, date.day)

jdatetime_stub.date = _JalaliDate
sys.modules.setdefault("jdatetime", jdatetime_stub)

bcrypt_stub = types.ModuleType("bcrypt")
bcrypt_stub.hashpw = lambda value, salt: value
bcrypt_stub.gensalt = lambda: b"salt"
bcrypt_stub.checkpw = lambda plain, hashed: plain == hashed
sys.modules.setdefault("bcrypt", bcrypt_stub)

jwt_stub = types.ModuleType("jwt")
jwt_stub.ExpiredSignatureError = type("ExpiredSignatureError", (Exception,), {})
jwt_stub.InvalidTokenError = type("InvalidTokenError", (Exception,), {})
jwt_stub.encode = lambda *args, **kwargs: "token"
jwt_stub.decode = lambda *args, **kwargs: {}
sys.modules.setdefault("jwt", jwt_stub)

from app.api import detection_logs as api
from app.core.detection_log_store import DetectionLogRecord
from app.core.detection_media import DetectionMediaStorage


def _record(**overrides) -> DetectionLogRecord:
    values = {
        "id": 7,
        "source_system": "face_recognition",
        "source_event_key": "human-track:s:c:1",
        "source_human_log_id": 11,
        "personnel_id": None,
        "person": "Unknown",
        "confidence": 0.75,
        "detection_time": "2026-07-28T10:00:00+00:00",
        "ref_img_id": None,
        "room_id": None,
        "camera_id": "camera-1",
        "access_granted": False,
        "counts_for_attendance": True,
        "log_type": "camera_rtsp",
        "import_source_parts": None,
        "face_image": None,
        "face_thumbnail": None,
        "body_image": None,
        "snapshot_image": None,
        "video": None,
        "face_video_or_unknown_faces": None,
        "video_status": "missing",
        "face_video_status": "missing",
        "media_finalized_at": None,
        "created_by": None,
        "updated_by": None,
        "created_at_utc": "2026-07-28T10:00:00+00:00",
        "updated_at_utc": "2026-07-28T10:00:00+00:00",
    }
    values.update(overrides)
    return DetectionLogRecord(**values)


def test_response_embeds_only_face_thumbnail_and_generates_protected_urls(
    tmp_path: Path, monkeypatch
) -> None:
    storage = DetectionMediaStorage(tmp_path)
    image = np.full((400, 300, 3), 90, dtype=np.uint8)
    face_key, _ = storage.save_jpeg(
        image, directory="human/detected_faces", filename="face.jpg", quality=92
    )
    thumb_key, _ = storage.save_jpeg(
        image,
        directory="human/face_thumbnails",
        filename="thumb.jpg",
        quality=72,
        max_size=224,
    )
    body_key, _ = storage.save_jpeg(
        image, directory="human/body_images", filename="body.jpg", quality=92
    )
    snapshot_key, _ = storage.save_jpeg(
        image, directory="human/full_frame_images", filename="snapshot.jpg", quality=92
    )
    video_path = tmp_path / "human" / "videos" / "clip.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"not-empty-video-for-url-contract")
    video_key = storage.key_for_path(video_path)

    monkeypatch.setattr(api, "get_detection_media_storage", lambda: storage)
    monkeypatch.setattr(api, "_resolve_usernames", lambda record: (None, None))
    monkeypatch.setattr(api, "_resolve_personnel_name", lambda personnel_id: None)
    monkeypatch.setattr(
        api, "_resolve_names", lambda room_id, camera_id: (None, None, None, None)
    )

    response = api._build_response(
        _record(
            face_image=face_key,
            face_thumbnail=thumb_key,
            body_image=body_key,
            snapshot_image=snapshot_key,
            video=video_key,
            video_status="ready",
            media_finalized_at="2026-07-28T10:01:00+00:00",
        ),
        include_detail=True,
        include_face_thumbnail=True,
    )

    assert response["face_thumbnail"].startswith("data:image/jpeg;base64,")
    assert response["face_image_url"] == "/api/v1/logs/7/face"
    assert response["body_image_url"] == "/api/v1/logs/7/body"
    assert response["snapshot_image_url"] == "/api/v1/logs/7/snapshot"
    assert response["video_url"] == "/api/v1/logs/7/video"
    assert response["body_thumbnail"] is None
    assert "human/detected_faces/face.jpg" not in str(response)
    assert "human/body_images/body.jpg" not in str(response)

    without_thumbnail = api._build_response(
        _record(face_image=face_key, face_thumbnail=thumb_key),
        include_face_thumbnail=False,
    )
    assert without_thumbnail["face_thumbnail"] is None
    assert without_thumbnail["face_image_url"] == "/api/v1/logs/7/face"


def test_video_url_is_hidden_until_ready(tmp_path: Path, monkeypatch) -> None:
    storage = DetectionMediaStorage(tmp_path)
    video = tmp_path / "human" / "videos" / "writing.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"partial")
    key = storage.key_for_path(video)

    monkeypatch.setattr(api, "get_detection_media_storage", lambda: storage)
    monkeypatch.setattr(api, "_resolve_usernames", lambda record: (None, None))
    monkeypatch.setattr(api, "_resolve_personnel_name", lambda personnel_id: None)
    monkeypatch.setattr(
        api, "_resolve_names", lambda room_id, camera_id: (None, None, None, None)
    )

    response = api._build_response(
        _record(video=key, video_status="writing"), include_face_thumbnail=False
    )
    assert response["video_status"] == "writing"
    assert response["video_url"] is None


def test_media_response_supports_byte_ranges(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"0123456789")
    app = FastAPI()

    @app.get("/clip")
    def clip(request: Request):
        return api._serve_media_file(media, request, download=False)

    client = TestClient(app)
    response = client.get("/clip", headers={"Range": "bytes=2-5"})
    assert response.status_code == 206
    assert response.content == b"2345"
    assert response.headers["content-range"] == "bytes 2-5/10"
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-disposition"].startswith("inline;")


def test_cleanup_preserves_media_still_referenced_by_another_log(
    tmp_path: Path, monkeypatch
) -> None:
    storage = DetectionMediaStorage(tmp_path)
    path = tmp_path / "human" / "detected_faces" / "shared.jpg"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"shared")

    class FakeStore:
        referenced = True

        def is_media_key_referenced(self, media_key: str) -> bool:
            assert media_key == "human/detected_faces/shared.jpg"
            return self.referenced

    fake_store = FakeStore()
    monkeypatch.setattr(api, "get_detection_media_storage", lambda: storage)
    monkeypatch.setattr(api, "get_detection_log_store", lambda: fake_store)

    record = _record(face_image="human/detected_faces/shared.jpg")
    api._delete_media_files(record)
    assert path.exists()

    fake_store.referenced = False
    api._delete_media_files(record)
    assert not path.exists()

import pytest

pytestmark = pytest.mark.unit
