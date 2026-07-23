from __future__ import annotations

import pytest
from pydantic import BaseModel, Field, ValidationError


class _CameraCreate(BaseModel):
    camera_url: str
    margin_level: float | None = Field(default=None, ge=0.0, le=4.0)
    face_rec_score: float | None = Field(default=None, ge=0.0, le=1.0)


def test_camera_schema_accepts_valid_values() -> None:
    camera = _CameraCreate(camera_url="rtsp://example.local/stream", margin_level=2.0, face_rec_score=0.5)
    assert camera.margin_level == 2.0
    assert camera.face_rec_score == 0.5


def test_camera_schema_rejects_invalid_margin_and_scores() -> None:
    with pytest.raises(ValidationError, match="margin_level"):
        _CameraCreate(camera_url="rtsp://example.local/stream", margin_level=5.1)

    with pytest.raises(ValidationError, match="face_rec_score"):
        _CameraCreate(camera_url="rtsp://example.local/stream", face_rec_score=1.1)