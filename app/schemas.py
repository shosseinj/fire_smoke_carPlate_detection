from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.core.source_registry import RTSP, SOURCE_TYPES
from app.core.types import TaskName


class SourceCreate(BaseModel):
    source_id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=300)
    enabled: bool = True
    tasks: set[TaskName] = Field(default_factory=set)
    source_uri: str | None = None
    frame_width: int = Field(default=640, ge=16, le=4096)
    frame_height: int = Field(default=640, ge=16, le=4096)
    source_type: str = RTSP
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Per-source confidence overrides (stored in the `sources` table)
    fire_confidence: float | None = Field(default=None, ge=0, le=1)
    smoke_confidence: float | None = Field(default=None, ge=0, le=1)
    plate_confidence: float | None = Field(default=None, ge=0, le=1)
    plate_iou: float | None = Field(default=None, ge=0, le=1)
    vehicle_confidence: float | None = Field(default=None, ge=0, le=1)
    vehicle_iou: float | None = Field(default=None, ge=0, le=1)
    face_human_confidence: float | None = Field(default=None, ge=0, le=1)
    face_detection_confidence: float | None = Field(default=None, ge=0, le=1)
    face_recognition_threshold: float | None = Field(default=None, ge=0, le=1)

    @field_validator("source_id")
    @classmethod
    def clean_source_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("source_id cannot be blank")
        return value

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, value: str) -> str:
        if value not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}")
        return value


class SourceUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=300)
    enabled: bool | None = None
    tasks: set[TaskName] | None = None
    source_uri: str | None = None
    frame_width: int | None = Field(default=None, ge=16, le=4096)
    frame_height: int | None = Field(default=None, ge=16, le=4096)
    source_type: str | None = None
    metadata: dict[str, Any] | None = None
    # Per-source confidence overrides (stored in the `sources` table)
    fire_confidence: float | None = Field(default=None, ge=0, le=1)
    smoke_confidence: float | None = Field(default=None, ge=0, le=1)
    plate_confidence: float | None = Field(default=None, ge=0, le=1)
    plate_iou: float | None = Field(default=None, ge=0, le=1)
    vehicle_confidence: float | None = Field(default=None, ge=0, le=1)
    vehicle_iou: float | None = Field(default=None, ge=0, le=1)
    face_human_confidence: float | None = Field(default=None, ge=0, le=1)
    face_detection_confidence: float | None = Field(default=None, ge=0, le=1)
    face_recognition_threshold: float | None = Field(default=None, ge=0, le=1)

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, value: str | None) -> str | None:
        if value is not None and value not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}")
        return value


class TaskAssignment(BaseModel):
    source_ids: list[str] = Field(min_length=1)
    tasks: set[TaskName]
    enabled: bool | None = None


class FrameRoundResponse(BaseModel):
    round_sequence: int
    received_frames: int
    accepted_sources: int
    task_submissions: int


class SourceResponse(BaseModel):
    source_id: str
    name: str
    enabled: bool
    tasks: list[TaskName]
    source_uri: str | None
    frame_width: int
    frame_height: int
    metadata: dict[str, Any]
    created_at_utc: str
    updated_at_utc: str
    # Resolved per-source confidence thresholds (from the `sources` table)
    fire_confidence: float | None = None
    smoke_confidence: float | None = None
    plate_confidence: float | None = None
    plate_iou: float | None = None
    vehicle_confidence: float | None = None
    vehicle_iou: float | None = None
    face_human_confidence: float | None = None
    face_detection_confidence: float | None = None
    face_recognition_threshold: float | None = None


class CameraCreate(BaseModel):
    camera_id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=300)
    source_uri: str | None = None
    frame_width: int = Field(default=640, ge=16, le=4096)
    frame_height: int = Field(default=640, ge=16, le=4096)
    source_type: str = RTSP
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("camera_id")
    @classmethod
    def clean_camera_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("camera_id cannot be blank")
        return value

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, value: str) -> str:
        if value not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}")
        return value


class CameraUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=300)
    source_uri: str | None = None
    frame_width: int | None = Field(default=None, ge=16, le=4096)
    frame_height: int | None = Field(default=None, ge=16, le=4096)
    source_type: str | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, value: str | None) -> str | None:
        if value is not None and value not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}")
        return value


class CameraBulkUpdate(CameraUpdate):
    camera_id: str = Field(min_length=1, max_length=200)

    @field_validator("camera_id")
    @classmethod
    def clean_camera_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("camera_id cannot be blank")
        return value


class CameraReplace(BaseModel):
    name: str = Field(min_length=1, max_length=300)
    source_uri: str | None = None
    frame_width: int = Field(default=640, ge=16, le=4096)
    frame_height: int = Field(default=640, ge=16, le=4096)
    source_type: str = RTSP
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, value: str) -> str:
        if value not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}")
        return value


class CameraResponse(BaseModel):
    camera_id: str
    name: str
    source_uri: str | None
    frame_width: int
    frame_height: int
    source_type: str = RTSP
    metadata: dict[str, Any]
    created_at_utc: str
    updated_at_utc: str
    settings_overrides: dict[str, Any] = Field(default_factory=dict)
    # effective_settings: dict[str, Any] = Field(default_factory=dict)


    @field_validator("source_type")
    @classmethod
    def ensure_source_type(cls, value: str) -> str:
        return value if value in SOURCE_TYPES else RTSP


class CameraSettingsPatch(BaseModel):
    video_ingest_fps: float | None = Field(default=None, gt=0, le=240)
    video_preview_fps: float | None = Field(default=None, gt=0, le=240)
    video_loop: bool | None = None
    rtsp_transport: str | None = None
    rtsp_open_timeout_ms: int | None = Field(default=None, gt=0)
    rtsp_read_timeout_ms: int | None = Field(default=None, gt=0)
    rtsp_reconnect_seconds: float | None = Field(default=None, ge=0.5)
    deepstream_rtsp_latency_ms: int | None = Field(default=None, gt=0)
    deepstream_rtsp_stall_timeout_seconds: int | None = Field(default=None, gt=0)
    fire_confidence: float | None = Field(default=None, ge=0, le=1)
    smoke_confidence: float | None = Field(default=None, ge=0, le=1)
    plate_confidence: float | None = Field(default=None, ge=0, le=1)
    plate_iou: float | None = Field(default=None, ge=0, le=1)
    vehicle_confidence: float | None = Field(default=None, ge=0, le=1)
    vehicle_iou: float | None = Field(default=None, ge=0, le=1)
    face_human_confidence: float | None = Field(default=None, ge=0, le=1)
    face_detection_confidence: float | None = Field(default=None, ge=0, le=1)
    face_recognition_threshold: float | None = Field(default=None, ge=0, le=1)

