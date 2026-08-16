from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.source_registry import RTSP, SOURCE_TYPES
from app.core.types import TaskName


class SourceCreate(BaseModel):
    source_uri: str | None = Field(default=None, min_length=1, max_length=500)
    static_video_id: int | None = Field(default=None, ge=1)
    name: str = Field(min_length=1, max_length=300)
    enabled: bool = True
    tasks: set[TaskName] = Field(default_factory=set)
    frame_width: int = Field(default=640, ge=16, le=4096)
    frame_height: int = Field(default=640, ge=16, le=4096)
    source_type: str = RTSP
    room_id: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    fps: float | None = Field(
        default=None,
        gt=0,
        le=240,
        description=(
            "Per-source delivery FPS override stored in sources.fps. "
            "NULL uses the source-native FPS."
        ),
    )
    loop: bool = True
    draw_human: bool = True
    draw_zone: bool = True
    draw_fire: bool = True
    draw_smoke: bool = True
    draw_vehicle: bool = True
    draw_plate: bool = True
    counts_for_attendance: bool = True
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

    @field_validator("source_uri")
    @classmethod
    def clean_source_uri(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("source_uri cannot be blank")
        return value

    @model_validator(mode="after")
    def require_source_identity(self) -> "SourceCreate":
        if self.source_uri is None and self.static_video_id is None:
            raise ValueError("source_uri or static_video_id is required")
        return self

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
    room_id: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] | None = None
    fps: float | None = Field(
        default=None,
        gt=0,
        le=240,
        description=(
            "Per-source delivery FPS override. Send null to restore native FPS."
        ),
    )
    loop: bool | None = None
    draw_human: bool | None = None
    draw_zone: bool | None = None
    draw_fire: bool | None = None
    draw_smoke: bool | None = None
    draw_vehicle: bool | None = None
    draw_plate: bool | None = None
    counts_for_attendance: bool | None = None
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


class BulkSourceUpdateItem(BaseModel):
    """Single item in a bulk source update request, identified by database id."""

    id: int
    source_uri: str | None = None
    name: str | None = Field(default=None, min_length=1, max_length=300)
    enabled: bool | None = None
    tasks: set[TaskName] | None = None
    frame_width: int | None = Field(default=None, ge=16, le=4096)
    frame_height: int | None = Field(default=None, ge=16, le=4096)
    source_type: str | None = None
    room_id: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] | None = None
    fps: float | None = Field(
        default=None,
        gt=0,
        le=240,
        description=(
            "Per-source delivery FPS override. Send null to restore native FPS."
        ),
    )
    loop: bool | None = None
    draw_human: bool | None = None
    draw_zone: bool | None = None
    draw_fire: bool | None = None
    draw_smoke: bool | None = None
    draw_vehicle: bool | None = None
    draw_plate: bool | None = None
    counts_for_attendance: bool | None = None
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


class BulkSourceCreate(BaseModel):
    """Bulk source creation request: a list of sources to create in one call."""

    sources: list[SourceCreate] = Field(min_length=1, max_length=200)


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
    id: int = 0
    source_uri: str
    static_video_id: int | None = None
    name: str
    enabled: bool
    tasks: list[TaskName]
    frame_width: int
    frame_height: int
    source_type: str = RTSP
    room_id: int | None = None
    metadata: dict[str, Any]
    fps: float | None = Field(
        default=None,
        description="NULL means source-native FPS; otherwise sources.fps is used.",
    )
    loop: bool = True
    draw_human: bool = True
    draw_zone: bool = True
    draw_fire: bool = True
    draw_smoke: bool = True
    draw_vehicle: bool = True
    draw_plate: bool = True
    counts_for_attendance: bool = True
    created_at_jalali: str = ""
    updated_at_jalali: str | None = None
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
    source_uri: str = Field(min_length=1, max_length=500)
    name: str = Field(min_length=1, max_length=300)
    frame_width: int = Field(default=640, ge=16, le=4096)
    frame_height: int = Field(default=640, ge=16, le=4096)
    source_type: str = RTSP
    room_id: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    counts_for_attendance: bool = True

    @field_validator("source_uri")
    @classmethod
    def clean_source_uri(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("source_uri cannot be blank")
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
    room_id: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] | None = None
    counts_for_attendance: bool | None = None

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, value: str | None) -> str | None:
        if value is not None and value not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}")
        return value


class CameraBulkUpdate(CameraUpdate):
    source_uri: str = Field(min_length=1, max_length=500)

    @field_validator("source_uri")
    @classmethod
    def clean_source_uri(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("source_uri cannot be blank")
        return value


class CameraReplace(BaseModel):
    name: str = Field(min_length=1, max_length=300)
    source_uri: str | None = None
    frame_width: int = Field(default=640, ge=16, le=4096)
    frame_height: int = Field(default=640, ge=16, le=4096)
    source_type: str = RTSP
    room_id: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    counts_for_attendance: bool = True

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, value: str) -> str:
        if value not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}")
        return value


class CameraResponse(BaseModel):
    id: int
    source_uri: str
    name: str
    frame_width: int
    frame_height: int
    source_type: str = RTSP
    room_id: int | None = None
    metadata: dict[str, Any]
    counts_for_attendance: bool = True
    created_at_jalali: str = ""
    updated_at_jalali: str | None = None
    settings_overrides: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_type")
    @classmethod
    def ensure_source_type(cls, value: str) -> str:
        return value if value in SOURCE_TYPES else RTSP


class CameraSettingsPatch(BaseModel):
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

