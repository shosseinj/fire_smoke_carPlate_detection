from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.core.types import TaskName


class SourceCreate(BaseModel):
    source_id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=300)
    enabled: bool = True
    tasks: set[TaskName] = Field(default_factory=set)
    source_uri: str | None = None
    frame_width: int = Field(default=640, ge=16, le=4096)
    frame_height: int = Field(default=640, ge=16, le=4096)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_id")
    @classmethod
    def clean_source_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("source_id cannot be blank")
        return value


class SourceUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=300)
    enabled: bool | None = None
    tasks: set[TaskName] | None = None
    source_uri: str | None = None
    frame_width: int | None = Field(default=None, ge=16, le=4096)
    frame_height: int | None = Field(default=None, ge=16, le=4096)
    metadata: dict[str, Any] | None = None


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


class CameraCreate(BaseModel):
    camera_id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=300)
    enabled: bool = True
    tasks: set[TaskName] = Field(default_factory=set)
    source_uri: str | None = None
    frame_width: int = Field(default=640, ge=16, le=4096)
    frame_height: int = Field(default=640, ge=16, le=4096)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("camera_id")
    @classmethod
    def clean_camera_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("camera_id cannot be blank")
        return value


class CameraUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=300)
    enabled: bool | None = None
    tasks: set[TaskName] | None = None
    source_uri: str | None = None
    frame_width: int | None = Field(default=None, ge=16, le=4096)
    frame_height: int | None = Field(default=None, ge=16, le=4096)
    metadata: dict[str, Any] | None = None


class CameraReplace(BaseModel):
    name: str = Field(min_length=1, max_length=300)
    enabled: bool = True
    tasks: set[TaskName] = Field(default_factory=set)
    source_uri: str | None = None
    frame_width: int = Field(default=640, ge=16, le=4096)
    frame_height: int = Field(default=640, ge=16, le=4096)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CameraResponse(BaseModel):
    camera_id: str
    name: str
    enabled: bool
    tasks: list[TaskName]
    source_uri: str | None
    frame_width: int
    frame_height: int
    metadata: dict[str, Any]
    created_at_utc: str
    updated_at_utc: str
