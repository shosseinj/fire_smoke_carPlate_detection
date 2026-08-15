from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
Box = tuple[float, float, float, float]


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1] = 1

    @staticmethod
    def _utc(value: datetime) -> bool:
        return value.tzinfo is not None and value.utcoffset() == timezone.utc.utcoffset(None)

    @staticmethod
    def _box(box: Box, width: int, height: int) -> bool:
        x1, y1, x2, y2 = box
        return all(value == value and abs(value) != float("inf") for value in box) and 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height

    @staticmethod
    def _text(value: str) -> str:
        if not value.strip():
            raise ValueError("required strings must not be blank")
        return value


class HumanDetectionEvent(_Event):
    event_id: str = Field(min_length=1)
    event_name: Literal["human"] = "human"
    event_type: Literal["track_ended"] = "track_ended"
    camera_id: str = Field(min_length=1)
    room_id: int | None = Field(default=None, gt=0)
    tracking_session_id: str = Field(min_length=1)
    track_id: int = Field(ge=0)
    personnel_id: int | None = Field(default=None, gt=0)
    ref_img_id: str | None = None
    name: str = Field(min_length=1)
    recognition_status: Literal["recognized", "unknown"]
    recognition_confidence: UnitFloat
    first_seen_at_utc: datetime
    last_seen_at_utc: datetime
    best_frame_at_utc: datetime
    best_frame_index: int = Field(ge=0)
    bounding_box: Box
    bounding_box_format: Literal["xyxy"] = "xyxy"
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    snapshot_quality: UnitFloat
    clip_start_at_utc: datetime
    clip_end_at_utc: datetime
    counts_for_attendance: bool
    created_at_utc: datetime

    @model_validator(mode="after")
    def _valid(self):
        for value in (self.event_id, self.camera_id, self.tracking_session_id, self.name): self._text(value)
        if self.ref_img_id is not None: self._text(self.ref_img_id)
        times = (self.first_seen_at_utc, self.last_seen_at_utc, self.best_frame_at_utc,
                 self.clip_start_at_utc, self.clip_end_at_utc, self.created_at_utc)
        if not all(self._utc(value) for value in times): raise ValueError("timestamps must be UTC-aware")
        if not (self.first_seen_at_utc <= self.best_frame_at_utc <= self.last_seen_at_utc): raise ValueError("observation timestamps are unordered")
        if not (self.clip_start_at_utc <= self.first_seen_at_utc <= self.last_seen_at_utc <= self.clip_end_at_utc) or self.created_at_utc < self.last_seen_at_utc: raise ValueError("clip timestamps are unordered")
        recognized = self.recognition_status == "recognized"
        if recognized != (self.personnel_id is not None and self.ref_img_id is not None and self.name != "Unknown"):
            raise ValueError("recognition identity is inconsistent")
        if not recognized and self.name != "Unknown": raise ValueError("unknown person name must be Unknown")
        if not self._box(self.bounding_box, self.frame_width, self.frame_height): raise ValueError("bounding box is outside frame")
        return self


class FireSmokeDetectionEvent(_Event):
    event_id: str = Field(min_length=1); event_name: Literal["fire_smoke"] = "fire_smoke"; event_type: Literal["incident_ended"] = "incident_ended"
    incident_id: str = Field(min_length=1); camera_id: str = Field(min_length=1); room_id: int | None = Field(default=None, gt=0)
    hazard_type: Literal["fire", "smoke", "fire_smoke"]; severity: Literal["low", "medium", "high"]
    fire_count: int = Field(ge=0); smoke_count: int = Field(ge=0); confidence: UnitFloat
    first_seen_at_utc: datetime; last_seen_at_utc: datetime; best_frame_at_utc: datetime
    bounding_boxes: tuple[Box, ...] = Field(min_length=1); bounding_box_format: Literal["xyxy"] = "xyxy"
    frame_width: int = Field(gt=0); frame_height: int = Field(gt=0)
    clip_start_at_utc: datetime; clip_end_at_utc: datetime; created_at_utc: datetime

    @model_validator(mode="after")
    def _valid(self):
        for value in (self.event_id, self.incident_id, self.camera_id): self._text(value)
        times = (self.first_seen_at_utc, self.last_seen_at_utc, self.best_frame_at_utc, self.clip_start_at_utc, self.clip_end_at_utc, self.created_at_utc)
        if not all(self._utc(v) for v in times) or not (self.clip_start_at_utc <= self.first_seen_at_utc <= self.best_frame_at_utc <= self.last_seen_at_utc <= self.clip_end_at_utc) or self.created_at_utc < self.last_seen_at_utc: raise ValueError("timestamps are invalid")
        if not all(self._box(b, self.frame_width, self.frame_height) for b in self.bounding_boxes): raise ValueError("bounding box is outside frame")
        return self


class PlateDetectionEvent(_Event):
    event_id: str = Field(min_length=1); event_name: Literal["plate"] = "plate"; event_type: Literal["plate_detected"] = "plate_detected"
    camera_id: str = Field(min_length=1); room_id: int | None = Field(default=None, gt=0); plate_id: int | None = Field(default=None, gt=0)
    plate_number: str = Field(min_length=1); raw_plate_text: str = Field(min_length=1); recognition_confidence: UnitFloat
    detected_at_utc: datetime; best_frame_at_utc: datetime; frame_index: int = Field(ge=0); bounding_box: Box
    bounding_box_format: Literal["xyxy"] = "xyxy"; frame_width: int = Field(gt=0); frame_height: int = Field(gt=0)
    clip_start_at_utc: datetime; clip_end_at_utc: datetime; created_at_utc: datetime

    @model_validator(mode="after")
    def _valid(self):
        for value in (self.event_id, self.camera_id, self.plate_number, self.raw_plate_text): self._text(value)
        times = (self.detected_at_utc, self.best_frame_at_utc, self.clip_start_at_utc, self.clip_end_at_utc, self.created_at_utc)
        if not all(self._utc(v) for v in times) or not (self.clip_start_at_utc <= self.detected_at_utc <= self.best_frame_at_utc <= self.clip_end_at_utc) or self.created_at_utc < self.detected_at_utc: raise ValueError("timestamps are invalid")
        if not self._box(self.bounding_box, self.frame_width, self.frame_height): raise ValueError("bounding box is outside frame")
        return self


class RecordingSegmentEvent(_Event):
    segment_id: str = Field(min_length=1); camera_id: str = Field(min_length=1)
    bucket: str = Field(min_length=1); object_key: str = Field(min_length=1)
    started_at_utc: datetime; ended_at_utc: datetime; frame_width: int = Field(gt=0); frame_height: int = Field(gt=0)
    fps: float = Field(gt=0, allow_inf_nan=False); status: Literal["ready"] = "ready"

    @model_validator(mode="after")
    def _valid(self):
        for value in (self.segment_id, self.camera_id, self.bucket, self.object_key): self._text(value)
        if not self._utc(self.started_at_utc) or not self._utc(self.ended_at_utc) or self.ended_at_utc <= self.started_at_utc: raise ValueError("segment timestamps are invalid")
        return self


DetectionEvent = Union[HumanDetectionEvent, FireSmokeDetectionEvent, PlateDetectionEvent, RecordingSegmentEvent]
