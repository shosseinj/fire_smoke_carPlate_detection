from __future__ import annotations

import re
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import date, datetime, time, timezone
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    Time,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, OperationalError

metadata = MetaData()
UTC_TS = DateTime(timezone=True)
ALEMBIC_HEAD_REVISION = "20260727_0028"


def _audit_columns() -> tuple[Column[Any], Column[Any]]:
    return (
        Column("created_at_utc", UTC_TS, nullable=False, server_default=text("CURRENT_TIMESTAMP")),
        Column("updated_at_utc", UTC_TS, nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    )


users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", Text, nullable=False, unique=True),
    Column("password_hash", Text, nullable=False),
    Column("role", Text, nullable=False, server_default="user"),
    Column("is_active", Integer, nullable=False, server_default="1"),
    Column("created_at_utc", UTC_TS, nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    Column("email", Text), Column("full_name", Text),
    Column("last_login_utc", UTC_TS), Column("login_attempts", Integer, nullable=False, server_default="0"),
    Column("locked_until_utc", UTC_TS),
    CheckConstraint("role IN ('superadmin', 'admin', 'user')", name="ck_users_role"),
)
Index("idx_users_username", users.c.username)
Index("idx_users_email", users.c.email)

revoked_tokens = Table(
    "revoked_tokens", metadata,
    Column("jti", Text, primary_key=True),
    Column("revoked_at_utc", UTC_TS, nullable=False),
    Column("expires_at_utc", UTC_TS, nullable=False),
)
Index("idx_revoked_expires", revoked_tokens.c.expires_at_utc)

face_quality_settings = Table(
    "face_quality_settings", metadata,
    Column("singleton", Integer, primary_key=True), Column("quality_threshold", Float, nullable=False),
    Column("blur_threshold", Float, nullable=False), Column("min_face_width", Integer, nullable=False),
    Column("min_face_height", Integer, nullable=False), Column("min_eye_distance", Float, nullable=False),
    Column("max_abs_yaw", Float, nullable=False), Column("max_abs_pitch", Float, nullable=False),
    Column("max_abs_roll", Float, nullable=False), Column("require_landmarks", Integer, nullable=False),
    Column("human_pose_enabled", Integer, nullable=False, server_default="1"),
    Column("human_pose_min_keypoints", Integer, nullable=False, server_default="4"),
    Column("human_pose_keypoint_confidence", Float, nullable=False, server_default="0.25"),
    Column("recognition_quality_weight", Float, nullable=False, server_default="0.5"),
    CheckConstraint("singleton = 1", name="ck_face_quality_singleton"),
)

detection_logs = Table(
    "detection_logs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("source_system", String(64), nullable=False, server_default="face_recognition"),
    Column("source_event_key", String(255), unique=True, nullable=True),
    Column("source_human_log_id", Integer),
    Column("personnel_id", Integer),
    Column("person", String(255), nullable=False, server_default="Unknown"),
    Column("confidence", Float, nullable=False, server_default="0"),
    Column("detection_time", UTC_TS, nullable=False),
    Column("ref_img_id", String(255)),
    Column("room_id", Integer),
    Column("camera_id", Text),
    Column("access_granted", Integer, nullable=False, server_default="0"),
    Column("counts_for_attendance", Integer, nullable=False, server_default="1"),
    Column("log_type", String(64), nullable=False, server_default="real_time"),
    Column("import_source_parts", Text),
    Column("face_image", Text),
    Column("body_image", Text),
    Column("snapshot_image", Text),
    Column("video", Text),
    Column("face_video_or_unknown_faces", Text),
    Column("created_by", Integer),
    Column("updated_by", Integer),
    *_audit_columns(),
)
Index("idx_detection_logs_time", detection_logs.c.detection_time)
Index("idx_detection_logs_personnel", detection_logs.c.personnel_id)
Index("idx_detection_logs_person", detection_logs.c.person)
Index("idx_detection_logs_room", detection_logs.c.room_id)
Index("idx_detection_logs_camera", detection_logs.c.camera_id)
Index("idx_detection_logs_attendance", detection_logs.c.counts_for_attendance)
Index("idx_detection_logs_type", detection_logs.c.log_type)
Index("idx_detection_logs_source_event", detection_logs.c.source_event_key)

fire_smoke_logs = Table(
    "fire_smoke_logs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True), Column("camera", Text, nullable=False),
    Column("time", UTC_TS, nullable=False), Column("incident_id", Text), Column("severity", Text, nullable=False),
    Column("fire_count", Integer, nullable=False), Column("smoke_count", Integer, nullable=False),
    Column("fire_confidence", Float, nullable=False), Column("smoke_confidence", Float, nullable=False),
    Column("window_seconds", Float, nullable=False), Column("snapshot_url", Text, nullable=False, server_default=""),
    Column("video_url", Text, nullable=False, server_default=""),
    Column("details_json", Text, nullable=False, server_default="{}"),
)
Index("idx_fire_smoke_logs_time", fire_smoke_logs.c.time)
Index("idx_fire_smoke_logs_camera", fire_smoke_logs.c.camera)
Index("idx_fire_smoke_logs_severity", fire_smoke_logs.c.severity)
Index("idx_fire_smoke_logs_incident", fire_smoke_logs.c.incident_id)

fire_smoke_settings = Table(
    "fire_smoke_settings", metadata,
    Column("id", Integer, primary_key=True), Column("window_seconds", Float, nullable=False),
    Column("low_count", Integer, nullable=False), Column("medium_count", Integer, nullable=False),
    Column("high_count", Integer, nullable=False), Column("updated_at_utc", UTC_TS, nullable=False),
    CheckConstraint("id = 1", name="ck_fire_smoke_settings_singleton"),
)

holidays = Table(
    "holidays", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True), Column("name", Text, nullable=False),
    Column("date_value", Date, nullable=False), Column("description", Text),
    Column("holiday_type", Text, nullable=False, server_default="national"),
    Column("every_year", Integer, nullable=False, server_default="0"),
    Column("is_active", Integer, nullable=False, server_default="1"), *_audit_columns(),
)
Index("idx_holidays_date", holidays.c.date_value)
Index("idx_holidays_active", holidays.c.is_active)

work_shifts = Table(
    "work_shifts", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True), Column("shift_name", Text, nullable=False),
    Column("shift_type", Text, nullable=False, server_default="morning"),
    Column("start_time", Time, nullable=False, server_default=text("TIME '08:00'")),
    Column("end_time", Time, nullable=False, server_default=text("TIME '16:00'")),
    Column("timezone_name", String(64), nullable=False, server_default="Asia/Tehran"),
    Column("max_minutes_delay", Integer, nullable=False, server_default="15"),
    Column("max_minutes_early", Integer, nullable=False, server_default="15"),
    Column("max_overtime_hours", Float, nullable=False, server_default="2.0"),
    *[Column(name, Integer, nullable=False, server_default=("0" if name in {"works_saturday", "works_sunday", "works_friday"} else "1")) for name in (
        "works_saturday", "works_sunday", "works_monday", "works_tuesday", "works_wednesday", "works_thursday", "works_friday"
    )], *_audit_columns(),
)

personnel = Table(
    "personnel", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True), Column("fname", Text, nullable=False),
    Column("lname", Text, nullable=False), Column("national_code", Text, nullable=False, unique=True),
    Column("employee_type", Text, nullable=False, server_default="unknown"), Column("degree", Text),
    Column("shift_id", Integer),
    Column("department_id", Integer),
    Column("last_seen", UTC_TS), *_audit_columns(),
)
Index("idx_personnel_national_code", personnel.c.national_code)
Index("idx_personnel_name", personnel.c.lname, personnel.c.fname)
Index("idx_personnel_shift", personnel.c.shift_id)
Index("idx_personnel_department", personnel.c.department_id)

personnel_images = Table(
    "personnel_images", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("personnel_id", Integer, ForeignKey("personnel.id", ondelete="CASCADE"), nullable=False),
    Column("storage_key", Text, nullable=False), Column("description", Text),
    Column("is_primary", Integer, nullable=False, server_default="0"),
    Column("uploaded_at_utc", UTC_TS, nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    Column("embedding_id", Text),
)
Index("idx_personnel_images_personnel", personnel_images.c.personnel_id)
Index("idx_personnel_images_primary", personnel_images.c.personnel_id, personnel_images.c.is_primary)

buildings = Table(
    "buildings", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True), Column("name", Text, nullable=False),
    Column("address", Text), Column("description", Text), *_audit_columns(),
)
Index("idx_buildings_name", buildings.c.name)
sections = Table(
    "sections", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("building_id", Integer, ForeignKey("buildings.id", ondelete="SET NULL")),
    Column("name", Text, nullable=False), Column("description", Text), *_audit_columns(),
)
Index("idx_sections_building", sections.c.building_id); Index("idx_sections_name", sections.c.name)
rooms = Table(
    "rooms", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    # Kept as a compatibility/cache column for existing zone and reporting code.
    # New room writes derive it from cam.section_id.
    Column("section_id", Integer, ForeignKey("sections.id", ondelete="SET NULL")),
    Column("cam_id", Integer, ForeignKey("cam.id", ondelete="RESTRICT")),
    Column("name", Text, nullable=False), Column("description", Text), Column("polygon_json", Text), *_audit_columns(),
)
Index("idx_rooms_section", rooms.c.section_id); Index("idx_rooms_cam", rooms.c.cam_id); Index("idx_rooms_name", rooms.c.name)
cam = Table(
    "cam", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("camera_name", Text, nullable=False),
    Column("camera_number", Integer, nullable=False),
    Column("width", Integer, nullable=False),
    Column("high", Integer, nullable=False),
    Column("source_type", String(16), nullable=False),
    Column("section_id", Integer, ForeignKey("sections.id", ondelete="RESTRICT"), nullable=False),
    Column("url", Text, nullable=False),
    *_audit_columns(),
    CheckConstraint("camera_number > 0", name="ck_cam_camera_number_positive"),
    CheckConstraint("width > 0", name="ck_cam_width_positive"),
    CheckConstraint("high > 0", name="ck_cam_high_positive"),
    CheckConstraint("source_type IN ('usb', 'rtsp', 'other')", name="ck_cam_source_type"),
    UniqueConstraint("section_id", "camera_number", name="uq_cam_section_camera_number"),
)
Index("idx_cam_section", cam.c.section_id)
Index("idx_cam_source_type", cam.c.source_type)
personnel_room_access = Table(
    "personnel_room_access", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("personnel_id", Integer, nullable=False),
    Column("room_id", Integer, ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False),
    Column("granted_at_utc", UTC_TS, nullable=False, server_default=text("CURRENT_TIMESTAMP")), Column("granted_by", Text),
    UniqueConstraint("personnel_id", "room_id", name="uq_personnel_room_access"),
)
Index("idx_access_personnel", personnel_room_access.c.personnel_id); Index("idx_access_room", personnel_room_access.c.room_id)
detection_room_matches = Table(
    "detection_room_matches", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True), Column("detection_type", Text, nullable=False),
    Column("detection_event_id", Integer, nullable=False),
    Column("room_id", Integer, ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False),
    Column("personnel_id", Integer),
    Column("camera_id", Text),
    Column("track_id", Integer),
    Column("transition_type", Text),
    Column("matched_at_utc", UTC_TS, nullable=False, server_default=text("CURRENT_TIMESTAMP")),
)
Index("idx_matches_room", detection_room_matches.c.room_id); Index("idx_matches_personnel", detection_room_matches.c.personnel_id)
Index("idx_matches_detection", detection_room_matches.c.detection_type, detection_room_matches.c.detection_event_id)

human_logs = Table(
    "human_logs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True), Column("session_id", Text, nullable=False),
    Column("camera", Text, nullable=False), Column("track_id", Integer, nullable=False),
    Column("name", Text, nullable=False, server_default="Unknown"),
    Column("first_seen", UTC_TS, nullable=False), Column("last_seen", UTC_TS, nullable=False),
    Column("recognition_score", Float, nullable=False, server_default="0"), Column("ref_img_id", Text),
    Column("snapshot_url", Text, nullable=False, server_default=""), Column("video_url", Text, nullable=False, server_default=""),
    Column("face_video_url", Text, nullable=False, server_default=""), Column("snapshot_quality", Float, nullable=False, server_default="0"),
    Column("best_face_quality", Float, nullable=False, server_default="0"),
    Column("full_frame_video_frames", Integer, nullable=False, server_default="0"),
    Column("accepted_face_frames", Integer, nullable=False, server_default="0"),
    Column("personnel_id", Integer),
    Column("counts_for_attendance", Integer, nullable=False, server_default="1"),
    UniqueConstraint("session_id", "camera", "track_id", name="uq_human_session_camera_track"),
)
Index("idx_human_logs_camera", human_logs.c.camera); Index("idx_human_logs_name", human_logs.c.name); Index("idx_human_logs_last_seen", human_logs.c.last_seen)

car_plates = Table(
    "car_plates", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("left_digits", String(2), nullable=False),
    Column("plate_alphabet", String(1), nullable=False),
    Column("right_digits", String(3), nullable=False),
    Column("iran_code", String(2), nullable=False),
    Column("plate_format", String(32), nullable=False, server_default="standard"),
    Column("usage_type", String(32), nullable=False),
    Column("vehicle_type", String(32), nullable=False),
    Column("owner_name", String(200), nullable=False),
    Column("owner_phone", String(16), nullable=False),
    Column("color", String(50)), Column("brand", String(80)), Column("model", String(80)),
    Column("manufacture_year", Integer), Column("description", Text),
    Column("is_active", Integer, nullable=False, server_default="1"),
    Column("deleted_at_utc", UTC_TS),
    Column("created_at_utc", UTC_TS, nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    Column("updated_at_utc", UTC_TS, nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    Column("created_by", Integer), Column("updated_by", Integer),
)
Index("idx_car_plates_owner_phone", car_plates.c.owner_phone)
Index("idx_car_plates_active", car_plates.c.is_active)
Index("idx_car_plates_usage_type", car_plates.c.usage_type)
Index("idx_car_plates_vehicle_type", car_plates.c.vehicle_type)

plate_logs = Table(
    "plate_logs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("camera", Text, nullable=False),
    Column("time", UTC_TS, nullable=False),
    Column("plate", Text, nullable=False),
    Column("snapshot_url", Text, nullable=False, server_default=""),
    Column("video_url", Text, nullable=False, server_default=""),
    Column("details_json", Text, nullable=False, server_default="{}"),
    Column("plate_id", Integer),
    Column("plate_full_number", String(32)),
    Column("raw_plate_text", String(64)),
    Column("detection_time", UTC_TS),
    Column("camera_id", String(200)),
    Column("confidence", Float),
    Column("direction", String(16), server_default="unknown"),
    Column("source_type", String(16), server_default="camera"),
    Column("snapshot_path", String(512)),
    Column("plate_crop_path", String(512)),
    Column("is_verified", Integer, server_default="0"),
    Column("created_by_user_id", Integer),
    Column("verified_by_user_id", Integer),
    Column("verified_at", UTC_TS),
    Column("notes", Text),
    Column("created_at", UTC_TS),
    Column("updated_at", UTC_TS),
)
Index("idx_plate_logs_time", plate_logs.c.time)
Index("idx_plate_logs_camera", plate_logs.c.camera)
Index("idx_plate_logs_plate", plate_logs.c.plate)
Index("idx_plate_logs_plate_id", plate_logs.c.plate_id)
Index("idx_plate_logs_full_number", plate_logs.c.plate_full_number)
Index("idx_plate_logs_camera_id", plate_logs.c.camera_id)
Index("idx_plate_logs_detection_time", plate_logs.c.detection_time)
Index("idx_plate_logs_direction", plate_logs.c.direction)

plate_general_settings = Table(
    "plate_general_settings", metadata,
    Column("id", Integer, primary_key=True), Column("vehicle_confidence", Float, nullable=False),
    Column("plate_confidence", Float, nullable=False), Column("ocr_confidence", Float, nullable=False),
    Column("min_vehicle_width_pixels", Integer, nullable=False), Column("min_vehicle_height_pixels", Integer, nullable=False),
    Column("min_vehicle_area_ratio", Float, nullable=False), Column("vehicle_crop_padding_ratio", Float, nullable=False),
    Column("updated_at_utc", UTC_TS, nullable=False), CheckConstraint("id = 1", name="ck_plate_general_singleton"),
)
plate_camera_settings = Table(
    "plate_camera_settings", metadata,
    Column("camera_id", Text, primary_key=True), Column("vehicle_confidence", Float), Column("plate_confidence", Float),
    Column("ocr_confidence", Float), Column("min_vehicle_width_pixels", Integer), Column("min_vehicle_height_pixels", Integer),
    Column("min_vehicle_area_ratio", Float), Column("vehicle_crop_padding_ratio", Float), Column("updated_at_utc", UTC_TS, nullable=False),
)

model_general_settings = Table(
    "model_general_settings", metadata,
    Column("id", Integer, primary_key=True), Column("fire_smoke_model", Text, nullable=False),
    Column("vehicle_detector_model", Text, nullable=False), Column("plate_detector_model", Text, nullable=False),
    Column("preferred_format", Text, nullable=False), Column("allow_onnx_fallback", Integer, nullable=False),
    Column("allow_pt_fallback", Integer, nullable=False), Column("export_imgsz", Integer, nullable=False),
    Column("export_batch_size", Integer, nullable=False), Column("export_workspace_gb", Float, nullable=False),
    Column("export_half", Integer, nullable=False), Column("export_dynamic", Integer, nullable=False),
    Column("export_timeout_seconds", Integer, nullable=False, server_default="300"), Column("updated_at_utc", UTC_TS, nullable=False),
    CheckConstraint("id = 1", name="ck_model_general_singleton"),
)
sources = Table(
    "sources", metadata,
    Column("id", Integer),
    Column("source_uri", Text, primary_key=True),
    Column("name", Text),
    Column("enabled", Integer, server_default="1"),
    Column("tasks_json", Text, server_default="[]"),
    Column("frame_width", Integer, server_default="640"),
    Column("frame_height", Integer, server_default="640"),
    Column("room_id", Integer, ForeignKey("rooms.id", ondelete="SET NULL")),
    Column("source_type", String(16), server_default="rtsp"),
    Column("metadata_json", Text, server_default="{}"),
    Column("fps", Float),
    Column("created_at_utc", UTC_TS),
    Column("updated_at_utc", UTC_TS, nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    Column("fire_confidence", Float),
    Column("smoke_confidence", Float),
    Column("plate_confidence", Float),
    Column("plate_iou", Float),
    Column("vehicle_confidence", Float),
    Column("vehicle_iou", Float),
    Column("face_human_confidence", Float),
    Column("face_detection_confidence", Float),
    Column("face_recognition_threshold", Float),
    Column("loop", Integer, server_default="1"),
    Column("draw_human", Integer, server_default="1"),
    Column("draw_zone", Integer, server_default="1"),
    Column("draw_fire", Integer, server_default="1"),
    Column("draw_smoke", Integer, server_default="1"),
    Column("draw_vehicle", Integer, server_default="1"),
    Column("draw_plate", Integer, server_default="1"),
)
UniqueConstraint("id", name="uq_sources_id")
Index("idx_sources_enabled", sources.c.enabled)
Index("idx_sources_room", sources.c.room_id)
general_settings = Table(
    "general_settings", metadata,
    Column("id", Integer, primary_key=True),
    Column("enable_processing", Integer, nullable=False, server_default="1"),
    Column("process_fire", Integer, nullable=False, server_default="0"),
    Column("process_plate", Integer, nullable=False, server_default="0"),
    Column("counts_for_attendance", Integer, nullable=False, server_default="1"),
    Column("margin_level", Float, nullable=False, server_default="1.0"),
    Column("draw_box", Integer, nullable=False, server_default="1"),
    Column("draw_face", Integer, nullable=False, server_default="1"),
    Column("draw_skeleton", Integer, nullable=False, server_default="0"),
    Column("draw_zones", Integer, nullable=False, server_default="1"),
    Column("face_rec_score", Float, nullable=False, server_default="0.4"),
    Column("face_det_score", Float, nullable=False, server_default="0.4"),
    Column("human_det_score", Float, nullable=False, server_default="0.4"),
    Column("confirmation_threshold", Float, nullable=False, server_default="0.6"),
    Column("force", Integer, nullable=False, server_default="0"),
    Column("operational_json", Text, nullable=False, server_default="{}"),
    Column("created_by", Integer, ForeignKey("users.id", ondelete="SET NULL")),
    Column("updated_by", Integer, ForeignKey("users.id", ondelete="SET NULL")),
    *_audit_columns(),
    CheckConstraint("id = 1", name="ck_general_settings_singleton"),
    CheckConstraint("margin_level >= 1.0 AND margin_level <= 5.0", name="ck_margin_level_range"),
    CheckConstraint("face_rec_score >= 0.0 AND face_rec_score <= 1.0", name="ck_face_rec_score_range"),
    CheckConstraint("face_det_score >= 0.0 AND face_det_score <= 1.0", name="ck_face_det_score_range"),
    CheckConstraint("human_det_score >= 0.0 AND human_det_score <= 1.0", name="ck_human_det_score_range"),
    CheckConstraint("confirmation_threshold >= 0.0 AND confirmation_threshold <= 1.0", name="ck_confirmation_threshold_range"),
    CheckConstraint("force IN (0, 1)", name="ck_general_settings_force"),
    CheckConstraint("confirmation_threshold >= face_rec_score", name="ck_confirmation_ge_face_rec"),
)

model_conversion_jobs = Table(
    "model_conversion_jobs", metadata,
    Column("job_id", Text, primary_key=True), Column("role", Text, nullable=False), Column("source_model", Text, nullable=False),
    Column("output_directory", Text, nullable=False), Column("status", Text, nullable=False),
    Column("created_at_utc", UTC_TS, nullable=False), Column("updated_at_utc", UTC_TS, nullable=False),
    Column("options_json", Text, nullable=False), Column("artifacts_json", Text, nullable=False), Column("errors_json", Text, nullable=False),
)

personnel_requests = Table(
    "personnel_requests", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("personnel_id", Integer, ForeignKey("personnel.id", ondelete="CASCADE"), nullable=False),
    Column("request_type", Text, nullable=False, server_default="earned_leave"),
    Column("duration_type", String(16)),
    Column("start_date", Date, nullable=False),
    Column("end_date", Date, nullable=False),
    Column("start_time", Time),
    Column("end_time", Time),
    Column("duration_days", Float),
    Column("duration_minutes", Integer),
    Column("reason", Text), Column("status", Text, nullable=False, server_default="pending"),
    Column("approved_by", Integer), Column("reviewed_by", Integer),
    Column("reviewed_at", UTC_TS),
    Column("admin_notes", Text), Column("rejection_reason", Text), *_audit_columns(),
)
Index("idx_requests_personnel", personnel_requests.c.personnel_id); Index("idx_requests_status", personnel_requests.c.status)
Index("idx_requests_dates", personnel_requests.c.start_date, personnel_requests.c.end_date)
Index("idx_requests_duration_type", personnel_requests.c.duration_type)
Index("idx_requests_reviewed_by", personnel_requests.c.reviewed_by)


face_embeddings = Table(
    "face_embeddings", metadata,
    Column("id", Text, primary_key=True),
    Column("person", Text, nullable=False),
    Column("ref_img_id", Text),
    Column("embedding", LargeBinary, nullable=False),
    Column("dimension", Integer, nullable=False),
    Column("created_at_utc", UTC_TS, nullable=False, server_default=text("CURRENT_TIMESTAMP")),
)
Index("idx_face_embeddings_person", face_embeddings.c.person)
Index("idx_face_embeddings_ref_img", face_embeddings.c.ref_img_id)

import_progress = Table(
    "import_progress", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("import_type", Text, nullable=False),
    Column("source_filename", Text, nullable=False),
    Column("total_rows", Integer, nullable=False, server_default="0"),
    Column("imported_rows", Integer, nullable=False, server_default="0"),
    Column("skipped_rows", Integer, nullable=False, server_default="0"),
    Column("failed_rows", Integer, nullable=False, server_default="0"),
    Column("status", Text, nullable=False, server_default="running"),
    Column("error_message", Text),
    Column("created_by", Integer, ForeignKey("users.id", ondelete="SET NULL")),
    *_audit_columns(),
)
Index("idx_import_progress_type", import_progress.c.import_type)
Index("idx_import_progress_status", import_progress.c.status)
Index("idx_import_progress_created_by", import_progress.c.created_by)

static_videos = Table(
    "static_videos", metadata,
    Column("source_uri", Text, primary_key=True),
    Column("name", Text, nullable=False),
    Column("source_type", String(16), nullable=False, server_default="static_video"),
)

_RETURNING_ID_TABLES = {
    table.name
    for table in metadata.tables.values()
    if "id" in table.c
    and table.c.id.primary_key
    and isinstance(table.c.id.type, Integer)
    and table.c.id.autoincrement is not False
}


def _format_value(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        return value.strftime("%H:%M:%S").rstrip("0").rstrip(":") if value.second or value.microsecond else value.strftime("%H:%M")
    return value


class Row(Mapping[str, Any]):
    def __init__(self, keys: Sequence[str], values: Sequence[Any]) -> None:
        self._keys = tuple(keys)
        self._values = tuple(_format_value(v) for v in values)
        self._mapping = dict(zip(self._keys, self._values))
    def __getitem__(self, key: str | int) -> Any:
        return self._values[key] if isinstance(key, int) else self._mapping[key]
    def __iter__(self) -> Iterator[str]: return iter(self._keys)
    def __len__(self) -> int: return len(self._keys)
    def keys(self): return self._mapping.keys()


class Cursor:
    def __init__(self, rows: list[Row] | None = None, rowcount: int = -1, lastrowid: int | None = None) -> None:
        self._rows = rows or []; self.rowcount = rowcount; self.lastrowid = lastrowid
    def fetchone(self) -> Row | None: return self._rows[0] if self._rows else None
    def fetchall(self) -> list[Row]: return list(self._rows)
    def __iter__(self) -> Iterator[Row]: return iter(self._rows)


def _replace_qmarks(sql: str) -> str:
    """Convert the stores' positional placeholders to psycopg2 placeholders."""
    out: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(sql):
        ch = sql[i]
        if quote:
            out.append(ch)
            if ch == quote:
                if i + 1 < len(sql) and sql[i + 1] == quote:
                    out.append(sql[i + 1])
                    i += 1
                else:
                    quote = None
        elif ch in {"'", '"'}:
            quote = ch
            out.append(ch)
        elif ch == "?":
            out.append("%s")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


class Connection:
    """Small transaction wrapper around a SQLAlchemy PostgreSQL connection."""

    def __init__(self, engine: Engine) -> None:
        self._conn = engine.connect()
        self._closed = False

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> Cursor:
        statement = _replace_qmarks(sql.strip().rstrip(";"))
        if not statement:
            return Cursor([], 0)
        params_tuple = tuple(params or ())
        insert_match = re.match(r"\s*INSERT\s+INTO\s+([a-zA-Z_][\w]*)", statement, re.I)
        wants_id = bool(
            insert_match
            and insert_match.group(1).lower() in _RETURNING_ID_TABLES
            and "RETURNING" not in statement.upper()
        )
        if wants_id:
            statement += " RETURNING id"
        result = self._conn.exec_driver_sql(statement, params_tuple)
        if result.returns_rows:
            keys = list(result.keys())
            raw_rows = result.fetchall()
            rows = [Row(keys, tuple(row)) for row in raw_rows]
            lastrowid = int(rows[0][0]) if wants_id and rows else None
            return Cursor([] if wants_id else rows, result.rowcount, lastrowid)
        return Cursor([], result.rowcount)

    def executemany(self, sql: str, seq_of_params: Iterable[Sequence[Any]]) -> Cursor:
        total = 0
        for params in seq_of_params:
            total += max(0, self.execute(sql, params).rowcount)
        return Cursor([], total)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        if not self._closed:
            self._conn.close()
            self._closed = True

    def __enter__(self) -> "Connection":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.rollback() if exc_type else self.commit()
        finally:
            self.close()


class Database:
    """Shared PostgreSQL engine. Alembic owns all schema creation and upgrades."""

    def __init__(
        self,
        url: str,
        *,
        echo: bool = False,
        pool_size: int = 10,
        max_overflow: int = 20,
    ) -> None:
        if not url.startswith(("postgresql+psycopg2://", "postgresql://")):
            raise ValueError("DATABASE_URL must be a PostgreSQL psycopg2 URL")
        self.url = url
        try:
            self.engine = create_engine(
                url,
                echo=echo,
                pool_pre_ping=True,
                pool_size=pool_size,
                max_overflow=max_overflow,
                connect_args={"options": "-c timezone=UTC"},
                future=True,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PostgreSQL requires psycopg2. Install requirements-postgres.txt."
            ) from exc

    def connection(self) -> Connection:
        return Connection(self.engine)

    def verify_connection(self) -> None:
        with self.engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")

    def verify_schema(self) -> None:
        required = set(metadata.tables) | {"alembic_version"}
        with self.engine.connect() as connection:
            rows = connection.exec_driver_sql(
                "SELECT tablename FROM pg_catalog.pg_tables "
                "WHERE schemaname = current_schema()"
            ).fetchall()
            existing = {str(row[0]) for row in rows}
            missing = sorted(required - existing)
            if missing:
                raise RuntimeError(
                    "PostgreSQL schema is not initialized. Run `alembic upgrade head`. "
                    f"Missing tables: {', '.join(missing)}"
                )
            revision = connection.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            ).scalar_one_or_none()
        if revision != ALEMBIC_HEAD_REVISION:
            raise RuntimeError(
                "PostgreSQL schema revision is not current. "
                f"Expected {ALEMBIC_HEAD_REVISION}, found {revision!r}. "
                "Run `alembic upgrade head`."
            )

    def dispose(self) -> None:
        self.engine.dispose()


_DATABASES: dict[str, Database] = {}
_DATABASES_LOCK = threading.Lock()


def get_database(
    url: str,
    *,
    echo: bool = False,
    pool_size: int = 10,
    max_overflow: int = 20,
) -> Database:
    with _DATABASES_LOCK:
        db = _DATABASES.get(url)
        if db is None:
            db = Database(
                url,
                echo=echo,
                pool_size=pool_size,
                max_overflow=max_overflow,
            )
            _DATABASES[url] = db
        return db


def ensure_database(value: Database | str) -> Database:
    return value if isinstance(value, Database) else get_database(str(value))
