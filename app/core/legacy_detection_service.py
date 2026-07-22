from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import jdatetime

from app.config import settings
from app.core.detection_log_store import DetectionLogRecord
from app.core.jalali_utils import gregorian_to_jalali_str
from app.time_utils import utc_now

LOGGER = logging.getLogger(__name__)


def _tehran_tz() -> ZoneInfo:
    return ZoneInfo("Asia/Tehran")


def _parse_iso(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def format_jalali(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    tehran_dt = dt.astimezone(_tehran_tz())
    j_date = jdatetime.date.fromgregorian(date=tehran_dt.date())
    return (
        f"{j_date.year:04d}-{j_date.month:02d}-{j_date.day:02d} "
        f"{tehran_dt.hour:02d}:{tehran_dt.minute:02d}"
    )


def format_jalali_date(d: date) -> str:
    return gregorian_to_jalali_str(d)


def format_time_hhmm(minutes: int) -> str:
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def legacy_detection_response(
    record: DetectionLogRecord,
    full_name: str | None = None,
    room_name: str | None = None,
    camera_name: str | None = None,
    section_name: str | None = None,
    building_name: str | None = None,
    created_by_username: str | None = None,
    updated_by_username: str | None = None,
    include_detail: bool = False,
) -> dict:
    tz = _tehran_tz()

    def _to_local(utc_str: str) -> str | None:
        if not utc_str:
            return None
        try:
            dt = _parse_iso(utc_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(tz).isoformat()
        except (ValueError, TypeError):
            return utc_str

    def _jalali(utc_str: str) -> str | None:
        if not utc_str:
            return None
        try:
            dt = _parse_iso(utc_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return format_jalali(dt)
        except (ValueError, TypeError):
            return None

    detection_local = _to_local(record.detection_time)
    detection_jalali = _jalali(record.detection_time)
    created_local = _to_local(record.created_at_utc)
    created_jalali = _jalali(record.created_at_utc)
    updated_local = _to_local(record.updated_at_utc)
    updated_jalali = _jalali(record.updated_at_utc)

    result: dict[str, Any] = {
        "id": record.id,
        "person": record.person,
        "confidence": record.confidence,
        "detection_time": record.detection_time,
        "detection_time_utc": record.detection_time,
        "detection_time_local": detection_local,
        "detection_time_jalali": detection_jalali,
        "created_at": record.created_at_utc,
        "created_at_utc": record.created_at_utc,
        "created_at_local": created_local,
        "created_at_jalali": created_jalali,
        "updated_at": record.updated_at_utc,
        "updated_at_utc": record.updated_at_utc,
        "updated_at_local": updated_local,
        "updated_at_jalali": updated_jalali,
        "face_image_url": record.face_image,
        "face_thumbnail": record.face_image,
        "body_image_url": record.body_image,
        "body_thumbnail": record.body_image,
        "snapshot_image_url": record.snapshot_image,
        "snapshot_thumbnail": record.snapshot_image,
        "video_url": record.video,
        "face_video_url": record.face_video_or_unknown_faces,
        "unknown_faces_path": record.face_video_or_unknown_faces,
        "log_type": record.log_type,
        "ref_img_id": record.ref_img_id,
        "full_name": full_name,
        "room_id": record.room_id,
        "access_granted": record.access_granted,
        "counts_for_attendance": record.counts_for_attendance,
        "created_by": record.created_by,
        "updated_by": record.updated_by,
        "created_by_username": created_by_username,
        "updated_by_username": updated_by_username,
        "room_name": room_name,
        "camera_name": camera_name,
        "section_name": section_name,
        "building_name": building_name,
    }

    if include_detail:
        fname: str | None = None
        lname: str | None = None
        if full_name:
            parts = full_name.split(" ", 1)
            fname = parts[0] if parts else None
            lname = parts[1] if len(parts) > 1 else None
        result.update({
            "fname": fname,
            "lname": lname,
            "personnel_id": record.personnel_id,
            "camera_id": record.camera_id,
            "section_id": None,
            "building_id": None,
            "import_source_parts": record.import_source_parts,
            "face_image": record.face_image,
            "body_image": record.body_image,
            "snapshot_image": record.snapshot_image,
            "reference_image": record.face_image,
        })

    return result


def calculate_access(
    personnel_id: int | None,
    room_id: int | None,
    location_store: Any,
) -> bool:
    if personnel_id is None:
        return False
    if room_id is None:
        return True
    room = location_store.get_room(room_id)
    if room is not None:
        room_name_lower = (room.name or "").lower()
        if "general" in room_name_lower or "عمومی" in room_name_lower:
            return True
    if location_store.check_room_access(personnel_id, room_id):
        return True
    return False


def compute_dedup_key(personnel_id: int | None, room_id: int | None) -> str:
    return f"{personnel_id}:{room_id}"


def evaluate_dedup_replacement(
    existing: DetectionLogRecord,
    new_confidence: float,
) -> tuple[bool, bool]:
    if new_confidence > existing.confidence:
        return (True, False)
    return (False, True)


def detection_time_to_utc(
    date_str: str,
    time_str: str | None = None,
    tz_name: str = "Asia/Tehran",
) -> str:
    tz = ZoneInfo(tz_name)
    if time_str:
        dt_str = f"{date_str} {time_str}"
    else:
        dt_str = date_str
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    utc_dt = dt.astimezone(timezone.utc)
    return utc_dt.isoformat()


def utc_to_local_parts(utc_str: str, tz_name: str = "Asia/Tehran") -> dict:
    tz = ZoneInfo(tz_name)
    dt = _parse_iso(utc_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local_dt = dt.astimezone(tz)
    j_date = jdatetime.date.fromgregorian(date=local_dt.date())
    jalali_str = (
        f"{j_date.year:04d}-{j_date.month:02d}-{j_date.day:02d} "
        f"{local_dt.hour:02d}:{local_dt.minute:02d}"
    )
    return {
        "utc": dt.isoformat(),
        "local": local_dt.isoformat(),
        "jalali": jalali_str,
    }
