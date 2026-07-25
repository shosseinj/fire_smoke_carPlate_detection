from __future__ import annotations

import asyncio
import base64
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import cv2

if TYPE_CHECKING:
    from app.runtime import Runtime

RECENT_DETECTIONS_LIMIT = max(0, int(os.getenv("WEBSOCKET_RECENT_DETECTIONS_LIMIT", "50")))
_JALALI_TZ = ZoneInfo(os.getenv("BUSINESS_TIMEZONE", "Asia/Tehran"))


def _parse_datetime(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _to_jalali_str(value: str | datetime | None) -> str | None:
    dt = _parse_datetime(value)
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(_JALALI_TZ)

    gy, gm, gd = dt.year, dt.month, dt.day
    g_days = [31, 28 + int((gy % 4 == 0 and gy % 100 != 0) or gy % 400 == 0), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    jy = gy - 1600
    gm -= 1
    gd -= 1
    day_no = 365 * jy + (jy + 3) // 4 - (jy + 99) // 100 + (jy + 399) // 400
    day_no += sum(g_days[:gm]) + gd
    j_day_no = day_no - 79
    cycles = j_day_no // 12053
    j_day_no %= 12053
    jy = 979 + 33 * cycles + 4 * (j_day_no // 1461)
    j_day_no %= 1461
    if j_day_no >= 366:
        jy += (j_day_no - 1) // 365
        j_day_no = (j_day_no - 1) % 365
    for index, days in enumerate([31, 31, 31, 31, 31, 31, 30, 30, 30, 30, 30, 29]):
        if j_day_no < days:
            jm = index + 1
            jd = j_day_no + 1
            break
        j_day_no -= days
    return f"{jy}/{jm:02d}/{jd:02d} {dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}"


def _classification(confidence: float, face_rec_score: float, confirmation_threshold: float) -> tuple[str, bool]:
    if confidence < face_rec_score:
        return "unknown", False
    if confidence < confirmation_threshold:
        return "unsure", True
    return "known", True


def _read_image(path: Path | None):
    if path is None or not path.is_file():
        return None
    return cv2.imread(str(path))


def _concat_if_needed(image, reference):
    if image is None or reference is None:
        return image
    if image.shape[:2] != reference.shape[:2]:
        reference = cv2.resize(reference, (image.shape[1], image.shape[0]))
    if len(image.shape) != len(reference.shape):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if len(reference.shape) == 2:
            reference = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] != reference.shape[2]:
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if reference.shape[2] == 1:
            reference = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    if image.dtype != reference.dtype:
        reference = reference.astype(image.dtype)
    return cv2.hconcat([image, reference])


def _media_path(runtime: Runtime, raw: str | None) -> Path | None:
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    media_root = runtime.settings.saved_media_path.resolve()
    direct = media_root / candidate
    if direct.is_file():
        return direct
    project_relative = Path.cwd() / candidate
    return project_relative if project_relative.is_file() else direct


def _reference_path(runtime: Runtime, row: dict[str, Any]) -> Path | None:
    ref_img_id = row.get("ref_img_id")
    if ref_img_id is not None:
        try:
            image = runtime.personnel_store.get_image(int(ref_img_id))
        except (TypeError, ValueError):
            image = None
        if image is not None:
            return runtime.personnel_store.get_image_path(image.storage_key)
    personnel_id = row.get("personnel_id")
    if personnel_id is not None:
        images = runtime.personnel_store.list_images(int(personnel_id))
        if images:
            return runtime.personnel_store.get_image_path(images[0].storage_key)
    return None


def _thresholds(runtime: Runtime, camera_id: str | None) -> tuple[float, float]:
    settings = runtime.general_settings.get()
    face_rec = float(settings.face_rec_score)
    confirmation = float(settings.confirmation_threshold)
    if camera_id:
        camera = runtime.registry.get(camera_id)
        if camera is not None:
            overrides = dict(camera.metadata.get("_settings_overrides") or {})
            face_rec = float(overrides.get("face_rec_score", face_rec))
            confirmation = float(overrides.get("confirmation_threshold", confirmation))
    return face_rec, confirmation


def _build_payload_from_enriched_row(runtime: Runtime, row: dict[str, Any]) -> dict[str, Any] | None:
    confidence = float(row.get("confidence") or 0.0)
    face_rec, confirmation = _thresholds(runtime, row.get("camera_id"))
    classification, concatenate = _classification(confidence, face_rec, confirmation)
    person = row.get("person") or "Unknown"
    fname = row.get("fname")
    lname = row.get("lname")
    if fname or lname:
        full_name = f"{fname or ''} {lname or ''}".strip()
    else:
        full_name = person

    face_image_b64: str | None = None
    image = _read_image(_media_path(runtime, row.get("face_image")))
    if image is not None:
        if concatenate:
            image = _concat_if_needed(image, _read_image(_reference_path(runtime, row)))
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if success:
            face_image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

    return {
        "id": row["id"],
        "area": row.get("room_name") or ("بدون ناحیه" if not row.get("room_id") else "ناحیه نامشخص"),
        "person": person,
        "full_name": full_name,
        "confidence": confidence,
        "detection_time": _to_jalali_str(row.get("detection_time")),
        "face_image_base64": face_image_b64,
        "access_granted": bool(row.get("access_granted")),
        "counts_for_attendance": bool(row.get("counts_for_attendance")),
        "classification": classification,
    }


def get_recent_detection_payloads(runtime: Runtime, limit: int = RECENT_DETECTIONS_LIMIT) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    with runtime.database.connection() as conn:
        rows = conn.execute(
            "SELECT d.*, p.fname, p.lname, p.national_code, r.name AS room_name "
            "FROM detection_logs d "
            "LEFT JOIN personnel p ON p.id = d.personnel_id "
            "LEFT JOIN rooms r ON r.id = d.room_id "
            "ORDER BY d.detection_time DESC LIMIT ?",
            (limit,),
        ).fetchall()

    payloads: list[dict[str, Any]] = []
    for raw_row in rows:
        payload = _build_payload_from_enriched_row(runtime, dict(raw_row))
        if payload is not None:
            payloads.append(payload)
    return payloads


def get_single_detection_payload_by_id(runtime: Runtime, log_id: int) -> dict[str, Any] | None:
    record = runtime.detection_log_store.get(log_id)
    if record is None:
        return None
    fname = lname = None
    if record.personnel_id is not None:
        personnel = runtime.personnel_store.get(record.personnel_id)
        if personnel is not None:
            fname = personnel.fname
            lname = personnel.lname
    room_name = None
    if record.room_id is not None:
        room = runtime.location_store.get_room(record.room_id)
        if room is not None:
            room_name = room.name
    row = {
        "id": record.id,
        "confidence": record.confidence,
        "camera_id": record.camera_id,
        "face_image": record.face_image,
        "ref_img_id": record.ref_img_id,
        "personnel_id": record.personnel_id,
        "person": record.person,
        "fname": fname,
        "lname": lname,
        "room_name": room_name,
        "room_id": record.room_id,
        "detection_time": record.detection_time,
        "access_granted": record.access_granted,
        "counts_for_attendance": record.counts_for_attendance,
    }
    return _build_payload_from_enriched_row(runtime, row)


def build_recent_detections_message(runtime: Runtime, limit: int = RECENT_DETECTIONS_LIMIT) -> dict[str, Any] | None:
    detections = get_recent_detection_payloads(runtime, limit)
    if not detections:
        return None
    return {"type": "recent_detections", "detections": detections, "count": len(detections)}


def build_recent_detection_refresh_message(payload: dict[str, Any], log_id: int) -> dict[str, Any]:
    return {
        "type": "recent_detections",
        "detections": [payload],
        "count": 1,
        "reason": "log_updated",
        "updated_log_id": log_id,
    }


async def get_and_build_refresh_message(runtime: Runtime, log_id: int) -> dict[str, Any] | None:
    payload = await asyncio.to_thread(get_single_detection_payload_by_id, runtime, log_id)
    if payload is None:
        return None
    return build_recent_detection_refresh_message(payload, log_id)
