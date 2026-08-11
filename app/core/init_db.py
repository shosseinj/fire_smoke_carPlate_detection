from __future__ import annotations

"""Database seeding for initial setup and development.

Provides seed data for personnel, detection logs, and foundational
records (building, section, shift, room) to bootstrap a new deployment.
All seed operations are idempotent — they run only when the target
tables are empty. Called from build_runtime() at startup.
"""

import json
import logging
import random
from datetime import date, datetime, timedelta, timezone
from typing import Any

from app.core.cam_store import CamRecord, CamStore
from app.core.detection_log_store import DetectionLogStore
from app.core.employee_type_store import EmployeeTypeStore
from app.core.location_store import LocationStore
from app.core.holiday_import import seed_default_official_holidays
from app.core.holiday_store import HolidayStore
from app.core.jalali_utils import parse_jalali_date
from app.core.personnel_store import PersonnelStore
from app.core.shift_store import ShiftStore
from app.core.source_registry import SourceRecord, SourceRegistry
from app.core.types import TaskName

LOGGER = logging.getLogger(__name__)

# ── Seed switches ─────────────────────────────────────────────────────
# Toggle initial/default data from this single file.  Schema creation and
# structural data migrations remain Alembic's responsibility.
CREATE_EMPLOYEE_TYPES = True
CREATE_BUILDINGS = True
CREATE_SECTIONS = True
CREATE_ROOMS = True
CREATE_CAMERAS = True
CREATE_SHIFTS = True
CREATE_HOLIDAYS = True
CREATE_PERSONNEL = True
CREATE_PERSONNEL_SHIFT_ASSIGNMENTS = True
CREATE_SAMPLE_DETECTION_LOGS = False

_SEED_FIRST_SHIFT_START_DATE: date = parse_jalali_date("1405-01-01")
_SEED_FIRST_SHIFT_END_DATE: date = parse_jalali_date("1405-02-31")
_SEED_SECOND_SHIFT_START_DATE: date = parse_jalali_date("1405-03-01")
_SEED_SECOND_SHIFT_END_DATE: date = parse_jalali_date("1405-12-29")
_LEGACY_SEED_SHIFT_START_DATE: date = parse_jalali_date("1405-01-01")
_LEGACY_SEED_SHIFT_END_DATE: date = parse_jalali_date("1405-12-29")

# ── Code maps (from the legacy init_database.py) ───────────────────────
_DEGREE_CODE_MAP: dict[str, str] = {
    "1": "بیسواد",
    "2": "زیر دیپلم",
    "3": "دیپلم",
    "4": "فوق دیپلم",
    "5": "لیسانس",
    "6": "فوق لیسانس",
    "7": "دکتری",
    "8": "نامشخص",
}

# Default employee types.  Names are intentionally Persian because the lookup
# table no longer has a separate machine ``code`` field.
DEFAULT_EMPLOYEE_TYPE_SEED_DATA: list[dict[str, Any]] = [
    {"name": "پیمانکار", "include_in_attendance_reports": False},
    {"name": "مشتری", "include_in_attendance_reports": False},
    {"name": "مهمان", "include_in_attendance_reports": False},
    {"name": "کارمند", "include_in_attendance_reports": True},
    {"name": "نامشخص", "include_in_attendance_reports": False},
]

# Personnel seed data from the legacy init_database.py.
DEFAULT_PERSONNEL_SEED_DATA: list[dict[str, Any]] = [
    {
        "fname": "رضا",
        "lname": "محمدلو",
        "national_code": "0311344119",
        "employee_type_name": "کارمند",
        "department_id": 1,
        "shift_id": 1,
        "degree_code": "5",
    },
    {
        "fname": "حسین",
        "lname": "حسین زاده",
        "national_code": "0410500666",
        "employee_type_name": "کارمند",
        "department_id": 1,
        "shift_id": 1,
        "degree_code": "4",
    },
    {
        "fname": "پوریا",
        "lname": "ابوحمزه",
        "national_code": "0430205562",
        "employee_type_name": "کارمند",
        "department_id": 1,
        "shift_id": 1,
        "degree_code": "4",
    },
    {
        "fname": "علی",
        "lname": "ابوحمزه",
        "national_code": "0430212097",
        "employee_type_name": "کارمند",
        "department_id": 1,
        "shift_id": 1,
        "degree_code": "4",
    },
    {
        "fname": "حسین",
        "lname": "جعفری",
        "national_code": "1451141981",
        "employee_type_name": "کارمند",
        "department_id": 1,
        "shift_id": 1,
        "degree_code": "6",
    },
    {
        "fname": "امین",
        "lname": "شریفی",
        "national_code": "3920139313",
        "employee_type_name": "کارمند",
        "department_id": 1,
        "shift_id": 1,
        "degree_code": "5",
    },
    {
        "fname": "صفی اله",
        "lname": "کریمی",
        "national_code": "3932041755",
        "employee_type_name": "کارمند",
        "department_id": 1,
        "shift_id": 1,
        "degree_code": "6",
    },
    {
        "fname": "امید",
        "lname": "کریمی",
        "national_code": "0010691782",
        "employee_type_name": "کارمند",
        "department_id": None,
        "shift_id": None,
        "degree_code": None,
    },
]


def create_default_employee_types(employee_type_store: EmployeeTypeStore) -> int:
    """Create the default employee types only when the table is empty.

    This is initial deployment seed data, not a reconciliation job.  If an
    administrator later renames or deletes an unused type, startup must not
    recreate or overwrite it.
    """
    if employee_type_store.list():
        return 0

    created = 0
    for item in DEFAULT_EMPLOYEE_TYPE_SEED_DATA:
        name = str(item["name"])
        try:
            record = employee_type_store.create(
                name=name,
                is_active=True,
                include_in_attendance_reports=bool(
                    item.get("include_in_attendance_reports", False)
                ),
            )
            created += 1
            LOGGER.info("INIT_DB created employee type '%s' id=%d", record.name, record.id)
        except ValueError as exc:
            LOGGER.warning("INIT_DB skip employee type '%s': %s", name, exc)
    return created


# ── Camera URLs (from the legacy init_database.py) ────────────────────
_SEED_CAMERA_URLS: list[str] = [
    "rtsp://admin:pMc897OmId@192.168.110.14:554/Streaming/Channels/101",
    "rtsp://admin:pMc897OmId@192.168.110.28:554/Streaming/Channels/101",
    "rtsp://admin:pMc897OmId@192.168.110.20:554/Streaming/Channels/101",
    "rtsp://admin:pMc897OmId@192.168.110.20:554/Streaming/Channels/201",
    "rtsp://admin:pMc897OmId@192.168.110.20:554/Streaming/Channels/301",
    "rtsp://admin:pMc897OmId@192.168.110.20:554/Streaming/Channels/401",
    "rtsp://admin:pMc897OmId@192.168.110.20:554/Streaming/Channels/501",
    "rtsp://admin:pMc897OmId@192.168.110.29:554/Streaming/Channels/101",
]


def _camera_name_from_url(url: str, index: int) -> str:
    """Generate a human-readable camera name from the URL, matching the
    naming convention in the legacy init_database.py."""
    if "192.168.110.14" in url:
        return "Camera 1 - Main Entrance"
    if "192.168.110.28" in url:
        return "Camera 2 - Reception"
    if "/Streaming/Channels/101" in url:
        return f"Camera {index} - Channel 101"
    if "/Streaming/Channels/201" in url:
        return f"Camera {index} - Channel 201"
    if "/Streaming/Channels/301" in url:
        return f"Camera {index} - Channel 301"
    if "/Streaming/Channels/401" in url:
        return f"Camera {index} - Channel 401"
    if "/Streaming/Channels/501" in url:
        return f"Camera {index} - Channel 501"
    return f"Camera {index}"


def _seed_camera_id(url: str, index: int) -> str:
    """Derive a stable source_id from the URL."""
    if "192.168.110.14" in url:
        return "camera_01"
    if "192.168.110.28" in url:
        return "camera_02"
    parts = url.replace("rtsp://", "").split("/")
    ip_port = parts[0] if parts else f"camera_{index}"
    channel = parts[-1] if len(parts) > 1 else f"ch{index}"
    safe = ip_port.replace(".", "_").replace(":", "_")
    return f"camera_{safe}_{channel}"


_DEFAULT_SEED_TASKS: set[TaskName] = {
    TaskName.FIRE_SMOKE,
    TaskName.PLATE_RECOGNITION,
    TaskName.FACE_RECOGNITION,
}


def create_default_cameras(
    registry: SourceRegistry,
) -> list[SourceRecord]:
    """Create seed cameras from :data:`_SEED_CAMERA_URLS` if the cameras
    table is empty. Returns the list of records (existing or created)."""
    existing = registry.list()
    if existing:
        return existing

    created: list[SourceRecord] = []
    for idx, url in enumerate(_SEED_CAMERA_URLS, 1):
        try:
            record = registry.create(
                SourceRecord(
                    source_uri=url,
                    name=_camera_name_from_url(url, idx),
                    enabled=True,
                    tasks=_DEFAULT_SEED_TASKS,
                    source_type="rtsp",
                )
            )
            created.append(record)
            LOGGER.info("INIT_DB created source camera '%s'", record.name)
        except Exception as exc:
            LOGGER.warning("INIT_DB skip source camera %d: %s", idx, exc)
    return created


def create_default_cam_records(
    cam_store: CamStore,
    section_id: int,
    urls: list[str] | None = None,
) -> tuple[list[CamRecord], int]:
    """Ensure each seed camera has a hierarchy record in ``cam``."""
    existing, _ = cam_store.list(section_id=section_id, limit=1000)
    by_url = {record.url: record for record in existing}
    by_number = {record.camera_number: record for record in existing}
    records: list[CamRecord] = []
    created_count = 0

    selected_urls = set(_SEED_CAMERA_URLS if urls is None else urls)
    for index, url in enumerate(_SEED_CAMERA_URLS, 1):
        if url not in selected_urls:
            continue
        record = by_url.get(url)
        if record is not None:
            records.append(record)
            continue

        conflict = by_number.get(index)
        if conflict is not None:
            LOGGER.warning(
                "INIT_DB skip cam %d: camera_number already belongs to '%s'",
                index,
                conflict.camera_name,
            )
            continue

        try:
            record = cam_store.create(
                camera_name=_camera_name_from_url(url, index),
                camera_number=index,
                width=640,
                high=640,
                source_type="rtsp",
                section_id=section_id,
                url=url,
            )
        except ValueError as exc:
            LOGGER.warning("INIT_DB skip cam %d: %s", index, exc)
            continue

        records.append(record)
        by_url[url] = record
        by_number[index] = record
        created_count += 1
        LOGGER.info("INIT_DB created cam '%s' id=%d", record.camera_name, record.id)

    return records, created_count


def create_rooms_for_cameras(
    location_store: LocationStore,
    cameras: list[SourceRecord],
    section_id: int | None = None,
    cam_records: list[CamRecord] | None = None,
) -> int:
    """Create one room per camera under *section_id* using the camera
    name as the room name and a full-frame polygon.

    Skips any camera whose name already matches an existing room name
    in that section. Returns the number of rooms created.
    """
    existing_rooms, _ = location_store.list_rooms(section_id=section_id, limit=1000)
    rooms_by_name = {record.name: record for record in existing_rooms if record.name}
    cams_by_url = {record.url: record for record in cam_records or []}
    count = 0
    for source in cameras:
        room_name = source.name
        cam = cams_by_url.get(source.source_uri)
        existing_room = rooms_by_name.get(room_name)
        if existing_room is not None:
            if cam is not None and existing_room.cam_id is None:
                updated = location_store.update_room(existing_room.id, cam_id=cam.id)
                if updated is not None:
                    rooms_by_name[room_name] = updated
                    count += 1
            elif cam is not None and existing_room.cam_id != cam.id:
                LOGGER.warning(
                    "INIT_DB room '%s' already belongs to cam %d",
                    room_name,
                    existing_room.cam_id,
                )
            continue
        try:
            room = location_store.create_room(
                name=room_name,
                section_id=section_id if cam is None else None,
                cam_id=cam.id if cam is not None else None,
                description=f"Room monitored by {source.name}",
                polygon_json=json.dumps([[0, 0], [640, 0], [640, 640], [0, 640]]),
            )
            count += 1
            rooms_by_name[room_name] = room
            LOGGER.info("INIT_DB created room '%s' for camera '%s'", room_name, source.name)
        except Exception as exc:
            LOGGER.warning("INIT_DB skip room for camera '%s': %s", source.name, exc)
    return count


def _seed_normalize_national_code(value: Any) -> str:
    """Normalize Excel-like national-code values and preserve leading zeros."""
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(int(value)).zfill(10)
    text_value = str(value).strip()
    if "." in text_value:
        text_value = text_value.split(".", 1)[0]
    return text_value.zfill(10)


def create_default_personnel(
    personnel_store: PersonnelStore,
    employee_type_store: EmployeeTypeStore,
    department_id: int | None = None,
    shift_id: int | None = None,
) -> int:
    """Create default personnel from seed data if the table is empty.

    Employee types are resolved by Persian name rather than hardcoded IDs, so
    seed correctness does not depend on sequence values.
    """
    if personnel_store.count() > 0:
        return 0

    type_ids_by_name = {
        record.name: record.id for record in employee_type_store.list()
    }
    count = 0
    for item in DEFAULT_PERSONNEL_SEED_DATA:
        try:
            national_code = _seed_normalize_national_code(item.get("national_code"))
            employee_type_name = str(item.get("employee_type_name") or "نامشخص")
            employee_type_id = type_ids_by_name.get(employee_type_name)
            if employee_type_id is None:
                raise ValueError(
                    f"Employee type seed '{employee_type_name}' does not exist"
                )
            degree_code = item.get("degree_code")
            personnel_store.create(
                fname=item["fname"],
                lname=item["lname"],
                national_code=national_code,
                employee_type=None,
                employee_type_id=employee_type_id,
                degree=_DEGREE_CODE_MAP.get(str(degree_code)) if degree_code is not None else None,
                shift_id=shift_id,
                department_id=department_id,
            )
            count += 1
        except ValueError as exc:
            LOGGER.warning(
                "INIT_DB skip personnel %s %s: %s",
                item.get("fname"),
                item.get("lname"),
                exc,
            )
    return count


def create_default_detection_logs(
    detection_log_store: DetectionLogStore,
    personnel_ids: list[int],
    room_id: int | None = None,
    target_count: int = 100,
) -> int:
    """Create development-only sample detection logs when the table is empty."""
    by_status = detection_log_store.count_by_status()
    if by_status and any("log_type:" in key for key in by_status):
        return 0
    if not personnel_ids:
        return 0

    base_time = datetime.now(timezone.utc)
    count = 0
    for i in range(target_count):
        personnel_id = random.choice(personnel_ids)
        detection_time = base_time - timedelta(minutes=i * 30)
        try:
            detection_log_store.create(
                source_system="face_recognition",
                personnel_id=personnel_id,
                person=f"Personnel_{personnel_id}",
                confidence=random.uniform(0.5, 1.0),
                detection_time=detection_time.isoformat(),
                room_id=room_id,
                camera_id=None,
                access_granted=random.random() > 0.2,
                counts_for_attendance=True,
                log_type="camera_rtsp",
            )
            count += 1
        except Exception as exc:
            LOGGER.warning("INIT_DB skip detection log %d: %s", i, exc)
    return count


# ── Default shifts (from the legacy init_database.py) ────────────────
_DEFAULT_SHIFTS: list[dict[str, Any]] = [
    {
        "shift_name": "شیفت صبح",
        "shift_type": "morning",
        "start_time": "07:00",
        "end_time": "15:30",
        "max_overtime_hours": 8.0,
        "max_minutes_delay": 15,
        "max_minutes_early": 0,
        "works_saturday": True,
        "works_sunday": True,
        "works_monday": True,
        "works_tuesday": True,
        "works_wednesday": True,
        "works_thursday": False,
        "works_friday": False,
    },
    {
        "shift_name": "شیفت عصر",
        "shift_type": "evening",
        "start_time": "16:00",
        "end_time": "00:00",
        "max_overtime_hours": 8.0,
        "max_minutes_delay": 15,
        "max_minutes_early": 0,
        "works_saturday": True,
        "works_sunday": True,
        "works_monday": True,
        "works_tuesday": True,
        "works_wednesday": True,
        "works_thursday": False,
        "works_friday": False,
    },
    {
        "shift_name": "شیفت شب",
        "shift_type": "night",
        "start_time": "00:00",
        "end_time": "08:00",
        "max_overtime_hours": 8.0,
        "max_minutes_delay": 15,
        "max_minutes_early": 0,
        "works_saturday": True,
        "works_sunday": True,
        "works_monday": True,
        "works_tuesday": True,
        "works_wednesday": True,
        "works_thursday": False,
        "works_friday": False,
    },
    {
        "shift_name": "دورکاری",
        "shift_type": "remote",
        "start_time": "09:00",
        "end_time": "17:00",
        "max_overtime_hours": 8.0,
        "max_minutes_delay": 15,
        "max_minutes_early": 0,
        "works_saturday": False,
        "works_sunday": False,
        "works_monday": True,
        "works_tuesday": True,
        "works_wednesday": True,
        "works_thursday": False,
        "works_friday": False,
    },
    {
        "shift_name": "شیفت منعطف",
        "shift_type": "flexible",
        "start_time": "07:00",
        "end_time": "19:00",
        "max_overtime_hours": 8.0,
        "max_minutes_delay": 15,
        "max_minutes_early": 0,
        "works_saturday": True,
        "works_sunday": True,
        "works_monday": True,
        "works_tuesday": True,
        "works_wednesday": True,
        "works_thursday": False,
        "works_friday": False,
    },
    {
        "shift_name": "شیفت چرخشی",
        "shift_type": "rotating",
        "start_time": "08:00",
        "end_time": "20:00",
        "max_overtime_hours": 8.0,
        "max_minutes_delay": 15,
        "max_minutes_early": 0,
        "works_saturday": True,
        "works_sunday": True,
        "works_monday": True,
        "works_tuesday": True,
        "works_wednesday": True,
        "works_thursday": False,
        "works_friday": False,
    },
    {
        "shift_name": "شیفت جنگ",
        "shift_type": "morning",
        "start_time": "07:00",
        "end_time": "14:00",
        "max_overtime_hours": 0.0,
        "max_minutes_delay": 15,
        "max_minutes_early": 0,
        "works_saturday": True,
        "works_sunday": True,
        "works_monday": True,
        "works_tuesday": True,
        "works_wednesday": True,
        "works_thursday": False,
        "works_friday": False,
    },
]


def _create_all_default_shifts(shift_store: ShiftStore) -> list[Any]:
    """Idempotently create missing default shifts and return all seeded defaults."""
    existing, _ = shift_store.list(limit=100)
    existing_by_name = {shift.shift_name: shift for shift in existing}

    created: list[Any] = []
    for s in _DEFAULT_SHIFTS:
        existing_shift = existing_by_name.get(s["shift_name"])
        if existing_shift is not None:
            created.append(existing_shift)
            continue
        try:
            shift = shift_store.create(
                shift_name=s["shift_name"],
                shift_type=s["shift_type"],
                start_time=s["start_time"],
                end_time=s["end_time"],
                max_overtime_hours=s["max_overtime_hours"],
                max_minutes_delay=s["max_minutes_delay"],
                max_minutes_early=s["max_minutes_early"],
                works_saturday=s["works_saturday"],
                works_sunday=s["works_sunday"],
                works_monday=s["works_monday"],
                works_tuesday=s["works_tuesday"],
                works_wednesday=s["works_wednesday"],
                works_thursday=s["works_thursday"],
                works_friday=s["works_friday"],
            )
            created.append(shift)
            LOGGER.info("INIT_DB created shift '%s' id=%d", s["shift_name"], shift.id)
        except Exception as exc:
            LOGGER.warning("INIT_DB skip shift '%s': %s", s["shift_name"], exc)
    return created


def init_database(
    *,
    personnel_store: PersonnelStore | None = None,
    employee_type_store: EmployeeTypeStore | None = None,
    detection_log_store: DetectionLogStore | None = None,
    location_store: LocationStore | None = None,
    shift_store: ShiftStore | None = None,
    holiday_store: HolidayStore | None = None,
    registry: SourceRegistry | None = None,
    cam_store: CamStore | None = None,
    target_log_count: int = 100,
    seed_sample_detections: bool = False,
) -> bool:
    """Seed enabled initial/default data.

    Every seed category is controlled by the ``CREATE_*`` switches at the top
    of this file and is idempotent.  Alembic is responsible only for schema
    changes and migration of pre-existing data, not normal default seed rows.
    """
    seeded = False

    # ── Employee types ─────────────────────────────────────────────
    if CREATE_EMPLOYEE_TYPES and employee_type_store is not None:
        if create_default_employee_types(employee_type_store) > 0:
            seeded = True

    # ── Building / section / base room ─────────────────────────────
    section: Any = None
    room: Any = None
    default_polygon = json.dumps([[0, 0], [640, 0], [640, 640], [0, 640]])
    if location_store is not None:
        building_records, _ = location_store.list_buildings(limit=1)
        building = building_records[0] if building_records else None
        if building is None and CREATE_BUILDINGS:
            building = location_store.create_building(
                name="مرغاب",
                address="Tehran",
                description="بهترین واحد هوش مصنوعی دنیا!",
            )
            LOGGER.info("INIT_DB created building id=%d", building.id)
            seeded = True

        section_records, _ = location_store.list_sections(limit=1)
        section = section_records[0] if section_records else None
        if section is None and CREATE_SECTIONS:
            if building is None:
                LOGGER.warning(
                    "INIT_DB cannot create section: no building exists and CREATE_BUILDINGS is disabled"
                )
            else:
                section = location_store.create_section(
                    name="معاونت هوش مصنوعی",
                    building_id=building.id,
                    description="شاخه هوش مصنوعی",
                )
                LOGGER.info("INIT_DB created section id=%d", section.id)
                seeded = True

        room_records, _ = location_store.list_rooms(limit=1)
        room = room_records[0] if room_records else None
        if room is None and CREATE_ROOMS:
            if section is None:
                LOGGER.warning(
                    "INIT_DB cannot create room: no section exists and CREATE_SECTIONS is disabled/unavailable"
                )
            else:
                room = location_store.create_room(
                    name="اتاق نظارت",
                    section_id=section.id,
                    description="اتاق پایش و نظارت تصویری",
                    polygon_json=default_polygon,
                )
                LOGGER.info("INIT_DB created room id=%d", room.id)
                seeded = True

    # ── Cameras (source registry + cam hierarchy records) ───────────
    if CREATE_CAMERAS and registry is not None and section is not None:
        source_count_before = len(registry.list())
        seed_cameras = create_default_cameras(registry)
        if len(seed_cameras) > source_count_before:
            seeded = True

        seed_cam_records: list[CamRecord] = []
        if cam_store is not None:
            seed_urls = [
                source.source_uri
                for source in seed_cameras
                if source.source_uri in _SEED_CAMERA_URLS
            ]
            seed_cam_records, cams_created = create_default_cam_records(
                cam_store, section.id, urls=seed_urls
            )
            if cams_created:
                seeded = True

        if CREATE_ROOMS and location_store is not None and seed_cameras:
            rooms_created = create_rooms_for_cameras(
                location_store,
                seed_cameras,
                section_id=section.id,
                cam_records=seed_cam_records,
            )
            if rooms_created > 0:
                LOGGER.info(
                    "INIT_DB created %d room(s) for seed cameras", rooms_created
                )
                seeded = True

            rooms, _ = location_store.list_rooms(section_id=section.id, limit=1000)
            rooms_by_name = {item.name: item for item in rooms}
            cams_by_url = {item.url: item for item in seed_cam_records}
            for source in seed_cameras:
                source_room = rooms_by_name.get(source.name)
                cam = cams_by_url.get(source.source_uri)
                if source_room is None:
                    continue
                if cam is not None and source_room.cam_id != cam.id:
                    LOGGER.warning(
                        "INIT_DB skip source room assignment for '%s': cam ownership mismatch",
                        source.name,
                    )
                    continue
                if source.room_id != source_room.id:
                    registry.update(source.source_uri, room_id=source_room.id)
                    seeded = True

    # ── Shifts ─────────────────────────────────────────────────────
    shifts: list[Any] = []
    if shift_store is not None:
        if CREATE_SHIFTS:
            shift_count_before = shift_store.count()
            shifts = _create_all_default_shifts(shift_store)
            if shift_store.count() > shift_count_before:
                seeded = True
        else:
            shifts, _ = shift_store.list(limit=100)
    shift = shifts[0] if shifts else None
    shifts_by_name = {item.shift_name: item for item in shifts}
    first_seed_shift = shifts_by_name.get("شیفت جنگ")
    second_seed_shift = shifts_by_name.get("شیفت صبح")

    # ── Official Iran holidays ─────────────────────────────────────
    if CREATE_HOLIDAYS and holiday_store is not None:
        holiday_seed = seed_default_official_holidays(holiday_store)
        if bool(holiday_seed["seeded"]):
            LOGGER.info(
                "INIT_DB created %d official holiday record(s) for Jalali year 1405",
                holiday_seed["inserted_count"],
            )
            seeded = True

    # ── Personnel ──────────────────────────────────────────────────
    if CREATE_PERSONNEL and personnel_store is not None:
        if employee_type_store is None:
            LOGGER.warning(
                "INIT_DB cannot create personnel: employee_type_store is unavailable"
            )
        else:
            dept_id = section.id if section is not None else None
            shift_id = shift.id if shift is not None else None
            created = create_default_personnel(
                personnel_store,
                employee_type_store,
                department_id=dept_id,
                shift_id=shift_id,
            )
            if created > 0:
                LOGGER.info("INIT_DB created %d personnel record(s)", created)
                seeded = True

    # ── Dated personnel shift assignments ──────────────────────────
    if (
        CREATE_PERSONNEL_SHIFT_ASSIGNMENTS
        and personnel_store is not None
        and shift_store is not None
        and first_seed_shift is not None
        and second_seed_shift is not None
    ):
        personnel_records, _ = personnel_store.list(limit=1000)
        for person in personnel_records:
            desired_assignments = (
                (
                    second_seed_shift.id,
                    _SEED_MORNING_1404_START_DATE,
                    _SEED_MORNING_1404_END_DATE,
                ),
                (
                    first_seed_shift.id,
                    _SEED_FIRST_SHIFT_START_DATE,
                    _SEED_FIRST_SHIFT_END_DATE,
                ),
                (
                    second_seed_shift.id,
                    _SEED_SECOND_SHIFT_START_DATE,
                    _SEED_SECOND_SHIFT_END_DATE,
                ),
            )
            existing_assignments = shift_store.list_assignments(person.id)
            if (
                len(existing_assignments) == 1
                and existing_assignments[0].shift_id == second_seed_shift.id
                and existing_assignments[0].start_date == _LEGACY_SEED_SHIFT_START_DATE
                and existing_assignments[0].end_date == _LEGACY_SEED_SHIFT_END_DATE
            ):
                shift_store.delete_assignment(existing_assignments[0].id)
                existing_assignments = []
                seeded = True

            for desired_shift_id, assignment_start, assignment_end in desired_assignments:
                if any(
                    assignment.shift_id == desired_shift_id
                    and assignment.start_date == assignment_start
                    and assignment.end_date == assignment_end
                    for assignment in existing_assignments
                ):
                    continue
                try:
                    assignment = shift_store.assign_personnel(
                        person.id,
                        desired_shift_id,
                        assignment_start,
                        assignment_end,
                    )
                    existing_assignments.append(assignment)
                    seeded = True
                except ValueError as exc:
                    LOGGER.warning(
                        "INIT_DB skip dated shift assignment for personnel %s: %s",
                        person.id,
                        exc,
                    )

    # ── Optional development detection logs ────────────────────────
    # Sample detections are intentionally disabled by default.  Keep both the
    # source-code seed switch and the runtime setting as explicit opt-ins.
    if (
        CREATE_SAMPLE_DETECTION_LOGS
        and seed_sample_detections
        and detection_log_store is not None
        and personnel_store is not None
    ):
        personnel_records, _ = personnel_store.list(limit=1000)
        personnel_ids = [person.id for person in personnel_records]
        room_id = room.id if room is not None else None
        created = create_default_detection_logs(
            detection_log_store,
            personnel_ids=personnel_ids,
            room_id=room_id,
            target_count=target_log_count,
        )
        if created > 0:
            LOGGER.info("INIT_DB created %d detection log(s)", created)
            seeded = True

    return seeded
