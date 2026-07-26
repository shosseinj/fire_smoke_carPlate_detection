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
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.cam_store import CamRecord, CamStore
from app.core.detection_log_store import DetectionLogStore
from app.core.location_store import LocationStore
from app.core.personnel_store import PersonnelStore
from app.core.shift_store import ShiftStore
from app.core.source_registry import SourceRecord, SourceRegistry
from app.core.types import TaskName

LOGGER = logging.getLogger(__name__)

# ── Code maps (from the legacy init_database.py) ───────────────────────
_EMPLOYEE_TYPE_CODE_MAP: dict[str, str] = {
    "1": "contractor",
    "2": "customer",
    "3": "guest",
    "4": "employee",
    "5": "unknown",
}

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

# Personnel seed data from the legacy init_database.py.
DEFAULT_PERSONNEL_SEED_DATA: list[dict[str, Any]] = [
    {
        "fname": "رضا",
        "lname": "محمدلو",
        "national_code": "0311344119",
        "employee_type_code": "4",
        "department_id": 1,
        "shift_id": 1,
        "degree_code": "5",
    },
    {
        "fname": "حسین",
        "lname": "حسین زاده",
        "national_code": "0410500666",
        "employee_type_code": "4",
        "department_id": 1,
        "shift_id": 1,
        "degree_code": "5",
    },
    {
        "fname": "پوریا",
        "lname": "ابوحمزه",
        "national_code": "0430205562",
        "employee_type_code": "4",
        "department_id": 1,
        "shift_id": 1,
        "degree_code": "5",
    },
    {
        "fname": "علی",
        "lname": "ابوحمزه",
        "national_code": "0430212097",
        "employee_type_code": "4",
        "department_id": 1,
        "shift_id": 1,
        "degree_code": "5",
    },
    {
        "fname": "حسین",
        "lname": "جعفری",
        "national_code": "1451141981",
        "employee_type_code": "4",
        "department_id": 1,
        "shift_id": 1,
        "degree_code": "5",
    },
    {
        "fname": "امین",
        "lname": "شریفی",
        "national_code": "3920139313",
        "employee_type_code": "4",
        "department_id": 1,
        "shift_id": 1,
        "degree_code": "5",
    },
    {
        "fname": "صفی اله",
        "lname": "کریمی",
        "national_code": "3932041755",
        "employee_type_code": "4",
        "department_id": 1,
        "shift_id": 1,
        "degree_code": "5",
    },
    {
        "fname": "امید",
        "lname": "کریمی",
        "national_code": "0010691782",
        "employee_type_code": "4",
        "department_id": None,
        "shift_id": None,
        "degree_code": None,
    },
]


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
    department_id: int | None = None,
    shift_id: int | None = None,
) -> int:
    """Create default personnel from seed data if the table is empty.

    Returns the number of personnel created (0 if the table already
    contained records).
    """
    if personnel_store.count() > 0:
        return 0
    count = 0
    for item in DEFAULT_PERSONNEL_SEED_DATA:
        try:
            national_code = _seed_normalize_national_code(item.get("national_code"))
            employee_type_code = str(item.get("employee_type_code") or "5")
            degree_code = item.get("degree_code")
            personnel_store.create(
                fname=item["fname"],
                lname=item["lname"],
                national_code=national_code,
                employee_type=_EMPLOYEE_TYPE_CODE_MAP.get(employee_type_code, "employee"),
                degree=_DEGREE_CODE_MAP.get(str(degree_code)) if degree_code is not None else None,
                shift_id=item.get("shift_id") or shift_id,
                department_id=item.get("department_id") or department_id,
            )
            count += 1
        except ValueError as exc:
            LOGGER.warning("INIT_DB skip personnel %s %s: %s", item.get("fname"), item.get("lname"), exc)
    return count


def create_default_detection_logs(
    detection_log_store: DetectionLogStore,
    personnel_ids: list[int],
    room_id: int | None = None,
    target_count: int = 100,
) -> int:
    """Create sample detection logs if the table is empty.

    Logs are spread across the available personnel and assigned
    ``log_type="camera_rtsp"`` to match the old project convention.
    Roughly 20% of logs have ``access_granted=False``.

    Returns the number of logs created (0 if the table already
    contained records).
    """
    by_status = detection_log_store.count_by_status()
    if by_status and any("log_type:" in k for k in by_status):
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
        "max_minutes_early": 15,
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
        "max_minutes_early": 15,
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
        "max_minutes_early": 15,
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
        "max_minutes_early": 15,
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
        "max_minutes_early": 15,
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
        "max_minutes_early": 15,
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
    """Idempotently create all default shifts. Returns existing shift list if any exist."""
    existing, _ = shift_store.list(limit=100)
    if existing:
        return existing

    created: list[Any] = []
    for s in _DEFAULT_SHIFTS:
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
    detection_log_store: DetectionLogStore | None = None,
    location_store: LocationStore | None = None,
    shift_store: ShiftStore | None = None,
    registry: SourceRegistry | None = None,
    cam_store: CamStore | None = None,
    target_log_count: int = 100,
) -> bool:
    """Seed the database with foundational records and sample data.

    Creates a seed building → section → room hierarchy, default
    work shifts, default personnel from
    :data:`DEFAULT_PERSONNEL_SEED_DATA`, and sample detection logs.
    Every operation is idempotent.

    Call this function from ``build_runtime()`` (or the startup
    entrypoint) after all stores have been constructed.

    Parameters
    ----------
    personnel_store, detection_log_store, location_store, shift_store,
    registry, cam_store:
        Store instances through which seed operations are performed.
        Any store that is ``None`` is skipped.
    target_log_count:
        Desired number of detection logs when seeding from empty.

    Returns
    -------
    True if any seed data was written.
    """
    seeded = False

    # ── Seed building → section → room ──────────────────────────────
    section: Any = None
    room: Any = None
    _DEFAULT_POLYGON = json.dumps([[0, 0], [640, 0], [640, 640], [0, 640]])
    if location_store is not None:
        building_records, _ = location_store.list_buildings(limit=1)
        if not building_records:
            building = location_store.create_building(
                name="مرغاب",
                address="Tehran",
                description="بهترین واحد هوش مصنوعی دنیا!",
            )
            LOGGER.info("INIT_DB created building id=%d", building.id)
            seeded = True
        else:
            building = building_records[0]

        section_records, _ = location_store.list_sections(limit=1)
        if not section_records:
            section = location_store.create_section(
                name="معاونت هوش مصنوعی",
                building_id=building.id,
                description="شاخه هوش مصنوعی",
            )
            LOGGER.info("INIT_DB created section id=%d", section.id)
            seeded = True
        else:
            section = section_records[0]

        room_records, _ = location_store.list_rooms(limit=1)
        if not room_records:
            room = location_store.create_room(
                name="اتاق نظارت",
                section_id=section.id,
                description="اتاق پایش و نظارت تصویری",
                polygon_json=_DEFAULT_POLYGON,
            )
            LOGGER.info("INIT_DB created room id=%d", room.id)
            seeded = True
        else:
            room = room_records[0]

    # ── Seed default cameras and per-camera rooms ────────────────────
    if registry is not None and section is not None:
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
                cam_store,
                section.id,
                urls=seed_urls,
            )
            if cams_created:
                seeded = True
        if location_store is not None and seed_cameras:
            rooms_created = create_rooms_for_cameras(
                location_store,
                seed_cameras,
                section_id=section.id,
                cam_records=seed_cam_records,
            )
            if rooms_created > 0:
                LOGGER.info("INIT_DB created %d room(s) for seed cameras", rooms_created)
                seeded = True
            rooms, _ = location_store.list_rooms(section_id=section.id, limit=1000)
            rooms_by_name = {item.name: item for item in rooms}
            cams_by_url = {item.url: item for item in seed_cam_records}
            for source in seed_cameras:
                room = rooms_by_name.get(source.name)
                cam = cams_by_url.get(source.source_uri)
                if room is None:
                    continue
                if cam is not None and room.cam_id != cam.id:
                    LOGGER.warning(
                        "INIT_DB skip source room assignment for '%s': cam ownership mismatch",
                        source.name,
                    )
                    continue
                if source.room_id != room.id:
                    registry.update(source.source_uri, room_id=room.id)
                    seeded = True

    # ── Seed all default shifts ─────────────────────────────────────
    shifts = _create_all_default_shifts(shift_store) if shift_store is not None else []
    shift = shifts[0] if shifts else None
    if not seeded and shift_store is not None:
        existing, _ = shift_store.list(limit=1)
        if existing:
            shift = existing[0]

    # ── Seed default personnel ──────────────────────────────────────
    if personnel_store is not None:
        dept_id = section.id if section is not None else None
        shift_id = shift.id if shift is not None else None
        created = create_default_personnel(
            personnel_store,
            department_id=dept_id,
            shift_id=shift_id,
        )
        if created > 0:
            LOGGER.info("INIT_DB created %d personnel record(s)", created)
            seeded = True

    # ── Seed sample detection logs ──────────────────────────────────
    if detection_log_store is not None and personnel_store is not None:
        personnel_records, _ = personnel_store.list(limit=1000)
        personnel_ids = [p.id for p in personnel_records]
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
