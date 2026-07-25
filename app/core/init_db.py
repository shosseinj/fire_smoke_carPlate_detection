from __future__ import annotations

"""Database seeding for initial setup and development.

Provides seed data for personnel, detection logs, and foundational
records (building, section, shift, room) to bootstrap a new deployment.
All seed operations are idempotent — they run only when the target
tables are empty. Called from build_runtime() at startup.
"""

import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.detection_log_store import DetectionLogStore
from app.core.location_store import LocationStore
from app.core.personnel_store import PersonnelStore
from app.core.shift_store import ShiftStore

LOGGER = logging.getLogger(__name__)

# Personnel seed data: (fname, lname, national_code, employee_type, degree)
# All national codes are valid Iranian codes (pass validate_national_code).
DEFAULT_PERSONNEL_SEED_DATA: list[tuple[str, str, str, str, str]] = [
    ("كارمند", "نمونه", "0311344119", "employee", "ليسانس"),
    ("كاربر", "آزمايشي", "0410500666", "employee", "فوق ديپلم"),
    ("پرسنل", "سوم", "0010691782", "employee", "ديپلم"),
]


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
    for fname, lname, national_code, employee_type, degree in DEFAULT_PERSONNEL_SEED_DATA:
        try:
            personnel_store.create(
                fname=fname,
                lname=lname,
                national_code=national_code,
                employee_type=employee_type,
                degree=degree,
                shift_id=shift_id,
                department_id=department_id,
            )
            count += 1
        except ValueError as exc:
            LOGGER.warning("INIT_DB skip personnel %s %s: %s", fname, lname, exc)
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


def init_database(
    *,
    personnel_store: PersonnelStore | None = None,
    detection_log_store: DetectionLogStore | None = None,
    location_store: LocationStore | None = None,
    shift_store: ShiftStore | None = None,
    target_log_count: int = 100,
) -> bool:
    """Seed the database with foundational records and sample data.

    Creates a seed building → section → room hierarchy, a default
    work shift, default personnel from
    :data:`DEFAULT_PERSONNEL_SEED_DATA`, and sample detection logs.
    Every operation is idempotent.

    Call this function from ``build_runtime()`` (or the startup
    entrypoint) after all stores have been constructed.

    Parameters
    ----------
    personnel_store, detection_log_store, location_store, shift_store:
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
    if location_store is not None:
        building_records, _ = location_store.list_buildings(limit=1)
        if not building_records:
            building = location_store.create_building(
                name="ساختمان مرکزي",
                address="Tehran",
                description="Initial seed building",
            )
            LOGGER.info("INIT_DB created building id=%d", building.id)
            seeded = True
        else:
            building = building_records[0]

        section_records, _ = location_store.list_sections(limit=1)
        if not section_records:
            section = location_store.create_section(
                name="دپارتمان فناوري اطلاعات",
                building_id=building.id,
                description="Initial seed section",
            )
            LOGGER.info("INIT_DB created section id=%d", section.id)
            seeded = True
        else:
            section = section_records[0]

        room_records, _ = location_store.list_rooms(limit=1)
        if not room_records:
            room = location_store.create_room(
                name="اتاق سرور",
                section_id=section.id,
                description="Initial seed room",
            )
            LOGGER.info("INIT_DB created room id=%d", room.id)
            seeded = True
        else:
            room = room_records[0]

    # ── Seed default shift ──────────────────────────────────────────
    shift = None
    if shift_store is not None:
        shift_records, _ = shift_store.list(limit=1)
        if not shift_records:
            try:
                shift = shift_store.create(
                    shift_name="شيفت اداري",
                    shift_type="morning",
                    start_time="08:00",
                    end_time="16:00",
                    timezone_name="Asia/Tehran",
                )
                LOGGER.info("INIT_DB created shift id=%d", shift.id)
                seeded = True
            except Exception as exc:
                LOGGER.warning("INIT_DB skip shift creation: %s", exc)
        else:
            shift = shift_records[0]

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
