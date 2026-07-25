from __future__ import annotations

from pathlib import Path

import pytest

from app.database import Database
from app.core.detection_log_store import DetectionLogStore
from app.core.location_store import LocationStore
from app.core.personnel_store import PersonnelStore
from app.core.shift_store import ShiftStore
from app.core.init_db import (
    DEFAULT_PERSONNEL_SEED_DATA,
    create_default_detection_logs,
    create_default_personnel,
    init_database,
)


def _make_personnel_store(db: Database) -> PersonnelStore:
    return PersonnelStore(db, saved_media_path=Path("C:\\temp\\test_media"))


def _make_log_store(db: Database) -> DetectionLogStore:
    return DetectionLogStore(db)


@pytest.mark.postgresql
def test_personnel_table_metadata_registered() -> None:
    from app.database import metadata
    assert "personnel" in metadata.tables


@pytest.mark.postgresql
def test_detection_logs_table_metadata_registered() -> None:
    from app.database import metadata
    assert "detection_logs" in metadata.tables


@pytest.mark.postgresql
def test_create_default_personnel_seeds_when_empty(
    postgres_database: Database,
) -> None:
    personnel_store = _make_personnel_store(postgres_database)
    created = create_default_personnel(personnel_store)
    assert created == len(DEFAULT_PERSONNEL_SEED_DATA)
    assert personnel_store.count() == len(DEFAULT_PERSONNEL_SEED_DATA)

    records, _ = personnel_store.list(limit=100)
    national_codes = {r.national_code for r in records}
    assert "0311344119" in national_codes
    assert "0410500666" in national_codes
    assert "0010691782" in national_codes
    assert all(len(r.national_code) == 10 for r in records)
    assert all(r.employee_type == "employee" for r in records)

    # Idempotent: second call returns 0
    created2 = create_default_personnel(personnel_store)
    assert created2 == 0
    assert personnel_store.count() == len(DEFAULT_PERSONNEL_SEED_DATA)


@pytest.mark.postgresql
def test_create_default_detection_logs_seeds_when_empty(
    postgres_database: Database,
) -> None:
    personnel_store = _make_personnel_store(postgres_database)
    log_store = _make_log_store(postgres_database)
    location_store = LocationStore(postgres_database)

    # Seed dependencies: building → section → room, shift, personnel
    building = location_store.create_building("Test Building")
    section = location_store.create_section("Test Section", building_id=building.id)
    room = location_store.create_room("Test Room", section_id=section.id)
    shift_store = ShiftStore(postgres_database)
    shift = shift_store.create(
        shift_name="Test Shift",
        shift_type="morning",
        start_time="08:00",
        end_time="16:00",
        timezone_name="Asia/Tehran",
    )
    personnel_store.create(
        fname="Test",
        lname="User",
        national_code="0311344119",
        employee_type="employee",
        shift_id=shift.id,
        department_id=section.id,
    )
    personnel_ids = [p.id for p in personnel_store.list(limit=100)[0]]

    created = create_default_detection_logs(
        log_store,
        personnel_ids=personnel_ids,
        room_id=room.id,
        target_count=100,
    )
    assert created == 100

    by_status = log_store.count_by_status()
    # count_by_status returns per-log_type and per-access_granted entries
    # The log_type:camera_rtsp count represents the total number of logs
    assert by_status.get("log_type:camera_rtsp", 0) == 100

    # At least some logs have access_granted True and some False
    assert "access_granted:True" in by_status
    assert "access_granted:False" in by_status

    # Idempotent: second call returns 0
    created2 = create_default_detection_logs(
        log_store,
        personnel_ids=personnel_ids,
        room_id=room.id,
        target_count=100,
    )
    assert created2 == 0


@pytest.mark.postgresql
def test_create_default_detection_logs_skips_when_no_personnel(
    postgres_database: Database,
) -> None:
    log_store = _make_log_store(postgres_database)
    created = create_default_detection_logs(
        log_store,
        personnel_ids=[],
        target_count=100,
    )
    assert created == 0


@pytest.mark.postgresql
def test_init_database_seeds_everything_when_empty(
    postgres_database: Database,
) -> None:
    personnel_store = _make_personnel_store(postgres_database)
    log_store = _make_log_store(postgres_database)
    location_store = LocationStore(postgres_database)
    shift_store = ShiftStore(postgres_database)

    result = init_database(
        personnel_store=personnel_store,
        detection_log_store=log_store,
        location_store=location_store,
        shift_store=shift_store,
        target_log_count=50,
    )
    assert result is True

    # Building, section, room created
    assert location_store.count_buildings() >= 1
    assert location_store.count_sections() >= 1
    assert location_store.count_rooms() >= 1

    # Shift created
    assert shift_store.count() >= 1

    # Personnel created
    assert personnel_store.count() == len(DEFAULT_PERSONNEL_SEED_DATA)

    # Detection logs created
    by_status = log_store.count_by_status()
    assert by_status.get("log_type:camera_rtsp", 0) == 50


@pytest.mark.postgresql
def test_init_database_is_idempotent(
    postgres_database: Database,
) -> None:
    personnel_store = _make_personnel_store(postgres_database)
    log_store = _make_log_store(postgres_database)
    location_store = LocationStore(postgres_database)
    shift_store = ShiftStore(postgres_database)

    # First call seeds everything
    result1 = init_database(
        personnel_store=personnel_store,
        detection_log_store=log_store,
        location_store=location_store,
        shift_store=shift_store,
        target_log_count=50,
    )
    assert result1 is True

    # Second call should seed nothing
    result2 = init_database(
        personnel_store=personnel_store,
        detection_log_store=log_store,
        location_store=location_store,
        shift_store=shift_store,
        target_log_count=50,
    )
    assert result2 is False

    # Counts unchanged
    assert personnel_store.count() == len(DEFAULT_PERSONNEL_SEED_DATA)
    by_status = log_store.count_by_status()
    assert by_status.get("log_type:camera_rtsp", 0) == 50
