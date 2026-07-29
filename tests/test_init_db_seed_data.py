from __future__ import annotations

from pathlib import Path

import pytest

from app.database import Database
from app.core.cam_store import CamStore
from app.core.detection_log_store import DetectionLogStore
from app.core.location_store import LocationStore
from app.core.personnel_store import PersonnelStore
from app.core.shift_store import ShiftStore
from app.core.source_registry import SourceRecord, SourceRegistry
from app.core.init_db import (
    DEFAULT_PERSONNEL_SEED_DATA,
    create_default_cam_records,
    create_default_detection_logs,
    create_default_personnel,
    create_rooms_for_cameras,
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
        seed_sample_detections=True,
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
def test_init_database_does_not_seed_detection_logs_by_default(
    postgres_database: Database,
) -> None:
    personnel_store = _make_personnel_store(postgres_database)
    log_store = _make_log_store(postgres_database)

    result = init_database(
        personnel_store=personnel_store,
        detection_log_store=log_store,
    )

    assert result is True
    assert personnel_store.count() == len(DEFAULT_PERSONNEL_SEED_DATA)
    assert log_store.count_by_status().get("log_type:camera_rtsp", 0) == 0


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
        seed_sample_detections=True,
    )
    assert result1 is True

    # Second call should seed nothing
    result2 = init_database(
        personnel_store=personnel_store,
        detection_log_store=log_store,
        location_store=location_store,
        shift_store=shift_store,
        target_log_count=50,
        seed_sample_detections=True,
    )
    assert result2 is False

    # Counts unchanged
    assert personnel_store.count() == len(DEFAULT_PERSONNEL_SEED_DATA)
    by_status = log_store.count_by_status()
    assert by_status.get("log_type:camera_rtsp", 0) == 50


@pytest.mark.postgresql
def test_init_database_seeds_cams_and_links_camera_rooms(
    postgres_database: Database,
    source_registry: SourceRegistry,
) -> None:
    location_store = LocationStore(postgres_database)
    cam_store = CamStore(postgres_database)

    result = init_database(
        location_store=location_store,
        registry=source_registry,
        cam_store=cam_store,
    )

    assert result is True
    cams, total = cam_store.list(limit=100)
    assert total == 8
    assert {cam.camera_number for cam in cams} == set(range(1, 9))
    assert all(cam.width == 640 and cam.high == 640 for cam in cams)
    assert all(cam.source_type == "rtsp" for cam in cams)

    cams_by_url = {cam.url: cam for cam in cams}
    sources = source_registry.list()
    assert len(sources) == 8
    for source in sources:
        assert source.room_id is not None
        room = location_store.get_room(source.room_id)
        assert room is not None
        assert room.cam_id == cams_by_url[source.source_uri].id

    assert init_database(
        location_store=location_store,
        registry=source_registry,
        cam_store=cam_store,
    ) is False
    assert cam_store.count() == 8


@pytest.mark.postgresql
def test_camera_room_seed_backfills_unassigned_room_without_overwriting_owner(
    postgres_database: Database,
) -> None:
    location_store = LocationStore(postgres_database)
    cam_store = CamStore(postgres_database)
    building = location_store.create_building("Test Building")
    section = location_store.create_section("Test Section", building_id=building.id)
    cams, created = create_default_cam_records(cam_store, section.id)
    assert created == 8

    source = SourceRecord(source_uri=cams[0].url, name=cams[0].camera_name)
    room = location_store.create_room(source.name, section_id=section.id)
    assert create_rooms_for_cameras(
        location_store,
        [source],
        section_id=section.id,
        cam_records=cams,
    ) == 1
    assert location_store.get_room(room.id).cam_id == cams[0].id

    conflicting_source = SourceRecord(
        source_uri=cams[1].url,
        name="Conflicting room",
    )
    conflicting_room = location_store.create_room(
        conflicting_source.name,
        cam_id=cams[2].id,
    )
    assert create_rooms_for_cameras(
        location_store,
        [conflicting_source],
        section_id=section.id,
        cam_records=cams,
    ) == 0
    assert location_store.get_room(conflicting_room.id).cam_id == cams[2].id


@pytest.mark.postgresql
def test_init_database_preserves_room_assignment_for_non_seed_source(
    postgres_database: Database,
    source_registry: SourceRegistry,
) -> None:
    source = source_registry.create(
        SourceRecord(source_uri="rtsp://example.test/custom", name="Custom source")
    )
    location_store = LocationStore(postgres_database)

    cam_store = CamStore(postgres_database)
    assert init_database(
        location_store=location_store,
        registry=source_registry,
        cam_store=cam_store,
    ) is True

    assigned_source = source_registry.require(source.source_uri)
    assert assigned_source.room_id is not None
    rooms, _ = location_store.list_rooms(search=source.name, limit=10)
    assert len(rooms) == 1
    assert rooms[0].id == assigned_source.room_id
    assert rooms[0].cam_id is None
    assert cam_store.count() == 0
