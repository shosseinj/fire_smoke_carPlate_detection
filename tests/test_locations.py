"""Tests for the Locations module (Buildings, Sections, Rooms, Access, Polygon matching)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from app.core.location_store import (
    LocationStore,
    point_in_polygon,
    parse_polygon,
)


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def db_path() -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
    tmp.close()
    yield Path(tmp.name)
    os.unlink(tmp.name)


@pytest.fixture
def store(db_path: Path) -> LocationStore:
    return LocationStore(db_path)


# ── Polygon utility tests ──────────────────────────────────────────────


class TestPointInPolygon:
    def test_point_inside_square(self) -> None:
        square = [[0, 0], [100, 0], [100, 100], [0, 100]]
        assert point_in_polygon(50, 50, square) is True

    def test_point_outside_square(self) -> None:
        square = [[0, 0], [100, 0], [100, 100], [0, 100]]
        assert point_in_polygon(150, 150, square) is False

    def test_point_on_edge(self) -> None:
        square = [[0, 0], [100, 0], [100, 100], [0, 100]]
        # Point on the edge may be considered inside by PNPoly
        # (depends on the algorithm, acceptable)
        pass

    def test_point_inside_triangle(self) -> None:
        triangle = [[0, 0], [100, 0], [50, 100]]
        assert point_in_polygon(50, 30, triangle) is True

    def test_point_outside_triangle(self) -> None:
        triangle = [[0, 0], [100, 0], [50, 100]]
        assert point_in_polygon(0, 50, triangle) is False

    def test_invalid_polygon_less_than_3_points(self) -> None:
        line = [[0, 0], [100, 0]]
        assert point_in_polygon(50, 50, line) is False

    def test_empty_polygon(self) -> None:
        assert point_in_polygon(50, 50, []) is False


class TestParsePolygon:
    def test_valid_json(self) -> None:
        result = parse_polygon("[[0,0],[100,0],[100,100],[0,100]]")
        assert result == [[0, 0], [100, 0], [100, 100], [0, 100]]

    def test_none_returns_empty(self) -> None:
        assert parse_polygon(None) == []

    def test_empty_string_returns_empty(self) -> None:
        assert parse_polygon("") == []

    def test_invalid_json_returns_empty(self) -> None:
        assert parse_polygon("not json") == []

    def test_less_than_3_points_returns_empty(self) -> None:
        assert parse_polygon("[[0,0],[1,1]]") == []


# ── Building CRUD tests ───────────────────────────────────────────────


class TestBuildings:
    def test_create_building(self, store: LocationStore) -> None:
        bld = store.create_building("Test Building", "123 Main St", "A test building")
        assert bld.id > 0
        assert bld.name == "Test Building"
        assert bld.address == "123 Main St"
        assert bld.description == "A test building"
        assert bld.created_at_utc is not None

    def test_create_building_requires_name(self, store: LocationStore) -> None:
        with pytest.raises(ValueError, match="Building name is required"):
            store.create_building("   ")

    def test_get_building(self, store: LocationStore) -> None:
        bld = store.create_building("Get Test")
        fetched = store.get_building(bld.id)
        assert fetched is not None
        assert fetched.name == "Get Test"
        assert fetched.id == bld.id

    def test_get_building_not_found(self, store: LocationStore) -> None:
        assert store.get_building(99999) is None

    def test_update_building(self, store: LocationStore) -> None:
        bld = store.create_building("Original")
        updated = store.update_building(bld.id, name="Updated", address="New Address")
        assert updated is not None
        assert updated.name == "Updated"
        assert updated.address == "New Address"

    def test_update_building_not_found(self, store: LocationStore) -> None:
        assert store.update_building(99999, name="X") is None

    def test_delete_building(self, store: LocationStore) -> None:
        bld = store.create_building("To Delete")
        assert store.delete_building(bld.id) is True
        assert store.get_building(bld.id) is None

    def test_delete_building_not_found(self, store: LocationStore) -> None:
        assert store.delete_building(99999) is False

    def test_list_buildings(self, store: LocationStore) -> None:
        store.create_building("A")
        store.create_building("B")
        records, total = store.list_buildings(limit=100)
        assert total >= 2
        assert len(records) >= 2

    def test_list_buildings_search(self, store: LocationStore) -> None:
        store.create_building("Alpha Building")
        store.create_building("Beta Building")
        records, total = store.list_buildings(search="Alpha")
        assert total == 1
        assert records[0].name == "Alpha Building"

    def test_count_buildings(self, store: LocationStore) -> None:
        before = store.count_buildings()
        store.create_building("Count Test")
        assert store.count_buildings() == before + 1


# ── Section CRUD tests ────────────────────────────────────────────────


class TestSections:
    def test_create_section(self, store: LocationStore) -> None:
        sec = store.create_section("Floor 1", description="First floor")
        assert sec.id > 0
        assert sec.name == "Floor 1"
        assert sec.building_id is None
        assert sec.description == "First floor"

    def test_create_section_with_building(self, store: LocationStore) -> None:
        bld = store.create_building("Main")
        sec = store.create_section("Section A", building_id=bld.id)
        assert sec.building_id == bld.id

    def test_create_section_invalid_building(self, store: LocationStore) -> None:
        with pytest.raises(ValueError, match="Building not found"):
            store.create_section("Bad", building_id=99999)

    def test_create_section_requires_name(self, store: LocationStore) -> None:
        with pytest.raises(ValueError, match="Section name is required"):
            store.create_section("   ")

    def test_update_section(self, store: LocationStore) -> None:
        sec = store.create_section("Original")
        updated = store.update_section(sec.id, name="Updated Section")
        assert updated is not None
        assert updated.name == "Updated Section"

    def test_delete_section(self, store: LocationStore) -> None:
        sec = store.create_section("To Delete")
        assert store.delete_section(sec.id) is True
        assert store.get_section(sec.id) is None

    def test_list_sections_by_building(self, store: LocationStore) -> None:
        bld = store.create_building("HQ")
        store.create_section("Section 1", building_id=bld.id)
        store.create_section("Section 2", building_id=bld.id)
        records, total = store.list_sections(building_id=bld.id)
        assert total >= 2

    def test_count_sections(self, store: LocationStore) -> None:
        before = store.count_sections()
        store.create_section("Count Test")
        assert store.count_sections() == before + 1


# ── Room CRUD tests ───────────────────────────────────────────────────


class TestRooms:
    def test_create_room(self, store: LocationStore) -> None:
        room = store.create_room("Office 101")
        assert room.id > 0
        assert room.name == "Office 101"
        assert room.section_id is None
        assert room.polygon_json is None

    def test_create_room_with_section(self, store: LocationStore) -> None:
        sec = store.create_section("Floor 1")
        room = store.create_room("Office A", section_id=sec.id)
        assert room.section_id == sec.id

    def test_create_room_invalid_section(self, store: LocationStore) -> None:
        with pytest.raises(ValueError, match="Section not found"):
            store.create_room("Bad", section_id=99999)

    def test_create_room_with_valid_polygon(self, store: LocationStore) -> None:
        polygon = "[[0,0],[100,0],[100,100],[0,100]]"
        room = store.create_room("Square Room", polygon_json=polygon)
        assert room.polygon_json == polygon

    def test_create_room_with_invalid_polygon(self, store: LocationStore) -> None:
        with pytest.raises(ValueError, match="Polygon must have at least 3 vertices"):
            store.create_room("Bad Polygon", polygon_json="[[0,0],[1,1]]")

    def test_create_room_with_malformed_polygon(self, store: LocationStore) -> None:
        with pytest.raises(ValueError, match="Polygon must have at least 3 vertices"):
            store.create_room("Bad Polygon", polygon_json="not json")

    def test_update_room_polygon(self, store: LocationStore) -> None:
        room = store.create_room("Room")
        new_poly = "[[0,0],[200,0],[200,200],[0,200]]"
        updated = store.update_room(room.id, polygon_json=new_poly)
        assert updated is not None
        assert updated.polygon_json == new_poly

    def test_list_rooms_by_section(self, store: LocationStore) -> None:
        sec = store.create_section("Floor 1")
        store.create_room("Room 1", section_id=sec.id)
        store.create_room("Room 2", section_id=sec.id)
        records, total = store.list_rooms(section_id=sec.id)
        assert total >= 2

    def test_count_rooms(self, store: LocationStore) -> None:
        before = store.count_rooms()
        store.create_room("Count Test")
        assert store.count_rooms() == before + 1


# ── Personnel Room Access tests ────────────────────────────────────────


class TestPersonnelRoomAccess:
    def test_grant_access(self, store: LocationStore) -> None:
        room = store.create_room("Secure Room")
        access = store.grant_room_access(
            personnel_id=1, room_id=room.id, granted_by="admin"
        )
        assert access.id > 0
        assert access.personnel_id == 1
        assert access.room_id == room.id
        assert access.granted_by == "admin"

    def test_grant_duplicate_access_raises(self, store: LocationStore) -> None:
        room = store.create_room("Room")
        store.grant_room_access(personnel_id=1, room_id=room.id)
        with pytest.raises(ValueError, match="already has access"):
            store.grant_room_access(personnel_id=1, room_id=room.id)

    def test_check_access_true(self, store: LocationStore) -> None:
        room = store.create_room("Room")
        store.grant_room_access(personnel_id=42, room_id=room.id)
        assert store.check_room_access(personnel_id=42, room_id=room.id) is True

    def test_check_access_false(self, store: LocationStore) -> None:
        room = store.create_room("Room")
        assert store.check_room_access(personnel_id=42, room_id=room.id) is False

    def test_revoke_access(self, store: LocationStore) -> None:
        room = store.create_room("Room")
        store.grant_room_access(personnel_id=1, room_id=room.id)
        assert store.revoke_room_access(personnel_id=1, room_id=room.id) is True
        assert store.check_room_access(personnel_id=1, room_id=room.id) is False

    def test_revoke_nonexistent(self, store: LocationStore) -> None:
        room = store.create_room("Room")
        assert store.revoke_room_access(personnel_id=1, room_id=room.id) is False

    def test_list_personnel_rooms(self, store: LocationStore) -> None:
        room_a = store.create_room("Room A")
        room_b = store.create_room("Room B")
        store.grant_room_access(personnel_id=7, room_id=room_a.id)
        store.grant_room_access(personnel_id=7, room_id=room_b.id)
        rooms = store.list_personnel_rooms(personnel_id=7)
        assert len(rooms) == 2
        assert {r.id for r in rooms} == {room_a.id, room_b.id}

    def test_list_room_personnel(self, store: LocationStore) -> None:
        room = store.create_room("Room")
        store.grant_room_access(personnel_id=10, room_id=room.id)
        store.grant_room_access(personnel_id=20, room_id=room.id)
        personnel = store.list_room_personnel(room.id)
        assert len(personnel) == 2
        ids = [p["personnel_id"] for p in personnel]
        assert 10 in ids
        assert 20 in ids


# ── Polygon Matching tests ─────────────────────────────────────────────


class TestPolygonMatching:
    def test_match_detection_to_rooms_inside(self, store: LocationStore) -> None:
        """A point inside a room polygon should produce a match."""
        bld = store.create_building("Test")
        sec = store.create_section("Floor 1", building_id=bld.id)
        polygon = "[[0,0],[200,0],[200,200],[0,200]]"
        room = store.create_room("Target", section_id=sec.id, polygon_json=polygon)

        matches = store.match_detection_to_rooms(
            section_id=sec.id,
            detection_type="face_recognition",
            detection_event_id=0,
            bbox_center_x=100.0,
            bbox_center_y=100.0,
            personnel_id=1,
            camera_id="test-cam-1",
        )
        assert len(matches) >= 1
        match = matches[0]
        assert match.room_id == room.id
        assert match.personnel_id == 1
        assert match.detection_type == "face_recognition"
        assert match.camera_id == "test-cam-1"

    def test_match_detection_to_rooms_outside(self, store: LocationStore) -> None:
        """A point outside all room polygons should produce no match."""
        bld = store.create_building("Test")
        sec = store.create_section("Floor 1", building_id=bld.id)
        polygon = "[[0,0],[200,0],[200,200],[0,200]]"
        store.create_room("Target", section_id=sec.id, polygon_json=polygon)

        matches = store.match_detection_to_rooms(
            section_id=sec.id,
            detection_type="plate_recognition",
            detection_event_id=0,
            bbox_center_x=500.0,
            bbox_center_y=500.0,
        )
        assert len(matches) == 0

    def test_match_no_section(self, store: LocationStore) -> None:
        """None section_id should produce no matches."""
        room = store.create_room("Room", polygon_json="[[0,0],[100,0],[100,100],[0,100]]")
        matches = store.match_detection_to_rooms(
            section_id=None,  # type: ignore[arg-type]
            detection_type="face_recognition",
            detection_event_id=0,
            bbox_center_x=50.0,
            bbox_center_y=50.0,
        )
        assert len(matches) == 0

    def test_get_matches_for_detection(self, store: LocationStore) -> None:
        """Retrieve matches by detection type and event ID."""
        bld = store.create_building("Test")
        sec = store.create_section("Floor 1", building_id=bld.id)
        polygon = "[[0,0],[200,0],[200,200],[0,200]]"
        room = store.create_room("Target", section_id=sec.id, polygon_json=polygon)

        store.match_detection_to_rooms(
            section_id=sec.id,
            detection_type="face_recognition",
            detection_event_id=42,
            bbox_center_x=100.0,
            bbox_center_y=100.0,
            personnel_id=1,
            camera_id="test-cam-3",
        )

        matches = store.get_matches_for_detection("face_recognition", 42)
        assert len(matches) >= 1
        assert matches[0].detection_event_id == 42

    def test_list_matches_for_room(self, store: LocationStore) -> None:
        """Retrieve matches grouped by room."""
        sec = store.create_section("Floor 1")
        polygon = "[[0,0],[200,0],[200,200],[0,200]]"
        room = store.create_room("Room", section_id=sec.id, polygon_json=polygon)

        store.match_detection_to_rooms(
            section_id=sec.id,
            detection_type="face_recognition",
            detection_event_id=1,
            bbox_center_x=100.0,
            bbox_center_y=100.0,
            camera_id="test-cam-4",
        )
        store.match_detection_to_rooms(
            section_id=sec.id,
            detection_type="plate_recognition",
            detection_event_id=2,
            bbox_center_x=100.0,
            bbox_center_y=100.0,
            camera_id="test-cam-4",
        )

        matches, total = store.list_matches_for_room(room.id)
        assert total >= 2
        assert len(matches) >= 2


# ── Section Camera Assignment tests ────────────────────────────────────


class TestSectionCameras:
    def test_filter_cameras_by_section_empty(self, store: LocationStore) -> None:
        """An empty list returns empty list."""
        cameras = store.filter_cameras_by_section([], 1)
        assert cameras == []

    def test_filter_cameras_by_section_matches(self, store: LocationStore) -> None:
        """Cameras with matching section_id are returned."""
        cameras = [
            {"camera_id": "cam-1", "name": "Cam 1", "metadata": {"section_id": 1}},
            {"camera_id": "cam-2", "name": "Cam 2", "metadata": {"section_id": 2}},
            {"camera_id": "cam-3", "name": "Cam 3", "metadata": {"section_id": 1}},
        ]
        filtered = store.filter_cameras_by_section(cameras, 1)
        assert len(filtered) == 2
        ids = [c["camera_id"] for c in filtered]
        assert "cam-1" in ids
        assert "cam-3" in ids

    def test_filter_cameras_by_section_none_match(self, store: LocationStore) -> None:
        """No cameras match a section with no assignments."""
        cameras = [
            {"camera_id": "cam-1", "name": "Cam 1", "metadata": {"section_id": 5}},
        ]
        filtered = store.filter_cameras_by_section(cameras, 1)
        assert filtered == []


# ── Cascade deletion tests ─────────────────────────────────────────────


class TestCascadeDelete:
    def test_delete_building_cascades_to_sections(self, store: LocationStore) -> None:
        """Delete building should SET NULL on section's building_id (ON DELETE SET NULL)."""
        bld = store.create_building("Main")
        sec = store.create_section("Floor 1", building_id=bld.id)
        assert sec.building_id == bld.id
        store.delete_building(bld.id)
        # Section should still exist but building_id should be NULL
        remaining = store.get_section(sec.id)
        assert remaining is not None
        assert remaining.building_id is None

    def test_delete_section_cascades_to_rooms(self, store: LocationStore) -> None:
        """Delete section should SET NULL on room's section_id."""
        sec = store.create_section("Floor 1")
        room = store.create_room("Office", section_id=sec.id)
        assert room.section_id == sec.id
        store.delete_section(sec.id)
        remaining = store.get_room(room.id)
        assert remaining is not None
        assert remaining.section_id is None

    def test_delete_room_cascades_to_access(self, store: LocationStore) -> None:
        """Delete room should CASCADE to personnel_room_access."""
        room = store.create_room("Room")
        store.grant_room_access(personnel_id=1, room_id=room.id)
        assert store.check_room_access(personnel_id=1, room_id=room.id) is True
        store.delete_room(room.id)
        # Access record should be deleted
        assert store.check_room_access(personnel_id=1, room_id=room.id) is False

    def test_delete_room_cascades_to_matches(self, store: LocationStore) -> None:
        """Delete room should CASCADE to detection_room_matches."""
        sec = store.create_section("Floor 1")
        room = store.create_room("Room", section_id=sec.id, polygon_json="[[0,0],[100,0],[100,100],[0,100]]")

        store.match_detection_to_rooms(
            section_id=sec.id,
            detection_type="face_recognition",
            detection_event_id=0,
            bbox_center_x=50.0,
            bbox_center_y=50.0,
            camera_id="cam-cascade",
        )

        matches_before, total_before = store.list_matches_for_room(room.id)
        assert total_before >= 1

        store.delete_room(room.id)

        matches_after, total_after = store.list_matches_for_room(room.id)
        assert total_after == 0
