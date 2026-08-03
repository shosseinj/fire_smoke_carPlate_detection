from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import pytest
from sqlalchemy.schema import Computed

from app.core.location_store import LocationStore
from app.core.detection_log_store import DetectionLogStore
from app.core.plate_log_store import PlateLogStore
from app.database import Connection, Cursor, Row, metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _DriverResult:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class _DriverConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def exec_driver_sql(self, statement: str, params: object) -> _DriverResult:
        self.calls.append((statement, params))
        return _DriverResult(len(params))


def _connection_with_driver(driver: _DriverConnection) -> Connection:
    connection = object.__new__(Connection)
    connection._conn = driver
    connection._closed = False
    return connection


def test_executemany_uses_one_driver_call() -> None:
    driver = _DriverConnection()
    connection = _connection_with_driver(driver)

    cursor = connection.executemany(
        "INSERT INTO plate_logs (plate_number) VALUES (?)",
        (("11الف11111",), ("22ب22222",), ("33ج33333",)),
    )

    assert cursor.rowcount == 3
    assert len(driver.calls) == 1
    statement, params = driver.calls[0]
    assert statement.endswith("VALUES (%s)")
    assert params == [
        ("11الف11111",),
        ("22ب22222",),
        ("33ج33333",),
    ]


def test_executemany_empty_input_skips_database() -> None:
    driver = _DriverConnection()
    connection = _connection_with_driver(driver)

    assert connection.executemany("INSERT INTO x (value) VALUES (?)", []).rowcount == 0
    assert driver.calls == []


def test_executemany_rejects_returning() -> None:
    driver = _DriverConnection()
    connection = _connection_with_driver(driver)

    with pytest.raises(ValueError, match="does not support RETURNING"):
        connection.executemany(
            "INSERT INTO x (value) VALUES (?) RETURNING id", [(1,)]
        )


def test_query_indexes_and_generated_plate_are_in_metadata() -> None:
    expected_indexes = {
        "idx_detection_logs_attendance_personnel_time",
        "idx_detection_logs_attendance_person_time",
        "idx_detection_logs_source_human",
        "idx_detection_logs_personnel_room_time",
        "idx_matches_camera_track_room_time",
        "idx_human_logs_attendance_personnel_last_seen",
        "idx_requests_person_status_dates",
        "idx_personnel_images_personnel_order",
        "idx_car_plates_active_normalized",
    }
    actual_indexes = {
        index.name
        for table in metadata.tables.values()
        for index in table.indexes
    }
    assert expected_indexes <= actual_indexes
    assert isinstance(metadata.tables["car_plates"].c.normalized_plate.computed, Computed)


class _SingleRowResult:
    def __init__(self, row: Row | None = None, rows: list[Row] | None = None) -> None:
        self._row = row
        self._rows = rows or ([] if row is None else [row])

    def fetchone(self) -> Row | None:
        return self._row

    def fetchall(self) -> list[Row]:
        return list(self._rows)


class _ContextConnection:
    def __init__(self, execute_callback) -> None:
        self.execute_callback = execute_callback

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, sql, params=()):
        return self.execute_callback(sql, params)


def test_room_polygon_cache_avoids_repeated_query_and_invalidates() -> None:
    store = object.__new__(LocationStore)
    store._lock = threading.RLock()
    store._room_polygon_cache = OrderedDict()
    store._camera_polygon_cache = OrderedDict()
    calls: list[str] = []

    def execute(sql, params):
        calls.append(sql)
        return _SingleRowResult(
            Row(["polygon_json"], ['[[0,0],[100,0],[100,100],[0,100]]'])
        )

    store._connection = lambda: _ContextConnection(execute)

    assert store.get_polygon_for_room(7)[0] == [0, 0]
    assert store.get_polygon_for_room(7)[2] == [100, 100]
    assert len(calls) == 1

    store._invalidate_polygon_cache()
    store.get_polygon_for_room(7)
    assert len(calls) == 2


def test_plate_registration_lookup_is_batched() -> None:
    store = object.__new__(PlateLogStore)
    calls: list[tuple[str, object]] = []

    def execute(sql, params):
        calls.append((sql, params))
        return _SingleRowResult(
            rows=[
                Row(["id", "normalized_plate"], [5, "11الف11111"]),
                Row(["id", "normalized_plate"], [8, "22ب22222"]),
            ]
        )

    store._connect = lambda: _ContextConnection(execute)
    result = store._registered_plate_ids({"22ب22222", "11الف11111"})

    assert result == {"11الف11111": 5, "22ب22222": 8}
    assert len(calls) == 1
    assert "normalized_plate IN" in calls[0][0]


def test_detection_create_many_uses_one_multirow_statement() -> None:
    store = object.__new__(DetectionLogStore)
    store._lock = threading.RLock()
    calls: list[tuple[str, object]] = []
    notifications: list[tuple[str, int]] = []

    def execute(sql, params):
        calls.append((sql, params))
        return _SingleRowResult(
            rows=[Row(["id"], [11]), Row(["id"], [12])]
        )

    store._connection = lambda: _ContextConnection(execute)
    store._row_to_log = lambda row: int(row["id"])
    store._notify = lambda action, record: notifications.append((action, record))

    result = store.create_many(
        [
            {"person": "A", "detection_time": "2026-08-02T08:00:00+00:00"},
            {"person": "B", "detection_time": "2026-08-02T08:01:00+00:00"},
        ]
    )

    assert result == [11, 12]
    assert notifications == [("created", 11), ("created", 12)]
    assert len(calls) == 1
    assert calls[0][0].count("RETURNING *") == 1
    assert calls[0][0].count("(") >= 2


def test_bulk_access_resolution_uses_bounded_queries() -> None:
    store = object.__new__(LocationStore)
    store._lock = threading.RLock()
    calls: list[tuple[str, object]] = []

    def execute(sql, params):
        calls.append((sql, params))
        if "SELECT id, name FROM rooms" in sql:
            return _SingleRowResult(
                rows=[
                    Row(["id", "name"], [7, "General Hall"]),
                    Row(["id", "name"], [8, "Restricted"]),
                ]
            )
        return _SingleRowResult(
            rows=[Row(["personnel_id", "room_id"], [2, 8])]
        )

    store._connection = lambda: _ContextConnection(execute)
    result = store.resolve_access_for_pairs({(1, 7), (1, 8), (2, 8)})

    assert result == {(1, 7): True, (1, 8): False, (2, 8): True}
    assert len(calls) == 2


def test_hot_path_sources_use_set_based_queries() -> None:
    human_source = (PROJECT_ROOT / "app/core/human_log_store.py").read_text()
    attendance_source = (
        PROJECT_ROOT / "app/core/attendance_summary_service.py"
    ).read_text()
    location_source = (PROJECT_ROOT / "app/core/location_store.py").read_text()
    personnel_source = (PROJECT_ROOT / "app/core/personnel_store.py").read_text()
    detection_source = (
        PROJECT_ROOT / "app/core/detection_log_store.py"
    ).read_text()

    assert "ON CONFLICT (session_id, camera, track_id) DO UPDATE" in human_source
    assert "RETURNING id, snapshot_url, video_url, face_video_url" in human_source
    assert "d.personnel_id IN" in attendance_source
    assert "d.person IN" in attendance_source
    assert "AND (" + "{identity_sql}" not in attendance_source
    assert "RETURNING id, detection_type" in location_source
    assert "WHERE personnel_id IN" in personnel_source
    assert "WITH incoming(candidate_index" in detection_source
    assert "INSERT INTO detection_logs" in detection_source
    assert "RETURNING *" in detection_source
