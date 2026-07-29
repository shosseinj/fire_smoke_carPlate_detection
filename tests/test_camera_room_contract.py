from __future__ import annotations

from types import SimpleNamespace

from app.api.locations import _assign_created_room_to_source
from app.core.source_registry import SourceRecord
from app.runtime import Runtime
from app.schemas import CameraCreate, CameraUpdate, SourceCreate


def test_source_record_room_id_round_trip() -> None:
    record = SourceRecord.from_dict(
        {"source_uri": "camera://one", "name": "One", "room_id": 7}
    )
    assert record.room_id == 7
    assert record.to_dict()["room_id"] == 7


def test_camera_and_source_contracts_accept_room_id() -> None:
    assert CameraCreate(source_uri="camera://one", name="One", room_id=3).room_id == 3
    assert SourceCreate(source_uri="camera://two", name="Two", room_id=4).room_id == 4


def test_camera_update_can_explicitly_unassign_room() -> None:
    update = CameraUpdate.model_validate({"room_id": None})
    assert update.model_dump(exclude_unset=True) == {"room_id": None}


def test_created_camera_room_assigns_and_refreshes_source() -> None:
    updates: list[tuple[str, int]] = []
    refreshes: list[bool] = []
    runtime = SimpleNamespace(
        cam_store=SimpleNamespace(
            get=lambda camera_id: SimpleNamespace(id=camera_id, url="camera://one")
        ),
        registry=SimpleNamespace(
            get=lambda source_uri: SimpleNamespace(source_uri=source_uri),
            update=lambda source_uri, **changes: updates.append(
                (source_uri, changes["room_id"])
            ),
        ),
        _refresh_all_source_zones=lambda: refreshes.append(True),
    )

    _assign_created_room_to_source(runtime, room_id=9, camera_id=3)

    assert updates == [("camera://one", 9)]
    assert refreshes == [True]


def test_startup_repairs_unassigned_source_from_newest_active_camera_room() -> None:
    updates: list[tuple[str, int]] = []
    runtime = SimpleNamespace(
        location_store=SimpleNamespace(
            list_rooms=lambda **_kwargs: (
                [
                    SimpleNamespace(id=12, cam_id=3, is_active=True),
                    SimpleNamespace(id=11, cam_id=3, is_active=True),
                ],
                2,
            )
        ),
        cam_store=SimpleNamespace(
            list=lambda **_kwargs: (
                [SimpleNamespace(id=3, url="camera://one")],
                1,
            )
        ),
        registry=SimpleNamespace(
            get=lambda _source_uri: SimpleNamespace(room_id=None),
            update=lambda source_uri, **changes: updates.append(
                (source_uri, changes["room_id"])
            ),
        ),
    )

    count = Runtime._synchronize_source_room_assignments(runtime)

    assert count == 1
    assert updates == [("camera://one", 12)]
