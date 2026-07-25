from __future__ import annotations

from app.core.source_registry import SourceRecord
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
