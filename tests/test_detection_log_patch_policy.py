from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.api.detection_logs import (
    DetectionLogResponse,
    DetectionLogUpdate,
    _calculate_old_person_access,
    _get_current_user_id,
)


def test_detection_log_response_omits_redundant_name_and_media_fields() -> None:
    payload = DetectionLogResponse.model_validate({
        "id": 1,
        "fname": "Ali",
        "lname": "Ahmadi",
        "snapshot_thumbnail": "/media/snapshot.jpg",
        "face_video_url": "/media/face.mp4",
    }).model_dump()

    assert "fname" not in payload
    assert "lname" not in payload
    assert "snapshot_thumbnail" not in payload
    assert "face_video_url" not in payload


def test_person_patch_requires_identity_and_trims_person() -> None:
    update = DetectionLogUpdate(person=" 0311344119 ")
    assert update.person == "0311344119"

    with pytest.raises(ValidationError):
        DetectionLogUpdate(person="   ")

    with pytest.raises(ValidationError):
        DetectionLogUpdate()

    with pytest.raises(ValidationError):
        DetectionLogUpdate(detection_time="   ")


def test_person_patch_accepts_jalali_detection_time_only() -> None:
    update = DetectionLogUpdate(detection_time="1404-05-17 08:30:00")
    assert update.person is None
    assert update.detection_time == "1404-05-17 08:30:00"

    combined = DetectionLogUpdate(
        person=" 0311344119 ", detection_time="۱۴۰۴/۰۵/۱۷ ۰۸:۳۰"
    )
    assert combined.person == "0311344119"
    assert combined.detection_time == "۱۴۰۴/۰۵/۱۷ ۰۸:۳۰"


def test_person_patch_reads_user_id_from_dict_or_user_record() -> None:
    assert _get_current_user_id({"id": 7}) == 7
    assert _get_current_user_id(type("UserRecord", (), {"id": 8})()) == 8


@pytest.mark.parametrize(
    ("person", "personnel_id", "room_id", "expected"),
    [
        ("Unknown", None, None, False),
        ("0311344119", None, None, True),
        ("0311344119", 1, 10, True),
    ],
)
def test_person_patch_access_matches_old_rules(
    person: str,
    personnel_id: int | None,
    room_id: int | None,
    expected: bool,
) -> None:
    location_store = SimpleNamespace(
        get_room=lambda _room_id: SimpleNamespace(name="general")
        if room_id == 10
        else None,
        check_room_access=lambda _personnel_id, _room_id: False,
    )
    assert _calculate_old_person_access(
        person,
        personnel_id,
        room_id,
        location_store,
    ) is expected

pytestmark = pytest.mark.unit
