from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.api.detection_logs import DetectionLogUpdate, _calculate_old_person_access


def test_person_patch_requires_identity_and_trims_person() -> None:
    update = DetectionLogUpdate(person=" 0311344119 ", confidence=0.8)
    assert update.person == "0311344119"
    assert update.confidence == 0.8

    with pytest.raises(ValidationError):
        DetectionLogUpdate(confidence=0.8)


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
