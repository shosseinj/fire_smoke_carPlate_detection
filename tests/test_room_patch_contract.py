from __future__ import annotations

from typing import get_type_hints

import pytest
from pydantic import ValidationError

from app.api.locations import RoomUpdate
from app.api.locations import rooms_router


def test_room_patch_accepts_only_requested_fields() -> None:
    payload = RoomUpdate(
        polygon_points=[[0, 0], [1, 0], [1, 1]],
        room_number="101",
        room_type="office",
        description="Updated",
        is_active=False,
    )
    assert payload.is_active is False
    assert payload.room_number == "101"


def test_room_patch_preserves_explicit_nullable_field_semantics() -> None:
    payload = RoomUpdate(description=None, polygon_points=None)
    assert payload.model_fields_set == {"description", "polygon_points"}


def test_room_patch_rejects_room_name() -> None:
    with pytest.raises(ValidationError):
        RoomUpdate(room_name="Not allowed")


def test_rooms_router_exposes_patch_with_exact_update_model() -> None:
    patch_routes = [route for route in rooms_router.routes if route.path == "/rooms/{room_id}" and "PATCH" in route.methods]
    assert len(patch_routes) == 1
    assert get_type_hints(patch_routes[0].endpoint)["payload"] is RoomUpdate

pytestmark = pytest.mark.unit
