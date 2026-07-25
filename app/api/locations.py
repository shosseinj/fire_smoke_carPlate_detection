from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.core.auth import require_role
from app.core.auth_store import UserRecord
from app.core.location_store import LocationStore
from app.runtime import Runtime

LOGGER = logging.getLogger("uvicorn.error")

buildings_router = APIRouter(prefix="/buildings", tags=["Buildings"])
sections_router = APIRouter(prefix="/sections", tags=["Sections"])
rooms_router = APIRouter(prefix="/rooms", tags=["Rooms"])


def get_runtime() -> Runtime:
    from app.main import runtime
    return runtime


def _store(runtime: Runtime) -> LocationStore:
    return runtime.location_store


# ── Building schemas ────────────────────────────────────────────────


class BuildingCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=500)
    address: str | None = Field(default=None, max_length=1000)
    description: str | None = Field(default=None, max_length=2000)
    is_active: bool = True


class BuildingUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=500)
    address: str | None = Field(default=None, max_length=1000)
    description: str | None = Field(default=None, max_length=2000)
    is_active: bool | None = None


class SectionMinimal(BaseModel):
    id: int
    section_name: str


class BuildingResponse(BaseModel):
    id: int
    name: str
    address: str | None = None
    description: str | None = None
    is_active: bool = True
    created_at: str
    updated_at: str | None = None
    sections: list[SectionMinimal] = []


# ── Section schemas ────────────────────────────────────────────────


class SectionCreate(BaseModel):
    section_name: str = Field(..., min_length=1, max_length=500)
    floor: str | None = Field(default=None, max_length=50)
    description: str | None = Field(default=None, max_length=2000)
    is_active: bool = True
    building_id: int


class SectionUpdate(BaseModel):
    section_name: str | None = Field(default=None, min_length=1, max_length=500)
    floor: str | None = Field(default=None, max_length=50)
    description: str | None = Field(default=None, max_length=2000)
    is_active: bool | None = None
    building_id: int | None = None


class CameraMinimal(BaseModel):
    id: int
    camera_name: str | None = None
    camera_url: str | None = None
    is_active: bool = True
    section_id: int | None = None


class SectionResponse(BaseModel):
    id: int
    section_name: str
    floor: str | None = None
    description: str | None = None
    is_active: bool = True
    building_id: int
    created_at: str
    updated_at: str | None = None
    building_name: str = ""
    section_full_name: str = ""
    cameras: list[CameraMinimal] = []


# ── Room schemas ───────────────────────────────────────────────────


class RoomCreate(BaseModel):
    room_number: str | None = Field(default=None, max_length=50)
    room_name: str = Field(..., min_length=1, max_length=500)
    room_type: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=2000)
    camera_id: int | None = None
    section_id: int | None = None
    polygon_points: list[list[float]] | None = None


class RoomUpdate(BaseModel):
    room_number: str | None = Field(default=None, max_length=50)
    room_name: str | None = Field(default=None, min_length=1, max_length=500)
    room_type: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=2000)
    camera_id: int | None = None
    section_id: int | None = None
    polygon_points: list[list[float]] | None = None


class RoomResponse(BaseModel):
    id: int
    room_number: str | None = None
    room_name: str | None = None
    room_type: str | None = None
    description: str | None = None
    is_active: bool = True
    camera_id: int | None = None
    polygon_points: list[list[float]] | None = None
    created_at: str
    updated_at: str | None = None


class PersonnelAccessEntry(BaseModel):
    personnel_id: int
    granted_at_utc: str
    granted_by: str | None = None


class DetectionMatchResponse(BaseModel):
    id: int
    detection_type: str
    detection_event_id: int
    room_id: int
    personnel_id: int | None = None
    camera_id: str | None = None
    track_id: int | None = None
    transition_type: str | None = None
    matched_at_utc: str


# ── Helpers ────────────────────────────────────────────────────────


def _polygon_to_list(polygon_json: str | None) -> list[list[float]] | None:
    if not polygon_json:
        return None
    try:
        points: list[list[float]] = json.loads(polygon_json)
        if isinstance(points, list) and len(points) >= 3:
            return points
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _build_response(store: LocationStore, b, sections=None) -> BuildingResponse:
    if sections is None:
        sec_rows, _ = store.list_sections(building_id=b.id, limit=1000)
        sections = [SectionMinimal(id=s.id, section_name=s.name) for s in sec_rows]
    return BuildingResponse(
        id=b.id,
        name=b.name,
        address=b.address,
        description=b.description,
        is_active=True,
        created_at=b.created_at_utc,
        updated_at=b.updated_at_utc,
        sections=sections,
    )


def _section_response(store: LocationStore, s) -> SectionResponse:
    bld_name = ""
    if s.building_id is not None:
        bld = store.get_building(s.building_id)
        bld_name = bld.name if bld else ""
    full_name = f"{bld_name} - {s.name}" if bld_name else s.name
    return SectionResponse(
        id=s.id,
        section_name=s.name,
        floor=None,
        description=s.description,
        is_active=True,
        building_id=s.building_id or 0,
        created_at=s.created_at_utc,
        updated_at=s.updated_at_utc,
        building_name=bld_name,
        section_full_name=full_name,
    )


def _room_response(r) -> RoomResponse:
    return RoomResponse(
        id=r.id,
        room_number=None,
        room_name=r.name,
        room_type=None,
        description=r.description,
        is_active=True,
        camera_id=None,
        polygon_points=_polygon_to_list(r.polygon_json),
        created_at=r.created_at_utc,
        updated_at=r.updated_at_utc,
    )


# ═════════════════════════════════════════════════════════════════════
# Buildings
# ═════════════════════════════════════════════════════════════════════


@buildings_router.get("/", response_model=list[BuildingResponse])
def list_buildings(
    active_only: bool = True,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
):
    store = _store(runtime)
    records, _ = store.list_buildings(offset=skip, limit=limit)
    result = []
    for b in records:
        sec_rows, _ = store.list_sections(building_id=b.id, limit=1000)
        sections = [SectionMinimal(id=s.id, section_name=s.name) for s in sec_rows]
        result.append(_build_response(store, b, sections))
    return result


@buildings_router.get("/{building_id}", response_model=BuildingResponse)
def get_building(
    building_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
):
    store = _store(runtime)
    b = store.get_building(building_id)
    if not b:
        raise HTTPException(status_code=404, detail="ساختمان یافت نشد")
    return _build_response(store, b)


@buildings_router.post("/", response_model=BuildingResponse, status_code=201)
def create_building(
    payload: BuildingCreate,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
):
    try:
        b = _store(runtime).create_building(
            name=payload.name,
            address=payload.address,
            description=payload.description,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _build_response(_store(runtime), b, sections=[])


@buildings_router.patch("/{building_id}", response_model=BuildingResponse)
def update_building(
    building_id: int,
    payload: BuildingUpdate,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
):
    changes: dict[str, Any] = {}
    if payload.name is not None:
        changes["name"] = payload.name
    if payload.address is not None:
        changes["address"] = payload.address
    if payload.description is not None:
        changes["description"] = payload.description
    if not changes:
        raise HTTPException(status_code=422, detail="No fields to update")
    store = _store(runtime)
    try:
        b = store.update_building(building_id=building_id, **changes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if not b:
        raise HTTPException(status_code=404, detail="ساختمان یافت نشد")
    return _build_response(store, b)


@buildings_router.delete("/{building_id}")
def delete_building(
    building_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
):
    store = _store(runtime)
    b = store.get_building(building_id)
    if not b:
        raise HTTPException(status_code=404, detail="ساختمان یافت نشد")
    sec_rows, _ = store.list_sections(building_id=building_id, limit=1)
    if sec_rows:
        total, _ = store.list_sections(building_id=building_id, limit=0)
        count = total
        raise HTTPException(
            status_code=400,
            detail=f"امکان حذف ساختمان وجود ندارد. این ساختمان دارای {count} بخش است. "
                   "Deactivate or reassign them first.",
        )
    store.delete_building(building_id)
    return {"message": "ساختمان با موفقیت حذف شد"}


# ═════════════════════════════════════════════════════════════════════
# Sections
# ═════════════════════════════════════════════════════════════════════


@sections_router.get("/", response_model=list[SectionResponse])
def list_sections(
    active_only: bool = True,
    building_id: int | None = Query(None, description="فیلتر بر اساس ساختمان"),
    skip: int = 0,
    limit: int = 100,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
):
    store = _store(runtime)
    records, _ = store.list_sections(offset=skip, limit=limit, building_id=building_id)
    return [_section_response(store, s) for s in records]


@sections_router.get("/{section_id}", response_model=SectionResponse)
def get_section(
    section_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
):
    store = _store(runtime)
    s = store.get_section(section_id)
    if not s:
        raise HTTPException(status_code=404, detail="بخش یافت نشد")
    return _section_response(store, s)


@sections_router.get("/{section_id}/cameras")
def get_section_cameras(
    section_id: int,
    active_only: bool = True,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
):
    store = _store(runtime)
    s = store.get_section(section_id)
    if not s:
        raise HTTPException(status_code=404, detail="بخش یافت نشد")
    registry = runtime.registry
    cams = registry.list()
    if active_only:
        cams = [c for c in cams if c.enabled]
    room_ids = {room.id for room in store.list_rooms(section_id=section_id, limit=1000)[0]}
    section_cams = [c for c in cams if c.room_id in room_ids]
    return [
        {"id": idx, "camera_name": cam.name, "camera_url": cam.source_uri, "is_active": cam.enabled, "section_id": section_id, "room_id": cam.room_id}
        for idx, cam in enumerate(section_cams, start=1)
    ]


@sections_router.post("/", response_model=SectionResponse, status_code=201)
def create_section(
    payload: SectionCreate,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
):
    store = _store(runtime)
    bld = store.database.connection().execute(
        "SELECT id FROM buildings WHERE id = ?", (payload.building_id,)
    ).fetchone()
    if not bld:
        raise HTTPException(status_code=404, detail="ساختمان یافت نشد")
    try:
        s = store.create_section(
            name=payload.section_name,
            building_id=payload.building_id,
            description=payload.description,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _section_response(store, s)


@sections_router.patch("/{section_id}", response_model=SectionResponse)
def update_section(
    section_id: int,
    payload: SectionUpdate,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
):
    store = _store(runtime)
    s = store.get_section(section_id)
    if not s:
        raise HTTPException(status_code=404, detail="بخش یافت نشد")
    if payload.building_id is not None:
        bld = store.database.connection().execute(
            "SELECT id FROM buildings WHERE id = ?", (payload.building_id,)
        ).fetchone()
        if not bld:
            raise HTTPException(status_code=404, detail="ساختمان یافت نشد")
    changes: dict[str, Any] = {}
    if payload.section_name is not None:
        changes["name"] = payload.section_name
    if payload.building_id is not None:
        changes["building_id"] = payload.building_id
    if payload.description is not None:
        changes["description"] = payload.description
    if not changes:
        raise HTTPException(status_code=422, detail="No fields to update")
    try:
        s = store.update_section(section_id=section_id, **changes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if not s:
        raise HTTPException(status_code=404, detail="بخش یافت نشد")
    return _section_response(store, s)


@rooms_router.patch("/{room_id}/assign-camera")
def assign_camera_to_room(
    room_id: int,
    source_uri: str = Query(...),
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
):
    store = _store(runtime)
    room = store.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="اتاق یافت نشد")
    registry = runtime.registry
    cam = registry.get(source_uri)
    if cam is None:
        raise HTTPException(status_code=404, detail="دوربین یافت نشد")
    registry.update(source_uri, room_id=room_id)
    runtime._refresh_all_source_zones()
    return {"message": f"دوربین به اتاق {room_id} اختصاص داده شد", "source_uri": source_uri, "room_id": room_id}


@sections_router.delete("/{section_id}")
def delete_section(
    section_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
):
    store = _store(runtime)
    s = store.get_section(section_id)
    if not s:
        raise HTTPException(status_code=404, detail="بخش یافت نشد")
    registry = runtime.registry
    room_ids = {room.id for room in store.list_rooms(section_id=section_id, limit=1000)[0]}
    assigned = [c for c in registry.list() if c.room_id in room_ids]
    if assigned:
        raise HTTPException(
            status_code=400,
            detail=f"امکان حذف بخش وجود ندارد. این بخش دارای {len(assigned)} دوربین است. "
                   "Reassign or delete them first.",
        )
    store.delete_section(section_id)
    return {"message": "بخش با موفقیت حذف شد"}


# ═════════════════════════════════════════════════════════════════════
# Rooms
# ═════════════════════════════════════════════════════════════════════


@rooms_router.get("/", response_model=list[RoomResponse])
def list_rooms(
    active_only: bool = True,
    skip: int = 0,
    limit: int = 100,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
):
    records, _ = _store(runtime).list_rooms(offset=skip, limit=limit)
    return [_room_response(r) for r in records]


@rooms_router.get("/check-access/{personnel_id}/{room_id}")
def check_access(
    personnel_id: int,
    room_id: int,
    runtime: Runtime = Depends(get_runtime),
):
    has_access = _store(runtime).check_room_access(
        personnel_id=personnel_id, room_id=room_id
    )
    return {"has_access": has_access, "personnel_id": personnel_id, "room_id": room_id}


@rooms_router.get("/{room_id}", response_model=RoomResponse)
def get_room_by_id(
    room_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
):
    r = _store(runtime).get_room(room_id)
    if not r:
        raise HTTPException(status_code=404, detail="اتاق یافت نشد")
    return _room_response(r)


@rooms_router.get("/{room_id}/personnel")
def get_room_access_list(
    room_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
):
    store = _store(runtime)
    r = store.get_room(room_id)
    if not r:
        raise HTTPException(status_code=404, detail="اتاق یافت نشد")
    records = store.list_room_personnel(room_id)
    return [PersonnelAccessEntry(**p) for p in records]


@rooms_router.get("/personnel/{personnel_id}/rooms", response_model=list[RoomResponse])
def get_personnel_rooms_list(
    personnel_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
):
    rooms = _store(runtime).list_personnel_rooms(personnel_id)
    return [_room_response(r) for r in rooms]


@rooms_router.post("/", response_model=RoomResponse)
def create_new_room(
    payload: RoomCreate,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
):
    if payload.polygon_points and len(payload.polygon_points) < 3:
        raise HTTPException(status_code=400, detail="چندضلعی باید حداقل ۳ نقطه داشته باشد")
    polygon_json = json.dumps(payload.polygon_points) if payload.polygon_points else None
    try:
        r = _store(runtime).create_room(
            name=payload.room_name,
            section_id=payload.section_id,
            description=payload.description,
            polygon_json=polygon_json,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _room_response(r)


@rooms_router.put("/{room_id}", response_model=RoomResponse)
def update_room_info(
    room_id: int,
    payload: RoomUpdate,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
):
    if payload.polygon_points is not None and len(payload.polygon_points) < 3:
        raise HTTPException(status_code=400, detail="چندضلعی باید حداقل ۳ نقطه داشته باشد")
    store = _store(runtime)
    existing = store.get_room(room_id)
    if not existing:
        raise HTTPException(status_code=404, detail="اتاق یافت نشد")
    changes: dict[str, Any] = {}
    if payload.room_name is not None:
        changes["name"] = payload.room_name
    if payload.section_id is not None:
        changes["section_id"] = payload.section_id
    if payload.description is not None:
        changes["description"] = payload.description
    if payload.polygon_points is not None:
        changes["polygon_json"] = json.dumps(payload.polygon_points)
    if not changes:
        raise HTTPException(status_code=422, detail="No fields to update")
    try:
        r = store.update_room(room_id=room_id, **changes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if not r:
        raise HTTPException(status_code=404, detail="اتاق یافت نشد")
    runtime._refresh_all_source_zones()
    return _room_response(r)


@rooms_router.post("/{room_id}/grant/{personnel_id}")
def grant_access(
    room_id: int,
    personnel_id: int,
    assigned_by: str | None = None,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
):
    try:
        record = _store(runtime).grant_room_access(
            personnel_id=personnel_id,
            room_id=room_id,
            granted_by=assigned_by or getattr(_, "username", None),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT if "already" in str(exc)
            else status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    return {"message": "دسترسی با موفقیت ثبت شد"}


@rooms_router.post("/{room_id}/revoke/{personnel_id}")
@rooms_router.delete("/{room_id}/revoke/{personnel_id}", include_in_schema=False)
def revoke_access(
    room_id: int,
    personnel_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
):
    success = _store(runtime).revoke_room_access(
        personnel_id=personnel_id, room_id=room_id
    )
    if not success:
        raise HTTPException(status_code=404, detail="دسترسی یافت نشد")
    return {"message": "دسترسی با موفقیت لغو شد"}


@rooms_router.delete("/{room_id}")
def remove_room(
    room_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
):
    r = _store(runtime).get_room(room_id)
    if not r:
        raise HTTPException(status_code=404, detail="اتاق یافت نشد")
    assigned_sources = [item.source_uri for item in runtime.registry.list() if item.room_id == room_id]
    _store(runtime).delete_room(room_id)
    for source_uri in assigned_sources:
        runtime.registry.update(source_uri, room_id=None)
    runtime._refresh_all_source_zones()
    return {"message": "اتاق با موفقیت حذف شد"}


# ═════════════════════════════════════════════════════════════════════
# Zone Entry/Exit Events
# ═════════════════════════════════════════════════════════════════════


# @rooms_router.get("/{room_id}/entry-exits")
# def list_room_entry_exits(
#     room_id: int,
#     limit: int = Query(default=50, ge=1, le=1000),
#     offset: int = Query(default=0, ge=0),
#     transition_type: str | None = Query(default=None, description="Filter by 'entered' or 'exited'"),
#     runtime: Runtime = Depends(get_runtime),
#     _: UserRecord = Depends(require_role("operator")),
# ) -> dict:
#     """List zone entry/exit events for a specific room/polygon.

#     Returns events where a detected human entered or exited the polygon zone.
#     When transition_type is specified, filters to only 'entered' or 'exited' events.
#     """
#     store = _store(runtime)
#     room = store.get_room(room_id)
#     if room is None:
#         raise HTTPException(status_code=404, detail="اتاق یافت نشد")
#     records, total = store.list_matches_for_room(
#         room_id=room_id, limit=limit, offset=offset,
#         transition_type=transition_type,
#     )
#     return {
#         "items": [
#             DetectionMatchResponse(
#                 id=m.id,
#                 detection_type=m.detection_type,
#                 detection_event_id=m.detection_event_id,
#                 room_id=m.room_id,
#                 personnel_id=m.personnel_id,
#                 camera_id=m.camera_id,
#                 matched_at_utc=m.matched_at_utc,
#             )
#             for m in records
#             if m.transition_type is not None  # Only show actual entry/exit transitions
#         ],
#         "count": sum(1 for m in records if m.transition_type is not None),
#         "total": sum(1 for m in records if m.transition_type is not None),
#     }


# ═════════════════════════════════════════════════════════════════════
# Detection Room Matches
# ═════════════════════════════════════════════════════════════════════


# @rooms_router.get("/{room_id}/matches")
# def list_room_matches(
#     room_id: int,
#     limit: int = Query(default=50, ge=1, le=1000),
#     offset: int = Query(default=0, ge=0),
#     runtime: Runtime = Depends(get_runtime),
#     _: UserRecord = Depends(require_role("operator")),
# ) -> dict:
#     store = _store(runtime)
#     room = store.get_room(room_id)
#     if room is None:
#         raise HTTPException(status_code=404, detail="Room not found")
#     records, total = store.list_matches_for_room(
#         room_id=room_id, limit=limit, offset=offset
#     )
#     return {
#         "items": [
#             DetectionMatchResponse(
#                 id=m.id,
#                 detection_type=m.detection_type,
#                 detection_event_id=m.detection_event_id,
#                 room_id=m.room_id,
#                 personnel_id=m.personnel_id,
#                 camera_id=m.camera_id,
#                 matched_at_utc=m.matched_at_utc,
#             )
#             for m in records
#         ],
#         "count": len(records),
#         "total": total,
#     }
