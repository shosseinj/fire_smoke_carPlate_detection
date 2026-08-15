from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from app.core.auth import accessible_scope_ids, enforce_scoped_permission, get_current_user, require_permission
from app.core.auth_store import UserRecord
from app.core.common_schemas import UserBrief, resolve_user_brief, resolve_user_briefs
from app.core.jalali_utils import utc_iso_to_jalali_datetime
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


def _assign_created_room_to_source(
    runtime: Runtime,
    *,
    room_id: int,
    camera_id: int,
) -> None:
    camera = runtime.cam_store.get(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="دوربین یافت نشد")
    source = runtime.registry.get(camera.url)
    if source is None:
        LOGGER.warning(
            "Room %s owns camera %s but no source exists for %s",
            room_id,
            camera_id,
            camera.url,
        )
        return
    runtime.registry.update(camera.url, room_id=room_id)
    runtime._refresh_all_source_zones()


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
    name: str


class BuildingResponse(BaseModel):
    id: int
    name: str
    address: str | None = None
    description: str | None = None
    is_active: bool = True
    created_at: str
    updated_at: str | None = None
    created_at_jalali: str = ""
    updated_at_jalali: str | None = None
    created_by: UserBrief | None = None
    updated_by: UserBrief | None = None
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
    camera_name: str
    camera_number: int
    width: int
    high: int
    source_type: str
    section_id: int
    url: str


class SectionResponse(BaseModel):
    id: int
    section_name: str
    floor: str | None = None
    description: str | None = None
    is_active: bool = True
    building_id: int
    created_at: str
    updated_at: str | None = None
    created_at_jalali: str = ""
    updated_at_jalali: str | None = None
    created_by: UserBrief | None = None
    updated_by: UserBrief | None = None
    building_name: str = ""
    section_full_name: str = ""
    cameras: list[CameraMinimal] = []


# ── Room schemas ───────────────────────────────────────────────────


class RoomCreate(BaseModel):
    room_number: str | None = Field(default=None, max_length=50)
    room_name: str = Field(..., min_length=1, max_length=500)
    room_type: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=2000)
    camera_id: int = Field(ge=1)
    is_active: bool = True
    polygon_points: list[list[float]] | None = None


class RoomUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_number: str | None = Field(default=None, max_length=50)
    room_type: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=2000)
    is_active: bool | None = None
    polygon_points: list[list[float]] | None = None


class LegacyRoomUpdate(RoomUpdate):
    room_name: str | None = Field(default=None, min_length=1, max_length=500)


class RoomResponse(BaseModel):
    id: int
    room_number: str | None = None
    room_name: str | None = None
    room_type: str | None = None
    description: str | None = None
    is_active: bool = True
    camera_id: int | None = None
    section_id: int | None = None
    polygon_points: list[list[float]] | None = None
    created_at: str
    updated_at: str | None = None
    created_at_jalali: str = ""
    updated_at_jalali: str | None = None
    created_by: UserBrief | None = None
    updated_by: UserBrief | None = None


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


def _resolve_audit_briefs(
    record,
    store: LocationStore,
    user_briefs: dict[int, UserBrief] | None = None,
):
    """Resolve ``created_by`` and ``updated_by`` to ``UserBrief`` using the store's database."""
    c = u = None
    if user_briefs is not None:
        return (
            user_briefs.get(record.created_by),
            user_briefs.get(record.updated_by),
        )
    if hasattr(record, "created_by") and record.created_by is not None:
        with store.database.connection() as conn:
            c = resolve_user_brief(record.created_by, conn)
    if hasattr(record, "updated_by") and record.updated_by is not None:
        with store.database.connection() as conn:
            u = resolve_user_brief(record.updated_by, conn)
    return c, u


def _build_response(
    store: LocationStore,
    b,
    sections=None,
    user_briefs: dict[int, UserBrief] | None = None,
) -> BuildingResponse:
    if sections is None:
        sec_rows, _ = store.list_sections(building_id=b.id, limit=1000)
        sections = [SectionMinimal(id=s.id, name=s.name) for s in sec_rows]
    c, u = _resolve_audit_briefs(b, store, user_briefs)
    return BuildingResponse(
        id=b.id,
        name=b.name,
        address=b.address,
        description=b.description,
        is_active=True,
        created_at=b.created_at_utc,
        updated_at=b.updated_at_utc,
        created_at_jalali=utc_iso_to_jalali_datetime(b.created_at_utc) or "",
        updated_at_jalali=utc_iso_to_jalali_datetime(b.updated_at_utc),
        created_by=c,
        updated_by=u,
        sections=sections,
    )


def _section_response(
    store: LocationStore,
    s,
    building_names: dict[int, str] | None = None,
    user_briefs: dict[int, UserBrief] | None = None,
) -> SectionResponse:
    bld_name = ""
    if s.building_id is not None:
        if building_names is not None:
            bld_name = building_names.get(s.building_id, "")
        else:
            bld = store.get_building(s.building_id)
            bld_name = bld.name if bld else ""
    full_name = f"{bld_name} - {s.name}" if bld_name else s.name
    c, u = _resolve_audit_briefs(s, store, user_briefs)
    return SectionResponse(
        id=s.id,
        section_name=s.name,
        floor=s.floor,
        description=s.description,
        is_active=s.is_active,
        building_id=s.building_id or 0,
        created_at=s.created_at_utc,
        updated_at=s.updated_at_utc,
        created_at_jalali=utc_iso_to_jalali_datetime(s.created_at_utc) or "",
        updated_at_jalali=utc_iso_to_jalali_datetime(s.updated_at_utc),
        created_by=c,
        updated_by=u,
        building_name=bld_name,
        section_full_name=full_name,
    )


def _room_response(
    r,
    store: LocationStore | None = None,
    user_briefs: dict[int, UserBrief] | None = None,
) -> RoomResponse:
    c = u = None
    if store is not None:
        c, u = _resolve_audit_briefs(r, store, user_briefs)
    return RoomResponse(
        id=r.id,
        room_name=r.name,
        room_number=r.room_number,
        room_type=r.room_type,
        description=r.description,
        is_active=r.is_active,
        camera_id=r.cam_id,
        section_id=r.section_id,
        polygon_points=_polygon_to_list(r.polygon_json),
        created_at=r.created_at_utc,
        updated_at=r.updated_at_utc,
        created_at_jalali=utc_iso_to_jalali_datetime(r.created_at_utc) or "",
        updated_at_jalali=utc_iso_to_jalali_datetime(r.updated_at_utc),
        created_by=c,
        updated_by=u,
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
    current_user: UserRecord = Depends(get_current_user),
):
    store = _store(runtime)
    allowed_ids = accessible_scope_ids(current_user, "buildings.read", "building")
    records, _ = store.list_buildings(offset=skip, limit=limit, allowed_ids=allowed_ids)
    sections_by_building = store.list_sections_for_buildings(
        {record.id for record in records}
    )
    audit_ids = {
        user_id
        for record in records
        for user_id in (record.created_by, record.updated_by)
        if user_id is not None
    }
    with store.database.connection() as connection:
        user_briefs = resolve_user_briefs(audit_ids, connection)
    result = []
    for b in records:
        sec_rows = sections_by_building.get(b.id, [])
        sections = [SectionMinimal(id=s.id, name=s.name) for s in sec_rows]
        result.append(_build_response(store, b, sections, user_briefs))
    return result


@buildings_router.get("/{building_id}", response_model=BuildingResponse)
def get_building(
    building_id: int,
    runtime: Runtime = Depends(get_runtime),
    current_user: UserRecord = Depends(get_current_user),
):
    store = _store(runtime)
    b = store.get_building(building_id)
    if not b:
        raise HTTPException(status_code=404, detail="ساختمان یافت نشد")
    enforce_scoped_permission(current_user, "buildings.read", "building", building_id)
    return _build_response(store, b)


@buildings_router.post("/", response_model=BuildingResponse, status_code=201)
def create_building(
    payload: BuildingCreate,
    runtime: Runtime = Depends(get_runtime),
    current_user: UserRecord = Depends(get_current_user),
):
    enforce_scoped_permission(current_user, "buildings.create", "global", 0)
    try:
        b = _store(runtime).create_building(
            name=payload.name,
            address=payload.address,
            description=payload.description,
            created_by=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _build_response(_store(runtime), b, sections=[])


@buildings_router.patch("/{building_id}", response_model=BuildingResponse)
def update_building(
    building_id: int,
    payload: BuildingUpdate,
    runtime: Runtime = Depends(get_runtime),
    current_user: UserRecord = Depends(get_current_user),
):
    enforce_scoped_permission(current_user, "buildings.edit", "building", building_id)
    changes: dict[str, Any] = {}
    if payload.name is not None:
        changes["name"] = payload.name
    if payload.address is not None:
        changes["address"] = payload.address
    if payload.description is not None:
        changes["description"] = payload.description
    if not changes:
        raise HTTPException(status_code=422, detail="هیچ فیلدی برای به‌روزرسانی وارد نشده است")
    changes["updated_by"] = current_user.id
    store = _store(runtime)
    try:
        b = store.update_building(building_id=building_id, **changes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if not b:
        raise HTTPException(status_code=404, detail="ساختمان یافت نشد")
    enforce_scoped_permission(current_user, "buildings.delete", "building", building_id)
    return _build_response(store, b)


@buildings_router.delete("/{building_id}")
def delete_building(
    building_id: int,
    runtime: Runtime = Depends(get_runtime),
    current_user: UserRecord = Depends(get_current_user),
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
                   "ابتدا آن‌ها را غیرفعال یا جابجا کنید",
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
    current_user: UserRecord = Depends(get_current_user),
):
    store = _store(runtime)
    allowed_ids = accessible_scope_ids(current_user, "sections.read", "section")
    records, _ = store.list_sections(offset=skip, limit=limit, building_id=building_id, allowed_ids=allowed_ids)
    building_names = store.get_building_names_by_ids(
        {record.building_id for record in records if record.building_id is not None}
    )
    audit_ids = {
        user_id
        for record in records
        for user_id in (record.created_by, record.updated_by)
        if user_id is not None
    }
    with store.database.connection() as connection:
        user_briefs = resolve_user_briefs(audit_ids, connection)
    return [
        _section_response(store, record, building_names, user_briefs)
        for record in records
    ]


@sections_router.get("/{section_id}", response_model=SectionResponse)
def get_section(
    section_id: int,
    runtime: Runtime = Depends(get_runtime),
    current_user: UserRecord = Depends(get_current_user),
):
    store = _store(runtime)
    s = store.get_section(section_id)
    if not s:
        raise HTTPException(status_code=404, detail="بخش یافت نشد")
    enforce_scoped_permission(current_user, "sections.read", "section", section_id)
    return _section_response(store, s)


@sections_router.get("/{section_id}/cameras", response_model=list[CameraMinimal])
def get_section_cameras(
    section_id: int,
    runtime: Runtime = Depends(get_runtime),
    current_user: UserRecord = Depends(get_current_user),
):
    store = _store(runtime)
    if not store.get_section(section_id):
        raise HTTPException(status_code=404, detail="بخش یافت نشد")
    allowed_ids = accessible_scope_ids(current_user, "cameras.read", "camera")
    cams, _ = runtime.cam_store.list(section_id=section_id, limit=1000, allowed_ids=allowed_ids)
    return [
        CameraMinimal(
            id=cam.id,
            camera_name=cam.camera_name,
            camera_number=cam.camera_number,
            width=cam.width,
            high=cam.high,
            source_type=cam.source_type,
            section_id=cam.section_id,
            url=cam.url,
        )
        for cam in cams
    ]


@sections_router.post("/", response_model=SectionResponse, status_code=201)
def create_section(
    payload: SectionCreate,
    runtime: Runtime = Depends(get_runtime),
    current_user: UserRecord = Depends(get_current_user),
):
    enforce_scoped_permission(current_user, "sections.create", "building", payload.building_id)
    store = _store(runtime)
    with store.database.connection() as connection:
        bld = connection.execute(
            "SELECT id FROM buildings WHERE id = ?", (payload.building_id,)
        ).fetchone()
    if not bld:
        raise HTTPException(status_code=404, detail="ساختمان یافت نشد")
    try:
        s = store.create_section(
            name=payload.section_name,
            building_id=payload.building_id,
            description=payload.description,
            floor=payload.floor,
            is_active=payload.is_active,
            created_by=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _section_response(store, s)


@sections_router.patch("/{section_id}", response_model=SectionResponse)
def update_section(
    section_id: int,
    payload: SectionUpdate,
    runtime: Runtime = Depends(get_runtime),
    current_user: UserRecord = Depends(get_current_user),
):
    store = _store(runtime)
    s = store.get_section(section_id)
    if not s:
        raise HTTPException(status_code=404, detail="بخش یافت نشد")
    if payload.building_id is not None:
        with store.database.connection() as connection:
            bld = connection.execute(
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
    if payload.floor is not None:
        changes["floor"] = payload.floor
    if payload.is_active is not None:
        changes["is_active"] = payload.is_active
    if not changes:
        raise HTTPException(status_code=422, detail="هیچ فیلدی برای به‌روزرسانی وارد نشده است")
    changes["updated_by"] = current_user.id
    try:
        s = store.update_section(section_id=section_id, **changes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if not s:
        raise HTTPException(status_code=404, detail="بخش یافت نشد")
    return _section_response(store, s)


@rooms_router.patch("/{room_id}/assign-camera", include_in_schema=False)
def assign_source_to_room(
    room_id: int,
    source_uri: str = Query(...),
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_permission("application.manage")),
):
    store = _store(runtime)
    room = store.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="اتاق یافت نشد")
    source = runtime.registry.get(source_uri)
    if source is None:
        raise HTTPException(status_code=404, detail="دوربین یافت نشد")
    runtime.registry.update(source_uri, room_id=room_id)
    runtime._refresh_all_source_zones()
    return {
        "message": f"دوربین به اتاق {room_id} اختصاص داده شد",
        "source_uri": source_uri,
        "room_id": room_id,
    }


# @rooms_router.patch("/{room_id}/assign-camera/{cam_id}", response_model=RoomResponse)
# def assign_camera_to_room(
#     room_id: int,
#     cam_id: int,
#     runtime: Runtime = Depends(get_runtime),
#     _: UserRecord = Depends(require_permission("application.manage")),
# ):
#     store = _store(runtime)
#     if not store.get_room(room_id):
#         raise HTTPException(status_code=404, detail="اتاق یافت نشد")
#     try:
#         room = store.update_room(room_id=room_id, cam_id=cam_id)
#     except ValueError as exc:
#         code = 404 if "not found" in str(exc).lower() else 422
#         raise HTTPException(status_code=code, detail=str(exc)) from exc
#     if room is None:
#         raise HTTPException(status_code=404, detail="اتاق یافت نشد")
#     runtime._refresh_all_source_zones()
#     return _room_response(room, store)


@sections_router.delete("/{section_id}")
def delete_section(
    section_id: int,
    runtime: Runtime = Depends(get_runtime),
    current_user: UserRecord = Depends(get_current_user),
):
    store = _store(runtime)
    s = store.get_section(section_id)
    if not s:
        raise HTTPException(status_code=404, detail="بخش یافت نشد")
    enforce_scoped_permission(current_user, "sections.edit", "section", section_id)
    if payload.building_id is not None and payload.building_id != s.building_id:
        enforce_scoped_permission(current_user, "sections.edit", "building", payload.building_id)
    enforce_scoped_permission(current_user, "sections.delete", "section", section_id)
    registry = runtime.registry
    room_ids = {room.id for room in store.list_rooms(section_id=section_id, limit=1000)[0]}
    assigned = [c for c in registry.list() if c.room_id in room_ids]
    cam_count = runtime.cam_store.count(section_id=section_id)
    if assigned or cam_count:
        camera_count = len(assigned) + cam_count
        raise HTTPException(
            status_code=400,
            detail=f"امکان حذف بخش وجود ندارد. این بخش دارای {camera_count} دوربین است. "
                   "ابتدا آن‌ها را جابجا یا حذف کنید",
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
    camera_id: int | None = Query(default=None, ge=1),
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_permission("application.read")),
):
    records, _ = _store(runtime).list_rooms(
        offset=skip, limit=limit, cam_id=camera_id
    )
    store = _store(runtime)
    audit_ids = {
        user_id
        for record in records
        for user_id in (record.created_by, record.updated_by)
        if user_id is not None
    }
    with store.database.connection() as connection:
        user_briefs = resolve_user_briefs(audit_ids, connection)
    return [_room_response(record, store, user_briefs) for record in records]


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
    _: UserRecord = Depends(require_permission("application.read")),
):
    r = _store(runtime).get_room(room_id)
    if not r:
        raise HTTPException(status_code=404, detail="اتاق یافت نشد")
    return _room_response(r, _store(runtime))


@rooms_router.get("/{room_id}/personnel")
def get_room_access_list(
    room_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_permission("application.read")),
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
    _: UserRecord = Depends(require_permission("application.read")),
):
    store = _store(runtime)
    rooms = store.list_personnel_rooms(personnel_id)
    return [_room_response(r, store) for r in rooms]


@rooms_router.post("/", response_model=RoomResponse)
def create_new_room(
    payload: RoomCreate,
    runtime: Runtime = Depends(get_runtime),
    current_user: UserRecord = Depends(require_permission("application.manage")),
):
    if payload.polygon_points and len(payload.polygon_points) < 3:
        raise HTTPException(status_code=400, detail="چندضلعی باید حداقل ۳ نقطه داشته باشد")
    polygon_json = json.dumps(payload.polygon_points) if payload.polygon_points else None
    try:
        r = _store(runtime).create_room(
            name=payload.room_name,
            cam_id=payload.camera_id,
            room_number=payload.room_number,
            room_type=payload.room_type,
            description=payload.description,
            is_active=payload.is_active,
            polygon_json=polygon_json,
            created_by=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _assign_created_room_to_source(
        runtime,
        room_id=r.id,
        camera_id=payload.camera_id,
    )
    return _room_response(r, _store(runtime))


def _update_room(
    room_id: int,
    payload: RoomUpdate | LegacyRoomUpdate,
    runtime: Runtime,
    current_user: UserRecord,
):
    if payload.polygon_points is not None and len(payload.polygon_points) < 3:
        raise HTTPException(status_code=400, detail="چندضلعی باید حداقل ۳ نقطه داشته باشد")
    store = _store(runtime)
    existing = store.get_room(room_id)
    if not existing:
        raise HTTPException(status_code=404, detail="اتاق یافت نشد")
    changes: dict[str, Any] = {}
    if getattr(payload, "room_name", None) is not None:
        changes["name"] = payload.room_name
    if "room_number" in payload.model_fields_set:
        changes["room_number"] = payload.room_number
    if "room_type" in payload.model_fields_set:
        changes["room_type"] = payload.room_type
    if "description" in payload.model_fields_set:
        changes["description"] = payload.description
    if payload.is_active is not None:
        changes["is_active"] = payload.is_active
    if "polygon_points" in payload.model_fields_set:
        changes["polygon_json"] = json.dumps(payload.polygon_points) if payload.polygon_points is not None else None
    if not changes:
        raise HTTPException(status_code=422, detail="هیچ فیلدی برای به‌روزرسانی وارد نشده است")
    changes["updated_by"] = current_user.id
    try:
        r = store.update_room(room_id=room_id, **changes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if not r:
        raise HTTPException(status_code=404, detail="اتاق یافت نشد")
    runtime._refresh_all_source_zones()
    return _room_response(r, store)


@rooms_router.patch("/{room_id}", response_model=RoomResponse)
def patch_room(
    room_id: int,
    payload: RoomUpdate,
    runtime: Runtime = Depends(get_runtime),
    current_user: UserRecord = Depends(require_permission("application.manage")),
):
    return _update_room(room_id, payload, runtime, current_user)


@rooms_router.put("/{room_id}", response_model=RoomResponse)
def update_room_info(
    room_id: int,
    payload: LegacyRoomUpdate,
    runtime: Runtime = Depends(get_runtime),
    current_user: UserRecord = Depends(require_permission("application.manage")),
):
    return _update_room(room_id, payload, runtime, current_user)


@rooms_router.post("/{room_id}/grant/{personnel_id}")
def grant_access(
    room_id: int,
    personnel_id: int,
    assigned_by: str | None = None,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_permission("application.manage")),
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
    _: UserRecord = Depends(require_permission("application.manage")),
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
    _: UserRecord = Depends(require_permission("application.system")),
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
#     _: UserRecord = Depends(require_permission("application.read")),
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
#     _: UserRecord = Depends(require_permission("application.read")),
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
