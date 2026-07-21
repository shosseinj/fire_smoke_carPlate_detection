from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.core.auth import require_role
from app.core.auth_store import UserRecord
from app.core.location_store import (
    BuildingRecord,
    LocationStore,
    RoomRecord,
    SectionRecord,
)
from app.runtime import Runtime

LOGGER = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/api/v1", tags=["locations"])


def get_runtime() -> Runtime:
    from app.main import runtime
    return runtime


def _store(runtime: Runtime) -> LocationStore:
    return runtime.location_store


# ── Pydantic models ─────────────────────────────────────────────────


class BuildingCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=500)
    address: str | None = Field(default=None, max_length=1000)
    description: str | None = Field(default=None, max_length=2000)


class BuildingUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=500)
    address: str | None = Field(default=None, max_length=1000)
    description: str | None = Field(default=None, max_length=2000)


class BuildingResponse(BaseModel):
    id: int
    name: str
    address: str | None = None
    description: str | None = None
    created_at_utc: str
    updated_at_utc: str


class SectionCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=500)
    building_id: int | None = Field(default=None)
    description: str | None = Field(default=None, max_length=2000)


class SectionUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=500)
    building_id: int | None = Field(default=None)
    description: str | None = Field(default=None, max_length=2000)


class SectionResponse(BaseModel):
    id: int
    building_id: int | None = None
    name: str
    description: str | None = None
    created_at_utc: str
    updated_at_utc: str


class RoomCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=500)
    section_id: int | None = Field(default=None)
    description: str | None = Field(default=None, max_length=2000)
    polygon_json: str | None = Field(
        default=None,
        description="JSON array of [x, y] vertices, e.g. [[0,0], [100,0], [100,100], [0,100]]",
    )


class RoomUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=500)
    section_id: int | None = Field(default=None)
    description: str | None = Field(default=None, max_length=2000)
    polygon_json: str | None = Field(default=None)


class RoomResponse(BaseModel):
    id: int
    section_id: int | None = None
    name: str
    description: str | None = None
    polygon_json: str | None = None
    created_at_utc: str
    updated_at_utc: str


class PersonnelAccessResponse(BaseModel):
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
    matched_at_utc: str


# ── Helper converters ───────────────────────────────────────────────


def _building_to_response(b: BuildingRecord) -> BuildingResponse:
    return BuildingResponse(
        id=b.id,
        name=b.name,
        address=b.address,
        description=b.description,
        created_at_utc=b.created_at_utc,
        updated_at_utc=b.updated_at_utc,
    )


def _section_to_response(s: SectionRecord) -> SectionResponse:
    return SectionResponse(
        id=s.id,
        building_id=s.building_id,
        name=s.name,
        description=s.description,
        created_at_utc=s.created_at_utc,
        updated_at_utc=s.updated_at_utc,
    )


def _room_to_response(r: RoomRecord) -> RoomResponse:
    return RoomResponse(
        id=r.id,
        section_id=r.section_id,
        name=r.name,
        description=r.description,
        polygon_json=r.polygon_json,
        created_at_utc=r.created_at_utc,
        updated_at_utc=r.updated_at_utc,
    )


# ═════════════════════════════════════════════════════════════════════
# Buildings
# ═════════════════════════════════════════════════════════════════════


@router.get("/buildings/", summary="List all buildings")
def list_buildings(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=1000),
    search: str | None = Query(default=None),
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> dict:
    records, total = _store(runtime).list_buildings(
        offset=offset, limit=limit, search=search
    )
    return {
        "items": [_building_to_response(r) for r in records],
        "count": len(records),
        "total": total,
    }


@router.post(
    "/buildings/",
    summary="Create a new building",
    status_code=status.HTTP_201_CREATED,
)
def create_building(
    payload: BuildingCreateRequest,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> BuildingResponse:
    try:
        record = _store(runtime).create_building(
            name=payload.name,
            address=payload.address,
            description=payload.description,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    return _building_to_response(record)


@router.get("/buildings/{building_id}", summary="Get a building by ID")
def get_building(
    building_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> BuildingResponse:
    record = _store(runtime).get_building(building_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Building not found"
        )
    return _building_to_response(record)


@router.put("/buildings/{building_id}", summary="Update a building")
def update_building(
    building_id: int,
    payload: BuildingUpdateRequest,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> BuildingResponse:
    changes = payload.model_dump(exclude_unset=True, exclude_none=True)
    if not changes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No fields to update",
        )
    try:
        record = _store(runtime).update_building(
            building_id=building_id, **changes
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Building not found"
        )
    return _building_to_response(record)


@router.delete("/buildings/{building_id}", summary="Delete a building")
def delete_building(
    building_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> dict:
    deleted = _store(runtime).delete_building(building_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Building not found"
        )
    return {"deleted": True, "building_id": building_id}


# ═════════════════════════════════════════════════════════════════════
# Sections
# ═════════════════════════════════════════════════════════════════════


@router.get("/sections/", summary="List all sections")
def list_sections(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=1000),
    building_id: int | None = Query(default=None),
    search: str | None = Query(default=None),
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> dict:
    records, total = _store(runtime).list_sections(
        offset=offset, limit=limit, building_id=building_id, search=search
    )
    return {
        "items": [_section_to_response(r) for r in records],
        "count": len(records),
        "total": total,
    }


@router.post(
    "/sections/",
    summary="Create a new section",
    status_code=status.HTTP_201_CREATED,
)
def create_section(
    payload: SectionCreateRequest,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> SectionResponse:
    try:
        record = _store(runtime).create_section(
            name=payload.name,
            building_id=payload.building_id,
            description=payload.description,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    return _section_to_response(record)


@router.get("/sections/{section_id}", summary="Get a section by ID")
def get_section(
    section_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> SectionResponse:
    record = _store(runtime).get_section(section_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Section not found"
        )
    return _section_to_response(record)


@router.put("/sections/{section_id}", summary="Update a section")
def update_section(
    section_id: int,
    payload: SectionUpdateRequest,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> SectionResponse:
    changes = payload.model_dump(exclude_unset=True, exclude_none=True)
    if not changes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No fields to update",
        )
    try:
        record = _store(runtime).update_section(
            section_id=section_id, **changes
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Section not found"
        )
    return _section_to_response(record)


@router.delete("/sections/{section_id}", summary="Delete a section")
def delete_section(
    section_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> dict:
    deleted = _store(runtime).delete_section(section_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Section not found"
        )
    return {"deleted": True, "section_id": section_id}


@router.post(
    "/sections/{section_id}/assign-camera/{camera_id}",
    summary="Assign a camera to a section",
    status_code=status.HTTP_200_OK,
)
def assign_camera_to_section(
    section_id: int,
    camera_id: str,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> dict:
    store = _store(runtime)
    # Verify section exists
    section = store.get_section(section_id)
    if section is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Section not found"
        )
    # Verify camera exists via registry
    registry = runtime.registry
    cam = registry.get(camera_id)
    if cam is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found"
        )
    # Update camera with section_id (via metadata for runtime compatibility)
    metadata = dict(cam.metadata) if cam.metadata else {}
    metadata["section_id"] = section_id
    registry.update(camera_id, metadata=metadata, section_id=section_id)
    return {
        "assigned": True,
        "camera_id": camera_id,
        "section_id": section_id,
    }


@router.get(
    "/sections/{section_id}/cameras",
    summary="List cameras assigned to a section",
)
def list_section_cameras(
    section_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> dict:
    store = _store(runtime)
    section = store.get_section(section_id)
    if section is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Section not found"
        )
    # Get cameras assigned to this section via registry filter
    registry = runtime.registry
    all_cameras = [
        {
            "camera_id": cam.source_id,
            "name": cam.name,
            "enabled": cam.enabled,
            "source_uri": cam.source_uri,
            "metadata": cam.metadata,
        }
        for cam in registry.list()
    ]
    filtered = store.filter_cameras_by_section(all_cameras, section_id)
    return {
        "items": filtered,
        "count": len(filtered),
    }


# ═════════════════════════════════════════════════════════════════════
# Rooms
# ═════════════════════════════════════════════════════════════════════


@router.get("/rooms/", summary="List all rooms")
def list_rooms(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=1000),
    section_id: int | None = Query(default=None),
    search: str | None = Query(default=None),
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> dict:
    records, total = _store(runtime).list_rooms(
        offset=offset, limit=limit, section_id=section_id, search=search
    )
    return {
        "items": [_room_to_response(r) for r in records],
        "count": len(records),
        "total": total,
    }


@router.post(
    "/rooms/",
    summary="Create a new room",
    status_code=status.HTTP_201_CREATED,
)
def create_room(
    payload: RoomCreateRequest,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> RoomResponse:
    try:
        record = _store(runtime).create_room(
            name=payload.name,
            section_id=payload.section_id,
            description=payload.description,
            polygon_json=payload.polygon_json,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    return _room_to_response(record)


@router.get("/rooms/{room_id}", summary="Get a room by ID")
def get_room(
    room_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> RoomResponse:
    record = _store(runtime).get_room(room_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Room not found"
        )
    return _room_to_response(record)


@router.put("/rooms/{room_id}", summary="Update a room")
def update_room(
    room_id: int,
    payload: RoomUpdateRequest,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> RoomResponse:
    changes = payload.model_dump(exclude_unset=True, exclude_none=True)
    if not changes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No fields to update",
        )
    try:
        record = _store(runtime).update_room(
            room_id=room_id, **changes
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Room not found"
        )
    return _room_to_response(record)


@router.delete("/rooms/{room_id}", summary="Delete a room")
def delete_room(
    room_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> dict:
    deleted = _store(runtime).delete_room(room_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Room not found"
        )
    return {"deleted": True, "room_id": room_id}


# ═════════════════════════════════════════════════════════════════════
# Personnel Room Access
# ═════════════════════════════════════════════════════════════════════


@router.post(
    "/rooms/{room_id}/grant/{personnel_id}",
    summary="Grant personnel access to a room",
    status_code=status.HTTP_201_CREATED,
)
def grant_room_access(
    room_id: int,
    personnel_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> dict:
    try:
        record = _store(runtime).grant_room_access(
            personnel_id=personnel_id,
            room_id=room_id,
            granted_by=getattr(_, "username", None),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT
            if "already" in str(exc)
            else status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    return {
        "granted": True,
        "personnel_id": record.personnel_id,
        "room_id": record.room_id,
        "granted_at_utc": record.granted_at_utc,
    }


@router.post(
    "/rooms/{room_id}/revoke/{personnel_id}",
    summary="Revoke personnel access from a room",
)
def revoke_room_access(
    room_id: int,
    personnel_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> dict:
    revoked = _store(runtime).revoke_room_access(
        personnel_id=personnel_id, room_id=room_id
    )
    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Access record not found",
        )
    return {
        "revoked": True,
        "personnel_id": personnel_id,
        "room_id": room_id,
    }


@router.get(
    "/rooms/check-access/{personnel_id}/{room_id}",
    summary="Check if a personnel has access to a room",
)
def check_room_access(
    personnel_id: int,
    room_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> dict:
    has_access = _store(runtime).check_room_access(
        personnel_id=personnel_id, room_id=room_id
    )
    return {
        "has_access": has_access,
        "personnel_id": personnel_id,
        "room_id": room_id,
    }


@router.get(
    "/rooms/personnel/{personnel_id}/rooms",
    summary="List rooms a personnel has access to",
)
def list_personnel_rooms(
    personnel_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> dict:
    rooms = _store(runtime).list_personnel_rooms(personnel_id)
    return {
        "items": [_room_to_response(r) for r in rooms],
        "count": len(rooms),
    }


@router.get(
    "/rooms/{room_id}/personnel",
    summary="List personnel with access to a room",
)
def list_room_personnel(
    room_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> dict:
    records = _store(runtime).list_room_personnel(room_id)
    return {
        "items": records,
        "count": len(records),
    }


# ═════════════════════════════════════════════════════════════════════
# Detection Room Matches (read-only via API)
# ═════════════════════════════════════════════════════════════════════


@router.get(
    "/rooms/{room_id}/matches",
    summary="List detection matches for a room",
)
def list_room_matches(
    room_id: int,
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("operator")),
) -> dict:
    store = _store(runtime)
    room = store.get_room(room_id)
    if room is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Room not found"
        )
    records, total = store.list_matches_for_room(
        room_id=room_id, limit=limit, offset=offset
    )
    return {
        "items": [
            DetectionMatchResponse(
                id=m.id,
                detection_type=m.detection_type,
                detection_event_id=m.detection_event_id,
                room_id=m.room_id,
                personnel_id=m.personnel_id,
                camera_id=m.camera_id,
                matched_at_utc=m.matched_at_utc,
            )
            for m in records
        ],
        "count": len(records),
        "total": total,
    }
