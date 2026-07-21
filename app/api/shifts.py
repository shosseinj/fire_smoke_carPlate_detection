"""API router for Work Shifts."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import require_role
from app.core.shift_store import (
    WEEKDAY_COLS,
    WorkShiftRecord,
    _is_overnight,
)

router = APIRouter(prefix="/api/v1/shifts", tags=["Shifts"])


def get_runtime() -> Any:
    from app.main import runtime
    return runtime


def get_shift_store() -> Any:
    return get_runtime().shift_store


@router.get("/")
def list_shifts(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    shift_type: str | None = Query(None),
    search: str | None = Query(None),
    _: dict = Depends(require_role("operator")),
) -> dict[str, Any]:
    store = get_shift_store()
    records, total = store.list(
        offset=offset, limit=limit,
        shift_type=shift_type, search=search,
    )
    return {
        "shifts": [_record_to_dict(r) for r in records],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@router.get("/{shift_id}")
def get_shift(
    shift_id: int,
    _: dict = Depends(require_role("operator")),
) -> dict[str, Any]:
    store = get_shift_store()
    record = store.get(shift_id)
    if record is None:
        raise HTTPException(404, "Shift not found")
    return {"shift": _record_to_dict(record)}


@router.post("/", status_code=201)
def create_shift(
    body: dict[str, Any],
    _: dict = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_shift_store()
    try:
        kwargs = {k: v for k, v in body.items() if k in WEEKDAY_COLS or k in (
            "shift_name", "shift_type", "start_time", "end_time",
            "max_minutes_delay", "max_minutes_early", "max_overtime_hours",
        )}
        record = store.create(**kwargs)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"shift": _record_to_dict(record)}


@router.put("/{shift_id}")
def update_shift(
    shift_id: int,
    body: dict[str, Any],
    _: dict = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_shift_store()
    try:
        kwargs = {k: v for k, v in body.items() if k in WEEKDAY_COLS or k in (
            "shift_name", "shift_type", "start_time", "end_time",
            "max_minutes_delay", "max_minutes_early", "max_overtime_hours",
        )}
        record = store.update(shift_id, **kwargs)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if record is None:
        raise HTTPException(404, "Shift not found")
    return {"shift": _record_to_dict(record)}


@router.delete("/{shift_id}")
def delete_shift(
    shift_id: int,
    _: dict = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_shift_store()
    deleted = store.delete(shift_id)
    if not deleted:
        raise HTTPException(404, "Shift not found")
    return {"deleted": True}


# ── Personnel assignment ──────────────────────────────────────────────


@router.post("/{shift_id}/assign/{personnel_id}")
def assign_personnel_to_shift(
    shift_id: int,
    personnel_id: int,
    _: dict = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_shift_store()
    try:
        store.assign_personnel(personnel_id, shift_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"assigned": True}


@router.delete("/assign/{personnel_id}")
def remove_personnel_shift(
    personnel_id: int,
    _: dict = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_shift_store()
    removed = store.remove_personnel_shift(personnel_id)
    return {"removed": removed}


@router.get("/{shift_id}/personnel")
def list_personnel_in_shift(
    shift_id: int,
    _: dict = Depends(require_role("operator")),
) -> dict[str, Any]:
    store = get_shift_store()
    personnel = store.list_personnel_in_shift(shift_id)
    return {"personnel": personnel, "count": len(personnel)}


# ── Statistics ────────────────────────────────────────────────────────


@router.get("/statistics/summary")
def shift_statistics(
    _: dict = Depends(require_role("operator")),
) -> dict[str, Any]:
    store = get_shift_store()
    return store.statistics()


# ── Helpers ───────────────────────────────────────────────────────────


def _record_to_dict(r: WorkShiftRecord) -> dict[str, Any]:
    return {
        "id": r.id,
        "shift_name": r.shift_name,
        "shift_type": r.shift_type,
        "start_time": r.start_time,
        "end_time": r.end_time,
        "max_minutes_delay": r.max_minutes_delay,
        "max_minutes_early": r.max_minutes_early,
        "max_overtime_hours": r.max_overtime_hours,
        "overnight": _is_overnight(r.start_time, r.end_time),
        "works_saturday": r.works_saturday,
        "works_sunday": r.works_sunday,
        "works_monday": r.works_monday,
        "works_tuesday": r.works_tuesday,
        "works_wednesday": r.works_wednesday,
        "works_thursday": r.works_thursday,
        "works_friday": r.works_friday,
        "created_at_utc": r.created_at_utc,
        "updated_at_utc": r.updated_at_utc,
    }
