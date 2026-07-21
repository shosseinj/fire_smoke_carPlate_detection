"""API router for Holidays."""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import require_role
from app.core.holiday_store import HolidayRecord

router = APIRouter(prefix="/api/v1/holidays", tags=["Holidays"])


def get_runtime() -> Any:
    from app.main import runtime
    return runtime


def get_holiday_store() -> Any:
    return get_runtime().holiday_store


@router.get("/")
def list_holidays(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    holiday_type: str | None = Query(None),
    is_active: bool | None = Query(None),
    search: str | None = Query(None),
    _: dict = Depends(require_role("operator")),
) -> dict[str, Any]:
    store = get_holiday_store()
    records, total = store.list(
        offset=offset, limit=limit,
        holiday_type=holiday_type,
        is_active=is_active,
        search=search,
    )
    return {
        "holidays": [_record_to_dict(r) for r in records],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@router.post("/", status_code=201)
def create_holiday(
    body: dict[str, Any],
    _: dict = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_holiday_store()
    try:
        record = store.create(
            name=body.get("name", ""),
            date_value=body.get("date_value", ""),
            description=body.get("description"),
            holiday_type=body.get("holiday_type", "national"),
            every_year=body.get("every_year", False),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"holiday": _record_to_dict(record)}


@router.get("/{holiday_id}")
def get_holiday(
    holiday_id: int,
    _: dict = Depends(require_role("operator")),
) -> dict[str, Any]:
    store = get_holiday_store()
    record = store.get(holiday_id)
    if record is None:
        raise HTTPException(404, "Holiday not found")
    return {"holiday": _record_to_dict(record)}


@router.put("/{holiday_id}")
def update_holiday(
    holiday_id: int,
    body: dict[str, Any],
    _: dict = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_holiday_store()
    try:
        record = store.update(
            holiday_id,
            name=body.get("name"),
            date_value=body.get("date_value"),
            description=body.get("description"),
            holiday_type=body.get("holiday_type"),
            every_year=body.get("every_year"),
            is_active=body.get("is_active"),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if record is None:
        raise HTTPException(404, "Holiday not found")
    return {"holiday": _record_to_dict(record)}


@router.delete("/{holiday_id}")
def delete_holiday(
    holiday_id: int,
    _: dict = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_holiday_store()
    deleted = store.delete(holiday_id)
    if not deleted:
        raise HTTPException(404, "Holiday not found")
    return {"deleted": True}


@router.get("/check/{date_value}")
def check_holiday(
    date_value: str,
    _: dict = Depends(require_role("operator")),
) -> dict[str, Any]:
    """Check if a given date is a holiday (supports Jalali input)."""
    from app.core.jalali_utils import parse_jalali_date
    try:
        try:
            d = date.fromisoformat(date_value)
        except (ValueError, TypeError):
            d = parse_jalali_date(date_value)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    store = get_holiday_store()
    is_holiday = store.is_holiday(d)
    return {
        "date": d.isoformat(),
        "is_holiday": is_holiday,
    }


# ── Helpers ───────────────────────────────────────────────────────────


def _record_to_dict(r: HolidayRecord) -> dict[str, Any]:
    return {
        "id": r.id,
        "name": r.name,
        "date_value": r.date_value,
        "description": r.description,
        "holiday_type": r.holiday_type,
        "every_year": r.every_year,
        "is_active": r.is_active,
        "created_at_utc": r.created_at_utc,
        "updated_at_utc": r.updated_at_utc,
    }
