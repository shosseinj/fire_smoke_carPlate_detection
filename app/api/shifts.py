"""API router for Work Shifts — legacy contract takes precedence on colliding routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from app.core.auth import require_role
from app.core.shift_store import (
    WEEKDAY_COLS,
    WEEKDAY_NAMES,
    WorkShiftRecord,
    _is_overnight,
)
from app.core.legacy_service import (
    legacy_shift_response,
    validate_timezone,
    validate_clock_time,
    shift_to_legacy_weekdays,
    legacy_weekdays_to_internal,
)

router = APIRouter(prefix="/api/v1/shifts", tags=["Shifts"])


def get_runtime() -> Any:
    from app.main import runtime
    return runtime


def get_shift_store() -> Any:
    return get_runtime().shift_store


# ── Legacy shift types ──────────────────────────────────────────────────


SHIFT_TYPES = [
    {"value": "morning", "label": "Morning Shift"},
    {"value": "evening", "label": "Evening Shift"},
    {"value": "night", "label": "Night Shift"},
    {"value": "remote", "label": "Remote Work"},
    {"value": "flexible", "label": "Flexible Hours"},
    {"value": "rotating", "label": "Rotating Shifts"},
]


@router.get("/types")
def list_shift_types(
    _: dict = Depends(require_role("admin")),
) -> list[dict[str, str]]:
    return SHIFT_TYPES


@router.get("/statistics")
def shift_statistics(
    _: dict = Depends(require_role("admin")),
) -> dict[str, Any]:
    return get_shift_store().statistics()


@router.get("/")
def list_shifts(
    _: dict = Depends(require_role("admin")),
) -> list[dict[str, Any]]:
    store = get_shift_store()
    records, _ = store.list(offset=0, limit=10000)
    return [_legacy_shift_dict(r, store) for r in records]


@router.get("/{shift_id}")
def get_shift(
    shift_id: int,
    _: dict = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_shift_store()
    record = store.get(shift_id)
    if record is None:
        raise HTTPException(404, "\u0634\u06cc\u0641\u062a \u06cc\u0627\u0641\u062a \u0646\u0634\u062f")
    return _legacy_shift_dict(record, store)


@router.post("/", status_code=201)
def create_shift(
    body: dict[str, Any],
    _: dict = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_shift_store()
    try:
        # Validate time
        start_time = str(body.get("start_time", "08:00"))
        end_time = str(body.get("end_time", "16:00"))
        validate_clock_time(start_time)
        validate_clock_time(end_time)

        # Validate timezone
        tz = str(body.get("timezone_name", "Asia/Tehran"))
        validate_timezone(tz)

        kwargs: dict[str, Any] = {
            "shift_name": str(body.get("shift_name", "")),
            "shift_type": str(body.get("shift_type", "morning")),
            "start_time": start_time,
            "end_time": end_time,
            "timezone_name": tz,
            "max_minutes_delay": int(body.get("max_minutes_delay", 0)),
            "max_minutes_early": int(body.get("max_minutes_early", 0)),
            "max_overtime_hours": float(body.get("max_overtime_hours", 8.0)),
        }
        weekdays = legacy_weekdays_to_internal(body)
        kwargs.update(weekdays)
        record = store.create(**kwargs)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return _legacy_shift_dict(record, store)


@router.put("/{shift_id}")
def update_shift(
    shift_id: int,
    body: dict[str, Any],
    _: dict = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_shift_store()
    try:
        kwargs: dict[str, Any] = {}
        if "shift_name" in body:
            kwargs["shift_name"] = str(body["shift_name"])
        if "shift_type" in body:
            kwargs["shift_type"] = str(body["shift_type"])
        if "start_time" in body:
            validate_clock_time(str(body["start_time"]))
            kwargs["start_time"] = str(body["start_time"])
        if "end_time" in body:
            validate_clock_time(str(body["end_time"]))
            kwargs["end_time"] = str(body["end_time"])
        if "timezone_name" in body:
            validate_timezone(str(body["timezone_name"]))
            kwargs["timezone_name"] = str(body["timezone_name"])
        if "max_minutes_delay" in body:
            kwargs["max_minutes_delay"] = int(body["max_minutes_delay"])
        if "max_minutes_early" in body:
            kwargs["max_minutes_early"] = int(body["max_minutes_early"])
        if "max_overtime_hours" in body:
            kwargs["max_overtime_hours"] = float(body["max_overtime_hours"])
        weekdays = legacy_weekdays_to_internal(body)
        kwargs.update(weekdays)
        record = store.update(shift_id, **kwargs)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if record is None:
        raise HTTPException(404, "\u0634\u06cc\u0641\u062a \u06cc\u0627\u0641\u062a \u0646\u0634\u062f")
    return _legacy_shift_dict(record, store)


@router.delete("/{shift_id}", status_code=204)
def delete_shift(
    shift_id: int,
    force: bool = Query(False),
    _: dict = Depends(require_role("admin")),
) -> Response:
    store = get_shift_store()
    try:
        deleted = store.delete(shift_id, force=force)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if not deleted:
        raise HTTPException(404, "\u0634\u06cc\u0641\u062a \u06cc\u0627\u0641\u062a \u0646\u0634\u062f")
    return Response(status_code=204)


@router.get("/{shift_id}/personnel")
def list_personnel_in_shift(
    shift_id: int,
    _: dict = Depends(require_role("admin")),
) -> list[dict[str, Any]]:
    store = get_shift_store()
    return store.list_personnel_in_shift(shift_id)


# ── Current non-conflicting routes ──────────────────────────────────────


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


@router.get("/statistics/summary")
def shift_statistics_summary(
    _: dict = Depends(require_role("operator")),
) -> dict[str, Any]:
    return get_shift_store().statistics()


# ── Helpers ─────────────────────────────────────────────────────────────


def _legacy_shift_dict(r: WorkShiftRecord, store: Any = None) -> dict[str, Any]:
    count = 0
    if store is not None:
        try:
            count = store.count_personnel_in_shift(r.id)
        except Exception:
            pass
    return legacy_shift_response(r, count)
