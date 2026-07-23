from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.holiday_schemas import HolidayCreate, HolidayResponse, HolidayUpdate
from app.core.auth import require_role
from app.core.holiday_store import HolidayRecord

router = APIRouter(prefix="/api/v1/holidays", tags=["Holidays"])


def get_runtime() -> Any:
    from app.main import runtime
    return runtime


def get_holiday_store() -> Any:
    return get_runtime().holiday_store


def _record_to_response(
    record: HolidayRecord,
    created_by_username: str | None = None,
    updated_by_username: str | None = None,
) -> HolidayResponse:
    parsed_date = date.fromisoformat(record.date_value) if isinstance(record.date_value, str) else record.date_value
    from app.core.jalali_utils import parse_jalali_date
    from app.core.legacy_service import format_jalali
    return HolidayResponse(
        id=record.id,
        name=record.name,
        date=parsed_date,
        description=record.description,
        holiday_type=record.holiday_type,
        every_year=record.every_year,
        is_active=record.is_active,
        created_at=record.created_at_utc,
        updated_at=record.updated_at_utc,
        created_by=record.created_by,
        updated_by=record.updated_by,
        created_by_username=created_by_username,
        updated_by_username=updated_by_username,
    )


def _resolve_usernames(store: Any, record: HolidayRecord) -> tuple[str | None, str | None]:
    c_user = None
    u_user = None
    if record.created_by:
        try:
            u = store.database.connection().execute(
                "SELECT username FROM users WHERE id = ?", (record.created_by,)
            ).fetchone()
            if u:
                c_user = u["username"]
        except Exception:
            pass
    if record.updated_by:
        try:
            u = store.database.connection().execute(
                "SELECT username FROM users WHERE id = ?", (record.updated_by,)
            ).fetchone()
            if u:
                u_user = u["username"]
        except Exception:
            pass
    return c_user, u_user


@router.get("/", response_model=list[HolidayResponse])
def list_holidays(
    is_active: bool | None = Query(True),
    holiday_type: str | None = Query(None),
    every_year: bool | None = Query(None),
    _: dict = Depends(require_role("operator")),
) -> list[HolidayResponse]:
    store = get_holiday_store()
    records, _ = store.list(offset=0, limit=10000, holiday_type=holiday_type, is_active=is_active)
    if every_year is not None:
        records = [r for r in records if r.every_year == every_year]
    records.sort(key=lambda r: r.date_value)
    return [_record_to_response(r, *_resolve_usernames(store, r)) for r in records]


@router.get("/{holiday_id}", response_model=HolidayResponse)
def get_holiday(
    holiday_id: int,
    _: dict = Depends(require_role("operator")),
) -> HolidayResponse:
    store = get_holiday_store()
    record = store.get(holiday_id)
    if record is None:
        raise HTTPException(404, "\u062a\u0639\u0637\u06cc\u0644\u06cc \u06cc\u0627\u0641\u062a \u0646\u0634\u062f!")
    return _record_to_response(record, *_resolve_usernames(store, record))


@router.post("/", response_model=HolidayResponse, status_code=201)
def create_holiday(
    body: HolidayCreate,
    current_user: dict = Depends(require_role("admin")),
) -> HolidayResponse:
    store = get_holiday_store()
    current_user_id: int | None = body.id if isinstance(body, dict) else None
    if isinstance(current_user, dict):
        current_user_id = current_user.get("id")
    else:
        current_user_id = getattr(current_user, "id", None)
    try:
        record = store.create(
            name=body.name,
            date_value=body.date.isoformat(),
            description=body.description,
            holiday_type=body.holiday_type,
            every_year=body.every_year,
            created_by=current_user_id,
        )
    except ValueError as exc:
        detail = str(exc)
        if "already exists" in detail:
            raise HTTPException(400, "\u0627\u06cc\u0646 \u062a\u0639\u0637\u06cc\u0644\u06cc \u062f\u0631 \u0627\u06cc\u0646 \u062a\u0627\u0631\u06cc\u062e \u0642\u0628\u0644\u0627\u064b \u062b\u0628\u062a \u0634\u062f\u0647 \u0627\u0633\u062a!")
        raise HTTPException(400, detail)
    return _record_to_response(record, *_resolve_usernames(store, record))


@router.patch("/{holiday_id}", response_model=HolidayResponse)
def patch_holiday(
    holiday_id: int,
    body: HolidayUpdate,
    current_user: dict = Depends(require_role("admin")),
) -> HolidayResponse:
    store = get_holiday_store()
    current_user_id: int | None = None
    if isinstance(current_user, dict):
        current_user_id = current_user.get("id")
    else:
        current_user_id = getattr(current_user, "id", None)
    try:
        kwargs: dict[str, Any] = {}
        if body.name is not None:
            kwargs["name"] = body.name
        if body.date is not None:
            kwargs["date_value"] = body.date.isoformat()
        if body.description is not None:
            kwargs["description"] = body.description
        if body.holiday_type is not None:
            kwargs["holiday_type"] = body.holiday_type
        if body.every_year is not None:
            kwargs["every_year"] = body.every_year
        if body.is_active is not None:
            kwargs["is_active"] = body.is_active
        kwargs["updated_by"] = current_user_id
        record = store.update(holiday_id, **kwargs)
    except ValueError as exc:
        detail = str(exc)
        if "already exists" in detail:
            raise HTTPException(400, "\u0627\u06cc\u0646 \u062a\u0639\u0637\u06cc\u0644\u06cc \u062f\u0631 \u0627\u06cc\u0646 \u062a\u0627\u0631\u06cc\u062e \u0642\u0628\u0644\u0627\u064b \u062b\u0628\u062a \u0634\u062f\u0647 \u0627\u0633\u062a!")
        raise HTTPException(400, detail)
    if record is None:
        raise HTTPException(404, "\u062a\u0639\u0637\u06cc\u0644\u06cc \u06cc\u0627\u0641\u062a \u0646\u0634\u062f!")
    return _record_to_response(record, *_resolve_usernames(store, record))


@router.delete("/{holiday_id}", response_model=dict)
def delete_holiday(
    holiday_id: int,
    current_user: dict = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_holiday_store()
    record = store.get(holiday_id)
    if record is None:
        raise HTTPException(404, "\u062a\u0639\u0637\u06cc\u0644\u06cc \u06cc\u0627\u0641\u062a \u0646\u0634\u062f!")
    store.delete(holiday_id)
    return {
        "message": "\u062a\u0639\u0637\u06cc\u0644\u06cc \u0628\u0627 \u0645\u0648\u0641\u0642\u06cc\u062a \u062d\u0630\u0641 \u0634\u062f.",
        "holiday_id": holiday_id,
    }