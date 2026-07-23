"""API router for Holidays — legacy contract takes precedence on colliding routes."""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import require_role
from app.core.holiday_store import HolidayRecord
from app.core.legacy_service import (
    legacy_holiday_response,
    validate_jalali_date,
    format_jalali,
)

router = APIRouter(prefix="/api/v1/holidays", tags=["Holidays"])


def get_runtime() -> Any:
    from app.main import runtime
    return runtime


def get_holiday_store() -> Any:
    return get_runtime().holiday_store


def _get_current_user_id(current_user: Any) -> int | None:
    if isinstance(current_user, dict):
        return current_user.get("id")
    return getattr(current_user, "id", None)


def _resolve_usernames(store: Any, record: HolidayRecord) -> tuple[str | None, str | None]:
    c_user = None
    u_user = None
    if record.created_by:
        try:
            from app.core.auth_store import UserRecord
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


@router.get("/")
def list_holidays(
    is_active: bool | None = Query(True),
    holiday_type: str | None = Query(None),
    every_year: bool | None = Query(None),
    _: dict = Depends(require_role("operator")),
) -> list[dict[str, Any]]:
    store = get_holiday_store()
    records, _ = store.list(offset=0, limit=10000, holiday_type=holiday_type, is_active=is_active)
    # Filter by every_year
    if every_year is not None:
        records = [r for r in records if r.every_year == every_year]
    # Order by date ascending
    records.sort(key=lambda r: r.date_value)
    return [
        legacy_holiday_response(r, *_resolve_usernames(store, r))
        for r in records
    ]


@router.get("/{holiday_id}")
def get_holiday(
    holiday_id: int,
    _: dict = Depends(require_role("operator")),
) -> dict[str, Any]:
    store = get_holiday_store()
    record = store.get(holiday_id)
    if record is None:
        raise HTTPException(404, "\u062a\u0639\u0637\u06cc\u0644\u06cc \u06cc\u0627\u0641\u062a \u0646\u0634\u062f!")
    return legacy_holiday_response(record, *_resolve_usernames(store, record))


@router.post("/", status_code=201)
def create_holiday(
    body: dict[str, Any],
    current_user: dict = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_holiday_store()
    try:
        date_str = str(body.get("date", body.get("date_value", "")))
        record = store.create(
            name=str(body.get("name", "")),
            date_value=date_str,
            description=body.get("description"),
            holiday_type=str(body.get("holiday_type", "national")),
            every_year=bool(body.get("every_year", False)),
            created_by=_get_current_user_id(current_user),
        )
    except ValueError as exc:
        detail = str(exc)
        if "already exists" in detail:
            return _duplicate_error()
        raise HTTPException(400, detail)
    return legacy_holiday_response(record, *_resolve_usernames(store, record))


@router.patch("/{holiday_id}")
def patch_holiday(
    holiday_id: int,
    body: dict[str, Any],
    current_user: dict = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_holiday_store()
    try:
        kwargs: dict[str, Any] = {}
        if "name" in body:
            kwargs["name"] = str(body["name"])
        if "date" in body or "date_value" in body:
            kwargs["date_value"] = str(body.get("date", body.get("date_value", "")))
        if "description" in body:
            kwargs["description"] = body.get("description")
        if "holiday_type" in body:
            kwargs["holiday_type"] = str(body["holiday_type"])
        if "every_year" in body:
            kwargs["every_year"] = bool(body["every_year"])
        if "is_active" in body:
            kwargs["is_active"] = bool(body["is_active"])
        kwargs["updated_by"] = _get_current_user_id(current_user)
        record = store.update(holiday_id, **kwargs)
    except ValueError as exc:
        detail = str(exc)
        if "already exists" in detail:
            return _duplicate_error()
        raise HTTPException(400, detail)
    if record is None:
        raise HTTPException(404, "\u062a\u0639\u0637\u06cc\u0644\u06cc \u06cc\u0627\u0641\u062a \u0646\u0634\u062f!")
    return legacy_holiday_response(record, *_resolve_usernames(store, record))


@router.delete("/{holiday_id}")
def delete_holiday(
    holiday_id: int,
    current_user: dict = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_holiday_store()
    record = store.get(holiday_id)
    if record is None:
        raise HTTPException(404, "\u062a\u0639\u0637\u06cc\u0644\u06cc \u06cc\u0627\u0641\u062a \u0646\u0634\u062f!")
    store.delete(holiday_id, updated_by=_get_current_user_id(current_user))
    return {
        "message": "\u062a\u0639\u0637\u06cc\u0644\u06cc \u0628\u0627 \u0645\u0648\u0641\u0642\u06cc\u062a \u062d\u0630\u0641 \u0634\u062f.",
        "holiday_id": holiday_id,
    }


# ── Current non-conflicting routes ──────────────────────────────────────


@router.put("/{holiday_id}")
def update_holiday(
    holiday_id: int,
    body: dict[str, Any],
    current_user: dict = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_holiday_store()
    try:
        record = store.update(
            holiday_id,
            name=body.get("name"),
            date_value=body.get("date_value", body.get("date")),
            description=body.get("description"),
            holiday_type=body.get("holiday_type"),
            every_year=body.get("every_year"),
            is_active=body.get("is_active"),
            updated_by=_get_current_user_id(current_user),
        )
    except ValueError as exc:
        if "already exists" in exc:
            return _duplicate_error()
        raise HTTPException(400, str(exc))
    if record is None:
        raise HTTPException(404, "\u062a\u0639\u0637\u06cc\u0644\u06cc \u06cc\u0627\u0641\u062a \u0646\u0634\u062f!")
    return legacy_holiday_response(record, *_resolve_usernames(store, record))


@router.get("/check/{date_value}")
def check_holiday(
    date_value: str,
    _: dict = Depends(require_role("operator")),
) -> dict[str, Any]:
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


# ── Helpers ─────────────────────────────────────────────────────────────


def _duplicate_error() -> HTTPException:
    msg = "\u0627\u06cc\u0646 \u062a\u0639\u0637\u06cc\u0644\u06cc \u062f\u0631 \u0627\u06cc\u0646 \u062a\u0627\u0631\u06cc\u062e \u0642\u0628\u0644\u0627\u064b \u062b\u0628\u062a \u0634\u062f\u0647 \u0627\u0633\u062a!"
    raise HTTPException(400, msg)
