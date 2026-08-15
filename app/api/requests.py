"""API router for Personnel Requests."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import require_permission
from app.core.request_store import PersonnelRequestRecord, VALID_REQUEST_STATUSES, VALID_REQUEST_TYPES

router = APIRouter(prefix="/api/v1/requests", tags=["Personnel Requests"])


def get_runtime() -> Any:
    from app.main import runtime
    return runtime


def get_request_store() -> Any:
    return get_runtime().request_store


@router.get("/")
def list_requests(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    personnel_id: int | None = Query(None),
    request_type: str | None = Query(None),
    status: str | None = Query(None),
    _: dict = Depends(require_permission("requests.read")),
) -> dict[str, Any]:
    store = get_request_store()
    records, total = store.list(
        offset=offset, limit=limit,
        personnel_id=personnel_id,
        request_type=request_type,
        status=status,
    )
    return {
        "requests": [_record_to_dict(r) for r in records],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@router.post("/", status_code=201)
def create_request(
    body: dict[str, Any],
    current_user: dict = Depends(require_permission("requests.read")),
) -> dict[str, Any]:
    store = get_request_store()
    try:
        record = store.create(
            personnel_id=body["personnel_id"],
            request_type=body.get("request_type", "leave"),
            start_date=body.get("start_date", ""),
            end_date=body.get("end_date", ""),
            reason=body.get("reason"),
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc))
    return {"request": _record_to_dict(record)}


@router.get("/{request_id}")
def get_request(
    request_id: int,
    _: dict = Depends(require_permission("requests.read")),
) -> dict[str, Any]:
    store = get_request_store()
    record = store.get(request_id)
    if record is None:
        raise HTTPException(404, "درخواست یافت نشد")
    return {"request": _record_to_dict(record)}


@router.post("/{request_id}/approve")
def approve_request(
    request_id: int,
    body: dict[str, Any],
    current_user: dict = Depends(require_permission("requests.approve")),
) -> dict[str, Any]:
    store = get_request_store()
    try:
        user_id = current_user.get("id") or 0
        rejection_reason = body.get("rejection_reason")
        if rejection_reason:
            record = store.reject(request_id, user_id, rejection_reason)
        else:
            record = store.approve(request_id, user_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if record is None:
        raise HTTPException(404, "درخواست یافت نشد")
    return {"request": _record_to_dict(record)}


@router.post("/{request_id}/cancel")
def cancel_request(
    request_id: int,
    _: dict = Depends(require_permission("requests.read")),
) -> dict[str, Any]:
    store = get_request_store()
    try:
        record = store.cancel(request_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if record is None:
        raise HTTPException(404, "درخواست یافت نشد")
    return {"request": _record_to_dict(record)}


@router.delete("/{request_id}")
def delete_request(
    request_id: int,
    _: dict = Depends(require_permission("requests.approve")),
) -> dict[str, Any]:
    store = get_request_store()
    deleted = store.delete(request_id)
    if not deleted:
        raise HTTPException(404, "درخواست یافت نشد")
    return {"deleted": True}


@router.get("/types/list")
def list_request_types(
    _: dict = Depends(require_permission("requests.read")),
) -> dict[str, Any]:
    return {"types": sorted(VALID_REQUEST_TYPES)}


@router.get("/statuses/list")
def list_request_statuses(
    _: dict = Depends(require_permission("requests.read")),
) -> dict[str, Any]:
    return {"statuses": sorted(VALID_REQUEST_STATUSES)}


# ── Helpers ───────────────────────────────────────────────────────────


def _record_to_dict(r: PersonnelRequestRecord) -> dict[str, Any]:
    from app.core.jalali_utils import utc_iso_to_jalali_datetime
    return {
        "id": r.id,
        "personnel_id": r.personnel_id,
        "request_type": r.request_type,
        "start_date": r.start_date,
        "end_date": r.end_date,
        "reason": r.reason,
        "status": r.status,
        "approved_by": r.approved_by,
        "rejection_reason": r.rejection_reason,
        "created_at_utc": r.created_at_utc,
        "updated_at_utc": r.updated_at_utc,
        "created_at_jalali": utc_iso_to_jalali_datetime(r.created_at_utc) or "",
        "updated_at_jalali": utc_iso_to_jalali_datetime(r.updated_at_utc),
    }
