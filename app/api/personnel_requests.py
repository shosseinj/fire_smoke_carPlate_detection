"""API router for Personnel Requests — legacy contract under /api/v1/personnel-requests."""

from __future__ import annotations

import random as _random
from datetime import date, time, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import require_role, get_current_user
from app.core.legacy_service import (
    LEGACY_DURATION_TYPES,
    LEGACY_REQUEST_TYPES,
    calculate_request_duration,
    format_jalali,
    internal_status,
    legacy_request_response,
    legacy_status,
    validate_clock_time,
    validate_jalali_date,
)
from app.core.personnel_store import PersonnelRecord
from app.core.shift_store import WorkShiftRecord

from app.core.request_store import PersonnelRequestRecord

LEGACY_STATUSES = frozenset({"waiting", "accepted", "rejected"})

router = APIRouter(prefix="/api/v1/personnel-requests", tags=["Personnel Requests"])


def get_runtime() -> Any:
    from app.main import runtime
    return runtime


def get_request_store() -> Any:
    return get_runtime().request_store


def get_shift_store() -> Any:
    return get_runtime().shift_store


def get_holiday_store() -> Any:
    return get_runtime().holiday_store


def get_personnel_store() -> Any:
    return get_runtime().personnel_store


def _get_user_id(current_user: Any) -> int:
    return getattr(current_user, "id", 0) or 0


def _get_personnel(personnel_id: int) -> PersonnelRecord | None:
    return get_personnel_store().get(personnel_id)


def _get_shift(personnel: PersonnelRecord) -> WorkShiftRecord | None:
    if personnel.shift_id is None:
        return None
    return get_shift_store().get(personnel.shift_id)


def _full_name(personnel: PersonnelRecord) -> str:
    return f"{personnel.fname} {personnel.lname}"


# ── List ──────────────────────────────────────────────────────────────────


@router.get("/")
def list_requests(
    personnel_id: int | None = Query(None),
    request_type: str | None = Query(None),
    status: str | None = Query(None),
    start_date_from: str | None = Query(None),
    start_date_to: str | None = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    _: Any = Depends(require_role("operator")),
) -> list[dict[str, Any]]:
    store = get_request_store()
    int_status = internal_status(status) if status else None
    records, _ = store.list(
        offset=skip,
        limit=limit,
        personnel_id=personnel_id,
        request_type=request_type,
        status=int_status,
        start_date_from=start_date_from,
        start_date_to=start_date_to,
    )
    result: list[dict[str, Any]] = []
    for r in records:
        name = None
        p = _get_personnel(r.personnel_id)
        if p:
            name = _full_name(p)
        result.append(legacy_request_response(r, full_name=name))
    return result


# ── My requests ───────────────────────────────────────────────────────────


@router.get("/my-requests")
def get_my_requests(
    status: str | None = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: Any = Depends(get_current_user),
) -> list[dict[str, Any]]:
    store = get_request_store()
    int_status = internal_status(status) if status else None
    records, _ = store.list(
        offset=skip,
        limit=limit,
        status=int_status,
    )
    result: list[dict[str, Any]] = []
    for r in records:
        name = None
        p = _get_personnel(r.personnel_id)
        if p:
            name = _full_name(p)
        result.append(legacy_request_response(r, full_name=name))
    return result


# ── Calculate time ────────────────────────────────────────────────────────


@router.post("/calculate-time")
def calculate_time(
    body: dict[str, Any],
    _: Any = Depends(require_role("operator")),
) -> dict[str, Any]:
    personnel_id = int(body.get("personnel_id", 0))
    request_type = str(body.get("request_type", "earned_leave"))
    duration_type = str(body.get("duration_type", "daily"))
    start_date_str = str(body.get("start_date", ""))
    end_date_str = str(body.get("end_date", ""))
    start_time_str = body.get("start_time")
    end_time_str = body.get("end_time")

    if request_type not in LEGACY_REQUEST_TYPES:
        raise HTTPException(400, f"نوع درخواست معتبر نیست: {request_type!r}")
    if duration_type not in LEGACY_DURATION_TYPES:
        raise HTTPException(400, f"نوع مدت معتبر نیست: {duration_type!r}")

    personnel = _get_personnel(personnel_id)
    if personnel is None:
        raise HTTPException(404, f"پرسنل با شناسه {personnel_id} یافت نشد")
    shift = _get_shift(personnel)
    if shift is None:
        raise HTTPException(400, f"برای پرسنل با شناسه {personnel_id} شیفتی تعیین نشده است")

    try:
        s = validate_jalali_date(start_date_str)
        e = validate_jalali_date(end_date_str) if end_date_str else s
        if e < s:
            raise ValueError("تاریخ پایان نمی‌تواند قبل از تاریخ شروع باشد")
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    st: time | None = None
    et: time | None = None

    if duration_type == "daily":
        if start_time_str or end_time_str:
            raise HTTPException(400, "درخواست روزانه نباید شامل start_time یا end_time باشد")
    elif duration_type == "hourly":
        if not start_time_str or not end_time_str:
            raise HTTPException(400, "درخواست ساعتی به start_time و end_time نیاز دارد")
        try:
            validate_clock_time(start_time_str)
            validate_clock_time(end_time_str)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        st_h, st_m = int(start_time_str.split(":")[0]), int(start_time_str.split(":")[1])
        et_h, et_m = int(end_time_str.split(":")[0]), int(end_time_str.split(":")[1])
        st = time(st_h, st_m)
        et = time(et_h, et_m)
        if s != e:
            raise HTTPException(400, "تاریخ شروع و پایان درخواست ساعتی باید یکسان باشد")
        if et <= st:
            raise HTTPException(400, "end_time باید بعد از start_time باشد")

    calc = calculate_request_duration(
        personnel, shift, get_holiday_store(), s, e, duration_type, st, et,
    )

    return {
        "personnel_id": personnel_id,
        "request_type": request_type,
        "duration_type": duration_type,
        "start_date": format_jalali(s),
        "end_date": format_jalali(e),
        "start_time": start_time_str if duration_type == "hourly" else None,
        "end_time": end_time_str if duration_type == "hourly" else None,
        "duration_days": calc["duration_days"] if duration_type == "daily" else None,
        "duration_minutes": calc["duration_minutes"] if duration_type == "hourly" else None,
        "duration_hours": calc["duration_hours"] if duration_type == "hourly" else None,
        "working_dates": calc["working_dates"],
        "excluded_non_working_dates": calc["excluded_non_working_dates"],
        "excluded_holiday_dates": calc["excluded_holiday_dates"],
    }


# ── Stats: my ─────────────────────────────────────────────────────────────


@router.get("/stats/my")
def get_my_request_stats(
    current_user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    store = get_request_store()
    by_status = store.count_by_status()
    total = sum(by_status.values())
    waiting = by_status.get("pending", 0)
    accepted = by_status.get("approved", 0)
    rejected = by_status.get("rejected", 0)
    return {
        "total": total,
        "pending": waiting,
        "accepted": accepted,
        "rejected": rejected,
    }


# ── Stats: admin ──────────────────────────────────────────────────────────


@router.get("/stats/admin")
def admin_statistics(
    _: Any = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_request_store()
    by_status = store.count_by_status()
    by_type = store.count_by_type()
    total = sum(by_status.values())
    pending = by_status.get("pending", 0)
    accepted = by_status.get("approved", 0)
    rejected = by_status.get("rejected", 0)

    type_map: dict[str, int] = {}
    for t in ["earned_leave", "sick_leave", "unpaid_leave", "mission", "overtime"]:
        type_map[t] = by_type.get(t, 0)

    return {
        "total": total,
        "pending": pending,
        "accepted": accepted,
        "rejected": rejected,
        "by_type": type_map,
    }


# ── Create ────────────────────────────────────────────────────────────────


@router.post("/", status_code=201)
def create_request(
    body: dict[str, Any],
    _: Any = Depends(require_role("operator")),
) -> dict[str, Any]:
    store = get_request_store()
    personnel_id = int(body.get("personnel_id", 0))
    request_type = str(body.get("request_type", "earned_leave"))
    duration_type = str(body.get("duration_type", "daily"))
    start_date_str = str(body.get("start_date", ""))
    end_date_str = str(body.get("end_date", ""))
    start_time_str = body.get("start_time")
    end_time_str = body.get("end_time")
    description = body.get("description")

    if request_type not in LEGACY_REQUEST_TYPES:
        raise HTTPException(400, f"نوع درخواست معتبر نیست: {request_type!r}")
    if duration_type not in LEGACY_DURATION_TYPES:
        raise HTTPException(400, f"نوع مدت معتبر نیست: {duration_type!r}")

    personnel = _get_personnel(personnel_id)
    if personnel is None:
        raise HTTPException(404, f"پرسنل با شناسه {personnel_id} یافت نشد")
    shift = _get_shift(personnel)
    if shift is None:
        raise HTTPException(400, f"برای پرسنل با شناسه {personnel_id} شیفتی تعیین نشده است")

    try:
        s = validate_jalali_date(start_date_str)
        e = validate_jalali_date(end_date_str) if end_date_str else s
        if e < s:
            raise ValueError("تاریخ پایان نمی‌تواند قبل از تاریخ شروع باشد")
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    st: time | None = None
    et: time | None = None

    if duration_type == "daily":
        if start_time_str or end_time_str:
            raise HTTPException(400, "درخواست روزانه نباید شامل start_time یا end_time باشد")
    elif duration_type == "hourly":
        if not start_time_str or not end_time_str:
            raise HTTPException(400, "درخواست ساعتی به start_time و end_time نیاز دارد")
        try:
            validate_clock_time(start_time_str)
            validate_clock_time(end_time_str)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        st_h, st_m = int(start_time_str.split(":")[0]), int(start_time_str.split(":")[1])
        et_h, et_m = int(end_time_str.split(":")[0]), int(end_time_str.split(":")[1])
        st = time(st_h, st_m)
        et = time(et_h, et_m)
        if s != e:
            raise HTTPException(400, "تاریخ شروع و پایان درخواست ساعتی باید یکسان باشد")
        if et <= st:
            raise HTTPException(400, "end_time باید بعد از start_time باشد")

    calc = calculate_request_duration(
        personnel, shift, get_holiday_store(), s, e, duration_type, st, et,
    )

    try:
        record = store.create(
            personnel_id=personnel_id,
            request_type=request_type,
            start_date=s.isoformat(),
            end_date=e.isoformat(),
            duration_type=duration_type,
            start_time=st.strftime("%H:%M") if st else None,
            end_time=et.strftime("%H:%M") if et else None,
            duration_days=calc["duration_days"] if duration_type == "daily" else None,
            duration_minutes=calc["duration_minutes"] if duration_type == "hourly" else None,
            reason=description,
            status="approved",
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    return legacy_request_response(record, full_name=_full_name(personnel))


# ── Generate fake (admin only) ────────────────────────────────────────────


@router.post("/generate-fake", status_code=201)
def generate_fake_requests(
    count: int = Query(10, ge=1, le=50, description="تعداد درخواست آزمایشی برای ایجاد"),
    admin_user: Any = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_request_store()
    personnel_store = get_personnel_store()
    all_personnel, _ = personnel_store.list(limit=1000)
    personnel_with_shift = [p for p in all_personnel if p.shift_id is not None]
    if not personnel_with_shift:
        raise HTTPException(404, "هیچ پرسنل دارای شیفتی یافت نشد")

    created = 0
    request_types_list = list(LEGACY_REQUEST_TYPES)
    start_date = date.today() - timedelta(days=180)

    for _ in range(count):
        person = _random.choice(personnel_with_shift)
        duration_type = _random.choice(list(LEGACY_DURATION_TYPES))
        req_type = _random.choice(request_types_list)
        request_start = start_date + timedelta(days=_random.randint(0, 180))

        if duration_type == "daily":
            request_end = request_start + timedelta(days=_random.randint(0, 10))
            st = None
            et = None
            st_str = None
            et_str = None
        else:
            request_end = request_start
            st_h = _random.randint(8, 14)
            st_m = _random.choice([0, 15, 30, 45])
            st = time(st_h, st_m)
            et = time(st_h + _random.randint(1, 3), st_m)
            st_str = st.strftime("%H:%M")
            et_str = et.strftime("%H:%M")

        shift = _get_shift(person)
        if shift is None:
            continue

        try:
            calc = calculate_request_duration(
                person, shift, get_holiday_store(),
                request_start, request_end, duration_type, st, et,
            )
        except (ValueError, Exception):
            continue

        int_status = _random.choice(["pending", "approved", "rejected"])
        try:
            store.create(
                personnel_id=person.id,
                request_type=req_type,
                start_date=request_start.isoformat(),
                end_date=request_end.isoformat(),
                duration_type=duration_type,
                start_time=st_str,
                end_time=et_str,
                duration_days=calc["duration_days"] if duration_type == "daily" else None,
                duration_minutes=calc["duration_minutes"] if duration_type == "hourly" else None,
                reason=None,
                status=int_status,
            )
            created += 1
        except (ValueError, Exception):
            continue

    return {"message": f"{created} درخواست آزمایشی ایجاد شد", "count": created}


# ── Detail ────────────────────────────────────────────────────────────────


@router.get("/{request_id}")
def get_request(
    request_id: int,
    _: Any = Depends(require_role("operator")),
) -> dict[str, Any]:
    store = get_request_store()
    record = store.get(request_id)
    if record is None:
        raise HTTPException(404, "درخواست یافت نشد")
    name = None
    p = _get_personnel(record.personnel_id)
    if p:
        name = _full_name(p)
    return legacy_request_response(record, full_name=name)


# ── PATCH (update status) ────────────────────────────────────────────────


@router.patch("/{request_id}")
def patch_request(
    request_id: int,
    body: dict[str, Any],
    current_user: Any = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_request_store()
    record = store.get(request_id)
    if record is None:
        raise HTTPException(404, "درخواست یافت نشد")

    status_str = body.get("status")
    if status_str is not None:
        if status_str not in LEGACY_STATUSES:
            raise HTTPException(400, f"وضعیت معتبر نیست: {status_str!r}")
        int_status = internal_status(status_str)
    else:
        int_status = record.status

    rejection_reason = body.get("rejection_reason")
    admin_notes = body.get("admin_notes")
    user_id = _get_user_id(current_user)

    try:
        updated = store.update_status(
            request_id,
            status=int_status,
            reviewed_by=user_id,
            rejection_reason=rejection_reason if int_status == "rejected" else None,
            admin_notes=admin_notes,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    name = None
    p = _get_personnel(record.personnel_id)
    if p:
        name = _full_name(p)
    return legacy_request_response(updated, full_name=name)


# ── Approve ────────────────────────────────────────────────────────────────


@router.patch("/{request_id}/approve")
def approve_request(
    request_id: int,
    admin_notes: str | None = Query(None),
    current_user: Any = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_request_store()
    record = store.get(request_id)
    if record is None:
        raise HTTPException(404, "درخواست یافت نشد")

    if record.status != "pending":
        raise HTTPException(400, "درخواست در وضعیت انتظار نیست")

    try:
        updated = store.update_status(
            request_id,
            status="approved",
            reviewed_by=_get_user_id(current_user),
            admin_notes=admin_notes,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    name = None
    p = _get_personnel(record.personnel_id)
    if p:
        name = _full_name(p)
    return legacy_request_response(updated, full_name=name)


# ── Reject ────────────────────────────────────────────────────────────────


@router.patch("/{request_id}/reject")
def reject_request(
    request_id: int,
    rejection_reason: str = Query(..., min_length=1),
    admin_notes: str | None = Query(None),
    current_user: Any = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_request_store()
    record = store.get(request_id)
    if record is None:
        raise HTTPException(404, "درخواست یافت نشد")

    if record.status != "pending":
        raise HTTPException(400, "درخواست در وضعیت انتظار نیست")

    try:
        updated = store.update_status(
            request_id,
            status="rejected",
            reviewed_by=_get_user_id(current_user),
            rejection_reason=rejection_reason,
            admin_notes=admin_notes,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    name = None
    p = _get_personnel(record.personnel_id)
    if p:
        name = _full_name(p)
    return legacy_request_response(updated, full_name=name)


# ── Delete ────────────────────────────────────────────────────────────────


@router.delete("/{request_id}")
def delete_request(
    request_id: int,
    _: Any = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_request_store()
    record = store.get(request_id)
    if record is None:
        raise HTTPException(404, "درخواست یافت نشد")
    store.delete(request_id)
    return {"message": "درخواست با موفقیت حذف شد"}
