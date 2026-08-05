"""API router for Personnel Requests — legacy contract under /api/v1/personnel-requests."""

from __future__ import annotations

import random as _random
from datetime import date, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.config import settings
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
from app.core.jalali_utils import local_date_range_bounds_utc, parse_jalali_date
from app.time_utils import utc_now_text

LEGACY_STATUSES = frozenset({"waiting", "accepted", "rejected"})

router = APIRouter(prefix="/api/v1/personnel-requests", tags=["Personnel Requests"])


class BulkPersonnelRequestItem(BaseModel):
    request_type: str = "earned_leave"
    duration_type: str = "daily"
    start_date: str = Field(min_length=1)
    end_date: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    description: str | None = None


class BulkPersonnelRequestCreate(BaseModel):
    personnel_id: int = Field(gt=0)
    requests: list[BulkPersonnelRequestItem] = Field(min_length=1, max_length=200)


class PersonnelRequestCreate(BaseModel):
    personnel_id: int = Field(gt=0)
    request_type: str = "earned_leave"
    duration_type: str = "daily"
    start_date: str = Field(min_length=1)
    end_date: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    description: str | None = None


class PersonnelRequestResponse(BaseModel):
    id: int
    personnel_id: int
    full_name: str | None
    request_type: str
    duration_type: str | None
    start_date: str
    end_date: str
    start_time: str | None
    end_time: str | None
    description: str | None
    duration_days: float | None
    duration_minutes: int | None
    duration_time: str | None
    status: str
    admin_notes: str | None
    rejection_reason: str | None
    created_at: str | None
    updated_at: str | None
    reviewed_by: int | None
    reviewed_at: str | None


class PersonnelRequestCreateResponse(PersonnelRequestResponse):
    removed_logs_count: int


class BulkPersonnelRequestResponse(BaseModel):
    personnel_id: int
    count: int
    requests: list[PersonnelRequestResponse]


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


def get_detection_log_store() -> Any:
    return get_runtime().detection_log_store


def _remove_detection_logs_for_daily_request(
    personnel_id: int,
    start_date: date,
    end_date: date,
) -> int:
    local_tz = ZoneInfo(settings.business_timezone_name)
    utc_start, utc_end = local_date_range_bounds_utc(
        start_date,
        end_date,
        local_tz,
    )
    return len(
        get_detection_log_store().delete_personnel_in_time_range(
            personnel_id,
            utc_start.isoformat(),
            utc_end.isoformat(),
        )
    )


def _get_user_id(current_user: Any) -> int:
    return getattr(current_user, "id", 0) or 0


def _get_personnel(personnel_id: int) -> PersonnelRecord | None:
    return get_personnel_store().get(personnel_id)


def _get_shift(
    personnel: PersonnelRecord,
    on_date: date | None = None,
) -> WorkShiftRecord | None:
    store = get_shift_store()
    if on_date is not None:
        assignment = store.get_assignment_for_date(personnel.id, on_date)
        return store.get(assignment.shift_id) if assignment is not None else None
    if personnel.shift_id is None:
        return None
    return store.get(personnel.shift_id)


def _calculate_request_duration_for_assignments(
    personnel: PersonnelRecord,
    start: date,
    end: date,
    duration_type: str,
    start_clock: time | None,
    end_clock: time | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "working_dates": [],
        "excluded_non_working_dates": [],
        "excluded_holiday_dates": [],
        "duration_days": 0.0,
        "duration_minutes": 0 if duration_type == "hourly" else None,
        "duration_hours": 0.0 if duration_type == "hourly" else None,
    }
    current = start
    while current <= end:
        shift = _get_shift(personnel, current)
        if shift is None:
            raise HTTPException(
                400,
                f"برای تاریخ {format_jalali(current)} شیفتی برای پرسنل تعیین نشده است",
            )
        day_result = calculate_request_duration(
            personnel,
            shift,
            get_holiday_store(),
            current,
            current,
            duration_type,
            start_clock,
            end_clock,
        )
        for key in (
            "working_dates",
            "excluded_non_working_dates",
            "excluded_holiday_dates",
        ):
            result[key].extend(day_result[key])
        result["duration_days"] += day_result["duration_days"]
        if duration_type == "hourly":
            result["duration_minutes"] += day_result["duration_minutes"] or 0
        current += timedelta(days=1)
    if duration_type == "hourly":
        result["duration_hours"] = round(result["duration_minutes"] / 60, 4)
    return result


def _full_name(personnel: PersonnelRecord) -> str:
    return f"{personnel.fname} {personnel.lname}"


def _prepare_bulk_request(
    personnel: PersonnelRecord,
    item: BulkPersonnelRequestItem,
) -> dict[str, Any]:
    request_type = item.request_type
    duration_type = item.duration_type
    if request_type not in LEGACY_REQUEST_TYPES:
        raise HTTPException(400, f"نوع درخواست معتبر نیست: {request_type!r}")
    if duration_type not in LEGACY_DURATION_TYPES:
        raise HTTPException(400, f"نوع مدت معتبر نیست: {duration_type!r}")

    try:
        start = validate_jalali_date(item.start_date)
        end = validate_jalali_date(item.end_date) if item.end_date else start
        if end < start:
            raise ValueError("تاریخ پایان نمی‌تواند قبل از تاریخ شروع باشد")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    start_clock: time | None = None
    end_clock: time | None = None
    if duration_type == "daily":
        if item.start_time or item.end_time:
            raise HTTPException(
                400,
                "درخواست روزانه نباید شامل start_time یا end_time باشد",
            )
    else:
        if not item.start_time or not item.end_time:
            raise HTTPException(
                400,
                "درخواست ساعتی به start_time و end_time نیاز دارد",
            )
        try:
            validate_clock_time(item.start_time)
            validate_clock_time(item.end_time)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        start_hour, start_minute = map(int, item.start_time.split(":")[:2])
        end_hour, end_minute = map(int, item.end_time.split(":")[:2])
        start_clock = time(start_hour, start_minute)
        end_clock = time(end_hour, end_minute)
        if start != end:
            raise HTTPException(
                400,
                "تاریخ شروع و پایان درخواست ساعتی باید یکسان باشد",
            )
        if end_clock <= start_clock:
            raise HTTPException(400, "end_time باید بعد از start_time باشد")

    try:
        calculation = _calculate_request_duration_for_assignments(
            personnel,
            start,
            end,
            duration_type,
            start_clock,
            end_clock,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "request_type": request_type,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "duration_type": duration_type,
        "start_time": start_clock.strftime("%H:%M") if start_clock else None,
        "end_time": end_clock.strftime("%H:%M") if end_clock else None,
        "duration_days": (
            calculation["duration_days"] if duration_type == "daily" else None
        ),
        "duration_minutes": (
            calculation["duration_minutes"] if duration_type == "hourly" else None
        ),
        "reason": item.description,
        "status": "approved",
    }


# ── List ──────────────────────────────────────────────────────────────────


@router.get("/", response_model=list[PersonnelRequestResponse])
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
        include_total=False,
    )
    names = get_personnel_store().get_full_names_by_ids(
        {record.personnel_id for record in records}
    )
    result: list[dict[str, Any]] = []
    for r in records:
        result.append(
            legacy_request_response(r, full_name=names.get(r.personnel_id))
        )
    return result


# ── My requests ───────────────────────────────────────────────────────────


@router.get("/my-requests", response_model=list[PersonnelRequestResponse])
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
        include_total=False,
    )
    names = get_personnel_store().get_full_names_by_ids(
        {record.personnel_id for record in records}
    )
    result: list[dict[str, Any]] = []
    for r in records:
        result.append(
            legacy_request_response(r, full_name=names.get(r.personnel_id))
        )
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

    calc = _calculate_request_duration_for_assignments(
        personnel, s, e, duration_type, st, et,
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


@router.post(
    "/",
    status_code=201,
    response_model=PersonnelRequestCreateResponse,
)
def create_request(
    body: PersonnelRequestCreate,
    remove_logs_in_request_dates: bool = False,
    _: Any = Depends(require_role("operator")),
) -> dict[str, Any]:
    store = get_request_store()
    personnel_id = body.personnel_id
    request_type = body.request_type
    duration_type = body.duration_type
    start_date_str = body.start_date
    end_date_str = body.end_date
    start_time_str = body.start_time
    end_time_str = body.end_time
    description = body.description

    if request_type not in LEGACY_REQUEST_TYPES:
        raise HTTPException(400, f"نوع درخواست معتبر نیست: {request_type!r}")
    if duration_type not in LEGACY_DURATION_TYPES:
        raise HTTPException(400, f"نوع مدت معتبر نیست: {duration_type!r}")

    personnel = _get_personnel(personnel_id)
    if personnel is None:
        raise HTTPException(404, f"پرسنل با شناسه {personnel_id} یافت نشد")
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

    calc = _calculate_request_duration_for_assignments(
        personnel, s, e, duration_type, st, et,
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

    removed_logs_count = 0
    if remove_logs_in_request_dates and duration_type == "daily":
        removed_logs_count = _remove_detection_logs_for_daily_request(
            personnel_id,
            s,
            e,
        )

    response = legacy_request_response(record, full_name=_full_name(personnel))
    response["removed_logs_count"] = removed_logs_count
    return response


# ── Generate fake (admin only) ────────────────────────────────────────────


@router.post(
    "/bulk",
    status_code=201,
    response_model=BulkPersonnelRequestResponse,
)
def create_bulk_requests(
    body: BulkPersonnelRequestCreate,
    _: Any = Depends(require_role("operator")),
) -> dict[str, Any]:
    """Create an atomic batch of requests for one personnel."""
    personnel = _get_personnel(body.personnel_id)
    if personnel is None:
        raise HTTPException(404, f"پرسنل با شناسه {body.personnel_id} یافت نشد")
    prepared = [
        _prepare_bulk_request(personnel, item) for item in body.requests
    ]
    try:
        records = get_request_store().create_many(
            personnel_id=body.personnel_id,
            requests=prepared,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    full_name = _full_name(personnel)
    return {
        "personnel_id": body.personnel_id,
        "count": len(records),
        "requests": [
            legacy_request_response(record, full_name=full_name)
            for record in records
        ],
    }


@router.post("/generate-fake", status_code=201)
def generate_fake_requests(
    count: int = Query(10, ge=1, description="تعداد درخواست آزمایشی برای ایجاد"),
    from_date: str | None = Query(None, description="تاریخ شروع شمسی YYYY-MM-DD"),
    to_date: str | None = Query(None, description="تاریخ پایان شمسی YYYY-MM-DD"),
    remove_logs_in_request_dates: bool = Query(
        True,
        description="حذف لاگ‌های همان پرسنل در تاریخ درخواست‌های روزانه ایجادشده",
    ),
    admin_user: Any = Depends(require_role("admin")),
) -> dict[str, Any]:
    if bool(from_date) != bool(to_date):
        raise HTTPException(400, "from_date و to_date باید با هم ارسال شوند")
    if from_date and to_date:
        try:
            generation_start = parse_jalali_date(from_date)
            generation_end = parse_jalali_date(to_date)
        except ValueError as exc:
            raise HTTPException(
                400,
                "فرمت تاریخ نامعتبر است (شمسی: YYYY-MM-DD)",
            ) from exc
        if generation_start > generation_end:
            raise HTTPException(400, "تاریخ شروع نباید بعد از تاریخ پایان باشد")
    else:
        generation_end = date.today()
        generation_start = generation_end - timedelta(days=180)

    store = get_request_store()
    personnel_store = get_personnel_store()
    all_personnel: list[PersonnelRecord] = []
    personnel_offset = 0
    personnel_total = 1
    while personnel_offset < personnel_total:
        personnel_page, personnel_total = personnel_store.list(
            offset=personnel_offset,
            limit=1000,
        )
        all_personnel.extend(personnel_page)
        if not personnel_page:
            break
        personnel_offset += len(personnel_page)
    personnel_with_shift = [
        p
        for p in all_personnel
        if get_shift_store().list_assignments(
            p.id,
            start_date=generation_start,
            end_date=generation_end,
        )
    ]
    if not personnel_with_shift:
        raise HTTPException(404, "هیچ پرسنل دارای شیفتی یافت نشد")

    created = 0
    removed_logs_count = 0
    request_types_list = (
        "earned_leave",
        "sick_leave",
        "unpaid_leave",
        "mission",
        "overtime",
    )
    duration_types = ("daily", "hourly")
    request_statuses = ("pending", "approved", "rejected")
    generation_days = (generation_end - generation_start).days

    for _ in range(count):
        person = _random.choice(personnel_with_shift)
        duration_type = _random.choice(duration_types)
        req_type = _random.choice(request_types_list)
        request_start = generation_start + timedelta(
            days=_random.randint(0, generation_days)
        )

        if duration_type == "daily":
            remaining_days = (generation_end - request_start).days
            request_end = request_start + timedelta(
                days=_random.randint(0, min(2, remaining_days))
            )
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

        try:
            calc = _calculate_request_duration_for_assignments(
                person, request_start, request_end, duration_type, st, et,
            )
        except (ValueError, Exception):
            continue

        int_status = _random.choice(request_statuses)
        reviewed_at = utc_now_text() if int_status != "pending" else None
        rejection_reason = (
            "Generated test rejection" if int_status == "rejected" else None
        )
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
                reviewed_at=reviewed_at,
                rejection_reason=rejection_reason,
            )
        except (ValueError, Exception):
            continue
        created += 1
        if remove_logs_in_request_dates and duration_type == "daily":
            removed_logs_count += _remove_detection_logs_for_daily_request(
                person.id,
                request_start,
                request_end,
            )

    return {
        "message": f"{created} درخواست آزمایشی ایجاد شد",
        "count": created,
        "removed_logs_count": removed_logs_count,
    }


# ── Detail ────────────────────────────────────────────────────────────────


@router.get("/{request_id}", response_model=PersonnelRequestResponse)
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


@router.patch("/{request_id}", response_model=PersonnelRequestResponse)
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


@router.patch("/{request_id}/approve", response_model=PersonnelRequestResponse)
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


@router.patch("/{request_id}/reject", response_model=PersonnelRequestResponse)
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
