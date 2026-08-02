"""Work-shift API with the legacy request/response contract."""

from __future__ import annotations

from enum import Enum
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator

from app.core.auth import require_role
from app.core.shift_store import WorkShiftRecord
from app.core.legacy_service import (
    legacy_shift_response,
    legacy_weekdays_to_internal,
    format_jalali,
    validate_clock_time,
    validate_jalali_date,
    validate_timezone,
)

router = APIRouter(prefix="/api/v1/shifts", tags=["Shifts"])


class ShiftTypeEnum(str, Enum):
    MORNING = "morning"
    EVENING = "evening"
    NIGHT = "night"
    REMOTE = "remote"
    FLEXIBLE = "flexible"
    ROTATING = "rotating"


class ShiftCreate(BaseModel):
    shift_name: str
    shift_type: ShiftTypeEnum
    start_time: str
    end_time: str
    max_minutes_delay: int | None = None
    max_minutes_early: int | None = None
    max_overtime_hours: float = 8.0
    timezone_name: str = "Asia/Tehran"
    monday: bool = True
    tuesday: bool = True
    wednesday: bool = True
    thursday: bool = False
    friday: bool = False
    saturday: bool = True
    sunday: bool = True


class ShiftUpdate(BaseModel):
    shift_name: str | None = None
    shift_type: ShiftTypeEnum | None = None
    start_time: str | None = None
    end_time: str | None = None
    timezone_name: str | None = None
    max_minutes_delay: int | None = None
    max_minutes_early: int | None = None
    max_overtime_hours: float | None = None
    monday: bool = True
    tuesday: bool = True
    wednesday: bool = True
    thursday: bool = False
    friday: bool = False
    saturday: bool = True
    sunday: bool = True


class ShiftAssignmentCreate(BaseModel):
    start_date: str
    end_date: str


class BulkShiftAssignmentCreate(BaseModel):
    shift_id: int
    start_date: str
    end_date: str
    personnel_ids: list[int] = Field(min_length=1)

    @field_validator("personnel_ids")
    @classmethod
    def validate_unique_personnel_ids(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("شناسه پرسنل تکراری مجاز نیست")
        return value


SHIFT_TYPES = [
    {"value": "morning", "label": "شیفت صبح"},
    {"value": "evening", "label": "شیفت عصر"},
    {"value": "night", "label": "شیفت شب"},
    {"value": "remote", "label": "دورکاری"},
    {"value": "flexible", "label": "ساعت کاری شناور"},
    {"value": "rotating", "label": "شیفت چرخشی"},
]


def get_runtime() -> Any:
    from app.main import runtime

    return runtime


def get_shift_store() -> Any:
    return get_runtime().shift_store


def _format_time(value: str) -> str:
    parts = value.split(":")
    if len(parts) == 2:
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}:00"
    return value


def _legacy_shift_dict(record: WorkShiftRecord, store: Any = None) -> dict[str, Any]:
    count = 0
    if store is not None:
        count = store.count_personnel_in_shift(record.id)
    response = legacy_shift_response(record, count)
    response["start_time"] = _format_time(response["start_time"])
    response["end_time"] = _format_time(response["end_time"])
    return response


def _parse_time(value: str) -> str:
    try:
        validate_clock_time(value)
        hours, minutes = (int(part) for part in value.split(":"))
    except (TypeError, ValueError):
        raise HTTPException(400, "فرمت زمان نامعتبر است. از HH:MM استفاده کنید")
    return f"{hours:02d}:{minutes:02d}"


def _parse_timezone(value: str) -> str:
    try:
        return validate_timezone(value)
    except ValueError:
        raise HTTPException(400, "نام منطقه زمانی نامعتبر است")


def _personnel_shift_response(item: dict[str, Any]) -> dict[str, Any]:
    shift = item.get("shift")
    if shift is not None:
        shift = dict(shift)
        shift["start_time"] = _format_time(shift["start_time"])
        shift["end_time"] = _format_time(shift["end_time"])
        shift["timezone_name"] = shift.get("timezone_name") or "Asia/Tehran"
        # The old nested response constructed these fields without values;
        # Pydantic therefore serialized their Optional defaults as null.
        shift["max_minutes_delay"] = None
        shift["max_minutes_early"] = None
    return {
        "personnel_id": item["personnel_id"],
        "full_name": item["full_name"],
        "national_code": item["national_code"],
        "department_id": item.get("department_id"),
        "department_name": item.get("department_name"),
        "degree": item.get("degree"),
        "shift": shift,
    }


def _weekday_payload(body: BaseModel) -> dict[str, bool]:
    return legacy_weekdays_to_internal(body.model_dump())


def _assignment_response(record: Any) -> dict[str, Any]:
    return {
        "id": record.id,
        "personnel_id": record.personnel_id,
        "shift_id": record.shift_id,
        "start_date": format_jalali(record.start_date),
        "end_date": format_jalali(record.end_date),
        "start_date_gregorian": record.start_date.isoformat(),
        "end_date_gregorian": record.end_date.isoformat(),
        "created_at_utc": record.created_at_utc,
        "updated_at_utc": record.updated_at_utc,
    }


@router.post("/bulk-assignments", status_code=status.HTTP_201_CREATED)
def bulk_assign_personnel_to_shift(
    body: BulkShiftAssignmentCreate,
    _: dict = Depends(require_role("admin")),
) -> dict[str, Any]:
    try:
        start_date = validate_jalali_date(body.start_date)
        end_date = validate_jalali_date(body.end_date)
        assignments = get_shift_store().assign_personnel_bulk(
            personnel_ids=body.personnel_ids,
            shift_id=body.shift_id,
            start_date=start_date,
            end_date=end_date,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {
        "shift_id": body.shift_id,
        "start_date": body.start_date,
        "end_date": body.end_date,
        "assigned_count": len(assignments),
        "assignments": [_assignment_response(item) for item in assignments],
    }


# Keep this registration order aligned with old/backend/app/routers/shifts.py.
@router.get("/")
def get_all_shifts(_: dict = Depends(require_role("operator"))) -> list[dict[str, Any]]:
    store = get_shift_store()
    records, _ = store.list(offset=0, limit=10000, search=None)
    records.sort(key=lambda record: record.shift_name)
    return [_legacy_shift_dict(record, store) for record in records]


@router.get("/statistics")
def get_shift_statistics(_: dict = Depends(require_role("operator"))) -> dict[str, Any]:
    return get_shift_store().statistics()


@router.get("/{shift_id}")
def get_shift_by_id(
    shift_id: int,
    _: dict = Depends(require_role("operator")),
) -> dict[str, Any]:
    store = get_shift_store()
    record = store.get(shift_id)
    if record is None:
        raise HTTPException(404, "شیفت یافت نشد")
    return _legacy_shift_dict(record, store)


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_shift(
    body: ShiftCreate,
    _: dict = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_shift_store()
    try:
        values = body.model_dump()
        values["start_time"] = _parse_time(values["start_time"])
        values["end_time"] = _parse_time(values["end_time"])
        values["timezone_name"] = _parse_timezone(values["timezone_name"])
        values["shift_type"] = values["shift_type"].value
        values["max_minutes_delay"] = values["max_minutes_delay"] or 0
        values["max_minutes_early"] = values["max_minutes_early"] or 0
        values.update(_weekday_payload(body))
        if store.get_by_name(values["shift_name"]) is not None:
            raise HTTPException(400, "نام شیفت قبلاً ثبت شده است")
        record = store.create(**values)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
            raise HTTPException(400, "نام شیفت قبلاً ثبت شده است")
        raise
    return _legacy_shift_dict(record, store)


@router.put("/{shift_id}")
def update_shift(
    shift_id: int,
    body: ShiftUpdate,
    _: dict = Depends(require_role("admin")),
) -> dict[str, Any]:
    store = get_shift_store()
    try:
        values = body.model_dump()
        if values["shift_type"] is not None:
            values["shift_type"] = values["shift_type"].value
        for field in ("start_time", "end_time"):
            if values[field] is not None:
                values[field] = _parse_time(values[field])
        if values["timezone_name"] is not None:
            values["timezone_name"] = _parse_timezone(values["timezone_name"])
        values.update(_weekday_payload(body))
        if values["shift_name"] is not None and store.get_by_name(
            values["shift_name"], exclude_id=shift_id
        ) is not None:
            raise HTTPException(400, "نام شیفت قبلاً ثبت شده است")
        record = store.update(shift_id, **values)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
            raise HTTPException(400, "نام شیفت قبلاً ثبت شده است")
        raise
    if record is None:
        raise HTTPException(404, "شیفت یافت نشد")
    return _legacy_shift_dict(record, store)


@router.delete("/{shift_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_shift(
    shift_id: int,
    force: bool = Query(False, description="حذف اجباری حتی اگر به پرسنل اختصاص داده شده باشد"),
    _: dict = Depends(require_role("admin")),
) -> Response:
    store = get_shift_store()
    try:
        deleted = store.delete(shift_id, force=force)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if not deleted:
        raise HTTPException(404, "شیفت یافت نشد")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{shift_id}/personnel")
def get_personnel_by_shift(
    shift_id: int,
    _: dict = Depends(require_role("admin")),
) -> list[dict[str, Any]]:
    store = get_shift_store()
    if store.get(shift_id) is None:
        raise HTTPException(404, "شیفت یافت نشد")
    return [_personnel_shift_response(item) for item in store.list_personnel_in_shift(shift_id)]


@router.get("/types")
def get_shift_types(_: dict = Depends(require_role("admin"))) -> list[dict[str, str]]:
    return SHIFT_TYPES


# Current-project extensions remain available after the legacy route block.
@router.post("/{shift_id}/assign/{personnel_id}", status_code=status.HTTP_201_CREATED)
def assign_personnel_to_shift(
    shift_id: int,
    personnel_id: int,
    body: ShiftAssignmentCreate,
    _: dict = Depends(require_role("admin")),
) -> dict[str, Any]:
    try:
        start_date = validate_jalali_date(body.start_date)
        end_date = validate_jalali_date(body.end_date)
        assignment = get_shift_store().assign_personnel(
            personnel_id, shift_id, start_date, end_date
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return _assignment_response(assignment)


@router.get("/{shift_id}/personnel/{personnel_id}/assignments")
def get_personnel_shift_assignments(
    shift_id: int,
    personnel_id: int,
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    _: dict = Depends(require_role("operator")),
) -> list[dict[str, Any]]:
    if get_shift_store().get(shift_id) is None:
        raise HTTPException(404, "شیفت یافت نشد")
    try:
        start = validate_jalali_date(start_date) if start_date else None
        end = validate_jalali_date(end_date) if end_date else None
    except ValueError:
        raise HTTPException(400, "تاریخ نامعتبر است")
    if start is not None and end is not None and end < start:
        raise HTTPException(400, "تاریخ پایان نمی‌تواند قبل از تاریخ شروع باشد")
    return [
        _assignment_response(item)
        for item in get_shift_store().list_assignments(personnel_id, start, end)
        if item.shift_id == shift_id
    ]


@router.delete("/{shift_id}/assignments/{assignment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_personnel_shift_assignment(
    shift_id: int,
    assignment_id: int,
    _: dict = Depends(require_role("admin")),
) -> Response:
    # shift_id keeps the assignment operation naturally scoped under its shift.
    if get_shift_store().get(shift_id) is None:
        raise HTTPException(404, "شیفت یافت نشد")
    if not get_shift_store().delete_assignment(assignment_id, shift_id=shift_id):
        raise HTTPException(404, "تخصیص شیفت یافت نشد")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/assign/{personnel_id}", include_in_schema=False)
def remove_personnel_shift(
    personnel_id: int,
    _: dict = Depends(require_role("superuser")),
) -> dict[str, Any]:
    return {"removed": get_shift_store().remove_personnel_shift(personnel_id)}


@router.get("/statistics/summary", include_in_schema=False)
def shift_statistics_summary(_: dict = Depends(require_role("operator"))) -> dict[str, Any]:
    return get_shift_store().statistics()
