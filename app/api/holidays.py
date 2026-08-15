from __future__ import annotations

from datetime import date
import io
import queue
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from app.api.holiday_schemas import (
    HolidayCreate,
    HolidayRangeResponse,
    HolidayResponse,
    HolidayUpdate,
)
from app.api.import_progress import ImportJobAccepted
from app.core.auth import require_permission
from app.core.common_schemas import UserBrief, resolve_user_brief
from app.core.holiday_import import (
    MAX_OFFICIAL_IMPORT_YEAR,
    MIN_OFFICIAL_IMPORT_YEAR,
    HolidayImportValidationError,
    build_holiday_excel_template,
    excel_choice_code_help,
    parse_holiday_excel,
)
from app.core.holiday_store import HolidayRecord
from app.core.jalali_utils import gregorian_to_jalali_str, parse_jalali_date

router = APIRouter(prefix="/api/v1/holidays", tags=["Holidays"])


def get_runtime() -> Any:
    from app.main import runtime
    return runtime


def get_holiday_store() -> Any:
    return get_runtime().holiday_store


def _record_to_response(
    record: HolidayRecord,
    created_by: UserBrief | None = None,
    updated_by: UserBrief | None = None,
) -> HolidayResponse:
    parsed_date = date.fromisoformat(record.date_value) if isinstance(record.date_value, str) else record.date_value
    from app.core.jalali_utils import utc_iso_to_jalali_datetime
    return HolidayResponse(
        id=record.id,
        name=record.name,
        date=parsed_date,
        date_jalali=gregorian_to_jalali_str(parsed_date),
        description=record.description,
        holiday_type=record.holiday_type,
        every_year=record.every_year,
        is_active=record.is_active,
        created_at=record.created_at_utc,
        updated_at=record.updated_at_utc,
        created_at_jalali=utc_iso_to_jalali_datetime(record.created_at_utc) or "",
        updated_at_jalali=utc_iso_to_jalali_datetime(record.updated_at_utc),
        created_by=created_by,
        updated_by=updated_by,
    )


def _resolve_briefs(store: Any, record: HolidayRecord) -> tuple[UserBrief | None, UserBrief | None]:
    """Resolve ``created_by`` and ``updated_by`` IDs to ``UserBrief`` objects."""
    c = u = None
    if record.created_by is not None:
        with store.database.connection() as conn:
            c = resolve_user_brief(record.created_by, conn)
    if record.updated_by is not None:
        with store.database.connection() as conn:
            u = resolve_user_brief(record.updated_by, conn)
    return c, u


@router.get("/", response_model=list[HolidayResponse])
def list_holidays(
    is_active: bool | None = Query(True),
    holiday_type: str | None = Query(None),
    every_year: bool | None = Query(None),
    _: dict = Depends(require_permission("holidays.read")),
) -> list[HolidayResponse]:
    store = get_holiday_store()
    records, _ = store.list(offset=0, limit=10000, holiday_type=holiday_type, is_active=is_active)
    if every_year is not None:
        records = [r for r in records if r.every_year == every_year]
    records.sort(key=lambda r: r.date_value)
    return [_record_to_response(r, *_resolve_briefs(store, r)) for r in records]


@router.get("/range", response_model=HolidayRangeResponse)
def get_holidays_in_range(
    start_date: str = Query(..., description="تاریخ شروع شمسی (مثلاً 1404-01-01)"),
    end_date: str = Query(..., description="تاریخ پایان شمسی (مثلاً 1404-12-29)"),
    _: dict = Depends(require_permission("holidays.read")),
) -> HolidayRangeResponse:
    try:
        start_day = parse_jalali_date(start_date)
        end_day = parse_jalali_date(end_date)
    except ValueError as exc:
        raise HTTPException(400, f"تاریخ شمسی نامعتبر است: {exc}")
    if start_day > end_day:
        raise HTTPException(400, "تاریخ شروع نباید بعد از تاریخ پایان باشد")
    store = get_holiday_store()
    records = store.get_holidays_in_range(start_day, end_day)
    records.sort(key=lambda r: r.date_value)
    holidays = [_record_to_response(r, *_resolve_briefs(store, r)) for r in records]
    unique_day_count = len({r.date_value for r in records})
    return HolidayRangeResponse(
        count=len(holidays), unique_day_count=unique_day_count, holidays=holidays
    )


@router.get("/import-excel/template")
def download_holiday_import_template(
    _: dict = Depends(require_permission("holidays.edit")),
) -> Response:
    """Download the Excel template used for official yearly imports."""
    content = build_holiday_excel_template()
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": "attachment; filename=official_holidays_template_fa.xlsx"
        },
    )


def _import_official_holidays_excel_sync(
    year: int = Query(
        ...,
        ge=MIN_OFFICIAL_IMPORT_YEAR,
        le=MAX_OFFICIAL_IMPORT_YEAR,
        description="سال جلالی؛ فقط از 1406 تا 1500",
    ),
    file: UploadFile = File(..., description="Completed official-holiday Excel template"),
    current_user: dict = Depends(require_permission("holidays.edit")),
) -> dict[str, Any]:
    """Validate the entire workbook, then atomically replace one official year."""
    filename = file.filename or ""
    if not filename.casefold().endswith((".xlsx", ".xlsm")):
        raise HTTPException(400, "فقط فایل‌های .xlsx و .xlsm پذیرفته می‌شوند")

    contents = file.file.read()
    if not contents:
        raise HTTPException(400, "فایل اکسل بارگذاری‌شده خالی است")
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(413, "حجم فایل اکسل بارگذاری‌شده بیشتر از ۵ مگابایت است")

    try:
        rows = parse_holiday_excel(contents, year=year)
    except HolidayImportValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.response_detail(year=year, filename=filename),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "فایل اکسل قابل پردازش نیست؛ هیچ تغییری در پایگاه داده انجام نشد.",
                "year": year,
                "filename": filename,
                "database_changed": False,
                "imported_count": 0,
                "summary": {
                    "total_data_rows": 0,
                    "valid_rows": 0,
                    "invalid_rows": 0,
                },
                "choice_codes": excel_choice_code_help(),
                "file_errors": [
                    {
                        "code": "invalid_workbook",
                        "message": "فایل اکسل معتبر نیست یا قابل خواندن نیست.",
                    }
                ],
                "failed_rows": [],
            },
        ) from exc

    if isinstance(current_user, dict):
        current_user_id = current_user.get("id")
    else:
        current_user_id = getattr(current_user, "id", None)

    result = get_holiday_store().replace_official_year(
        year,
        [row.to_store_mapping() for row in rows],
        user_id=current_user_id,
    )
    return {
        "message": "تعطیلات رسمی با موفقیت وارد شدند.",
        "filename": filename,
        "database_changed": True,
        "choice_codes": excel_choice_code_help(),
        "summary": {
            "total_data_rows": len(rows),
            "valid_rows": len(rows),
            "invalid_rows": 0,
        },
        **result,
    }


@router.post(
    "/import-excel",
    status_code=202,
    response_model=ImportJobAccepted,
)
async def import_official_holidays_excel(
    year: int = Query(
        ...,
        ge=MIN_OFFICIAL_IMPORT_YEAR,
        le=MAX_OFFICIAL_IMPORT_YEAR,
        description="سال جلالی؛ فقط از 1406 تا 1500",
    ),
    file: UploadFile = File(..., description="Completed official-holiday Excel template"),
    current_user: dict = Depends(require_permission("holidays.edit")),
) -> dict[str, Any]:
    filename = file.filename or "holidays.xlsx"
    if not filename.casefold().endswith((".xlsx", ".xlsm")):
        raise HTTPException(400, "فقط فایل‌های .xlsx و .xlsm پذیرفته می‌شوند")
    contents = await file.read()
    if not contents:
        raise HTTPException(400, "فایل اکسل بارگذاری‌شده خالی است")
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(413, "حجم فایل اکسل بارگذاری‌شده بیشتر از ۵ مگابایت است")

    def processor(report):
        upload = UploadFile(filename=filename, file=io.BytesIO(contents))
        try:
            result = _import_official_holidays_excel_sync(
                year=year,
                file=upload,
                current_user=current_user,
            )
            result["success"] = True
            result["summary"]["total_rows"] = result["summary"]["total_data_rows"]
            result["summary"]["imported"] = result.get("imported_count", 0)
            result["summary"]["failed"] = 0
            return result
        except HTTPException as exc:
            detail = exc.detail
            if isinstance(detail, dict):
                result = {"success": False, **detail}
                summary = result.setdefault("summary", {})
                total = int(summary.get("total_data_rows", 0))
                failed = int(summary.get("invalid_rows", 0))
                summary.update(total_rows=total, imported=0, skipped=max(0, total - failed), failed=failed)
                return result
            return {
                "success": False,
                "message": str(detail),
                "database_changed": False,
                "summary": {"total_rows": 1, "imported": 0, "skipped": 0, "failed": 1},
                "file_errors": [{"code": "invalid_workbook", "message": str(detail)}],
            }

    try:
        runtime = get_runtime()
        user_id = current_user.get("id") if isinstance(current_user, dict) else getattr(current_user, "id", None)
        record = runtime.excel_imports.submit(
            "holidays_excel", filename, user_id, processor
        )
    except queue.Full:
        raise HTTPException(503, "صف ورود فایل‌های اکسل پر است؛ بعداً دوباره تلاش کنید.")
    return {
        "job_id": record.id,
        "progress_id": record.id,
        "status": record.status,
        "status_url": f"/api/v1/import-progress/{record.id}",
        "message": "فایل دریافت شد و پردازش آن در صف قرار گرفت.",
    }


@router.get("/{holiday_id}", response_model=HolidayResponse)
def get_holiday(
    holiday_id: int,
    _: dict = Depends(require_permission("holidays.read")),
) -> HolidayResponse:
    store = get_holiday_store()
    record = store.get(holiday_id)
    if record is None:
        raise HTTPException(404, "\u062a\u0639\u0637\u06cc\u0644\u06cc \u06cc\u0627\u0641\u062a \u0646\u0634\u062f!")
    return _record_to_response(record, *_resolve_briefs(store, record))


@router.post("/", response_model=HolidayResponse, status_code=201)
def create_holiday(
    body: HolidayCreate,
    current_user: dict = Depends(require_permission("holidays.edit")),
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
    return _record_to_response(record, *_resolve_briefs(store, record))


@router.patch("/{holiday_id}", response_model=HolidayResponse)
def patch_holiday(
    holiday_id: int,
    body: HolidayUpdate,
    current_user: dict = Depends(require_permission("holidays.edit")),
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
    return _record_to_response(record, *_resolve_briefs(store, record))


@router.delete("/{holiday_id}", response_model=dict)
def delete_holiday(
    holiday_id: int,
    current_user: dict = Depends(require_permission("holidays.edit")),
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
