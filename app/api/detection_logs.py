from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import jdatetime
import openpyxl
from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from openpyxl.styles import Alignment

from app.config import settings
from app.core.auth import require_role
from app.core.detection_log_store import DetectionLogRecord, DetectionLogStore
from app.core.jalali_utils import (
    _get_tehran_tz,
    gregorian_to_jalali_str,
    jalali_month_utc_range,
    local_day_utc_range,
    parse_jalali_date,
)
from app.core.legacy_detection_service import (
    calculate_access,
    legacy_detection_response,
)
from app.core.recent_detection_service import (
    build_recent_detection_refresh_message,
    get_single_detection_payload_by_id,
)

router = APIRouter(prefix="/api/v1/logs", tags=["Detection Logs"])

_detection_log_store: DetectionLogStore | None = None


def get_runtime() -> Any:
    from app.main import runtime
    return runtime


def get_detection_log_store() -> DetectionLogStore:
    global _detection_log_store
    if _detection_log_store is None:
        _detection_log_store = DetectionLogStore(get_runtime().database)
    return _detection_log_store


def get_location_store() -> Any:
    return get_runtime().location_store


def _push_refresh_for_log(runtime: Any, log_id: int) -> None:
    broadcast = getattr(runtime, "broadcast", None)
    if broadcast is None:
        return
    payload = get_single_detection_payload_by_id(runtime, log_id)
    if payload is not None:
        broadcast.publish_control_event(
            build_recent_detection_refresh_message(payload, log_id)
        )


def get_personnel_store() -> Any:
    return get_runtime().personnel_store


def _get_current_user_id(current_user: dict) -> int | None:
    return current_user.get("id")


def _resolve_media_path(relative_path: str | None) -> Path:
    if not relative_path:
        return None
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return settings.saved_media_path / relative_path


def _resolve_usernames(
    record: DetectionLogRecord,
) -> tuple[str | None, str | None]:
    c_user = None
    u_user = None
    if record.created_by:
        try:
            u = get_runtime().database.connection().execute(
                "SELECT username FROM users WHERE id = ?", (record.created_by,)
            ).fetchone()
            if u:
                c_user = u["username"]
        except Exception:
            pass
    if record.updated_by:
        try:
            u = get_runtime().database.connection().execute(
                "SELECT username FROM users WHERE id = ?", (record.updated_by,)
            ).fetchone()
            if u:
                u_user = u["username"]
        except Exception:
            pass
    return c_user, u_user


def _resolve_names(
    room_id: int | None,
    camera_id: str | None,
) -> tuple[str | None, str | None, str | None, str | None]:
    room_name = None
    camera_name = None
    section_name = None
    building_name = None
    ls = get_location_store()
    if room_id is not None:
        room = ls.get_room(room_id)
        if room is not None:
            room_name = room.name
            from app.core.location_store import SectionRecord
            if room.section_id is not None:
                sec = ls.get_section(room.section_id)
                if sec is not None:
                    section_name = sec.name
                    if sec.building_id is not None:
                        bld = ls.get_building(sec.building_id)
                        if bld is not None:
                            building_name = bld.name
    if camera_id is not None:
        try:
            cam = get_runtime().registry.get(camera_id)
            if cam is not None:
                camera_name = cam.name
        except Exception:
            pass
    return room_name, camera_name, section_name, building_name


def _resolve_personnel_name(personnel_id: int | None) -> str | None:
    if personnel_id is None:
        return None
    p = get_personnel_store().get(personnel_id)
    if p is None:
        return None
    return f"{p.fname} {p.lname}"


def _build_response(
    record: DetectionLogRecord,
    include_detail: bool = False,
) -> dict:
    c_user, u_user = _resolve_usernames(record)
    full_name = _resolve_personnel_name(record.personnel_id)
    room_name, camera_name, section_name, building_name = _resolve_names(
        record.room_id, record.camera_id
    )
    return legacy_detection_response(
        record,
        full_name=full_name,
        room_name=room_name,
        camera_name=camera_name,
        section_name=section_name,
        building_name=building_name,
        created_by_username=c_user,
        updated_by_username=u_user,
        include_detail=include_detail,
    )


def _delete_media_files(record: DetectionLogRecord) -> None:
    for attr in ("face_image", "body_image", "snapshot_image", "video", "face_video_or_unknown_faces"):
        path_str = getattr(record, attr, None)
        if path_str:
            fpath = _resolve_media_path(path_str)
            if fpath and fpath.exists():
                fpath.unlink(missing_ok=True)


def _period_to_utc_range(
    period: str,
    from_date_jalali: str | None,
    to_date_jalali: str | None,
) -> tuple[str | None, str | None]:
    if period == "today":
        now_tehran = datetime.now(_get_tehran_tz())
        today_local = now_tehran.date()
        utc_start, utc_end = local_day_utc_range(today_local)
        return utc_start.isoformat(), utc_end.isoformat()
    if period == "last_week":
        now_tehran = datetime.now(_get_tehran_tz())
        today_local = now_tehran.date()
        start_local = today_local - timedelta(days=7)
        utc_start, _ = local_day_utc_range(start_local)
        _, utc_end = local_day_utc_range(today_local)
        return utc_start.isoformat(), utc_end.isoformat()
    if period == "last_month":
        now_tehran = datetime.now(_get_tehran_tz())
        today_local = now_tehran.date()
        start_local = today_local - timedelta(days=30)
        utc_start, _ = local_day_utc_range(start_local)
        _, utc_end = local_day_utc_range(today_local)
        return utc_start.isoformat(), utc_end.isoformat()
    if period == "custom":
        if not from_date_jalali or not to_date_jalali:
            raise HTTPException(400, "from_date_jalali and to_date_jalali are required for period=custom")
        from_g = parse_jalali_date(from_date_jalali)
        to_g = parse_jalali_date(to_date_jalali)
        utc_start, _ = local_day_utc_range(from_g)
        _, utc_end = local_day_utc_range(to_g)
        return utc_start.isoformat(), utc_end.isoformat()
    raise HTTPException(400, f"Unknown period: {period!r}")


# ── Static routes (before /{log_id}) ──────────────────────────────────


@router.post("/log")
def create_detection_log(
    body: dict[str, Any],
    current_user: dict = Depends(require_role("operator")),
) -> dict:
    store = get_detection_log_store()
    ls = get_location_store()

    personnel_id = body.get("personnel_id")
    room_id = body.get("room_id")
    person = str(body.get("person", "Unknown"))
    confidence = float(body.get("confidence", 0.0))
    ref_img_id = body.get("ref_img_id")
    face_image = body.get("face_image_path") or body.get("face_image")
    body_image = body.get("body_image_path") or body.get("body_image")
    snapshot_image = body.get("snapshot_image_path") or body.get("snapshot_image")
    unknown_faces = body.get("unknown_faces_path") or body.get("face_video_or_unknown_faces")
    log_type = str(body.get("log_type", "real_time"))
    detection_time = body.get("detection_time", "")
    access_granted = bool(body.get("access_granted", False))
    counts_for_attendance = bool(body.get("counts_for_attendance", True))
    camera_id = body.get("camera_id")

    if not detection_time:
        detection_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    if personnel_id is not None and room_id is not None:
        existing = store.find_dedup(
            personnel_id=int(personnel_id),
            room_id=int(room_id),
            detection_time_utc_str=detection_time,
            time_window_seconds=60,
        )
        if existing is not None:
            if confidence >= existing.confidence:
                _delete_media_files(existing)
                store.delete(existing.id)
            else:
                return _build_response(existing, include_detail=True)

    access = calculate_access(personnel_id, room_id, ls)

    user_id = _get_current_user_id(current_user)
    try:
        record = store.create(
            source_system="api",
            personnel_id=personnel_id,
            person=person,
            confidence=confidence,
            detection_time=detection_time,
            ref_img_id=ref_img_id,
            room_id=room_id,
            camera_id=camera_id,
            access_granted=access,
            counts_for_attendance=counts_for_attendance,
            log_type=log_type,
            face_image=face_image,
            body_image=body_image,
            snapshot_image=snapshot_image,
            face_video_or_unknown_faces=unknown_faces,
            created_by=user_id,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    return _build_response(record, include_detail=True)


@router.get("/filter")
def filter_logs(
    period: str = Query("today"),
    from_date_jalali: str | None = Query(None),
    to_date_jalali: str | None = Query(None),
    personnel_id: int | None = Query(None),
    national_code: str | None = Query(None),
    room_id: int | None = Query(None),
    camera_id: str | None = Query(None),
    section_id: int | None = Query(None),
    building_id: int | None = Query(None),
    access_granted: bool | None = Query(None),
    counts_for_attendance: bool | None = Query(None),
    log_type: str | None = Query(None),
    min_confidence: float | None = Query(None),
    max_confidence: float | None = Query(None),
    include_thumbnails: bool = Query(False),
    skip: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=1000),
    _: dict = Depends(require_role("operator")),
) -> list[dict]:
    from_date_utc, to_date_utc = _period_to_utc_range(period, from_date_jalali, to_date_jalali)

    store = get_detection_log_store()
    records, _ = store.list_filter(
        offset=skip,
        limit=limit,
        personnel_id=personnel_id,
        national_code=national_code,
        room_id=room_id,
        camera_id=camera_id,
        section_id=section_id,
        building_id=building_id,
        access_granted=access_granted,
        counts_for_attendance=counts_for_attendance,
        log_type=log_type,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        from_date_utc=from_date_utc,
        to_date_utc=to_date_utc,
        include_thumbnails=include_thumbnails,
    )
    return [_build_response(r, include_detail=True) for r in records]


@router.get("/daily-summary")
def daily_summary(
    period: str = Query("today"),
    from_date_jalali: str | None = Query(None),
    to_date_jalali: str | None = Query(None),
    include_non_workdays: bool = Query(False),
    include_absent: bool = Query(True),
    _: dict = Depends(require_role("operator")),
) -> list[dict]:
    from_date_utc, to_date_utc = _period_to_utc_range(period, from_date_jalali, to_date_jalali)
    store = get_detection_log_store()
    records, _ = store.list_filter(
        from_date_utc=from_date_utc,
        to_date_utc=to_date_utc,
    )
    grouped: dict[int, list[DetectionLogRecord]] = defaultdict(list)
    for r in records:
        if r.personnel_id is not None:
            grouped[r.personnel_id].append(r)

    result: list[dict] = []
    for pid, logs in grouped.items():
        full_name = _resolve_personnel_name(pid)
        local_tz = _get_tehran_tz()
        first_seen = None
        last_seen = None
        total_seconds = 0
        for log in logs:
            try:
                dt_log = datetime.fromisoformat(log.detection_time.replace("Z", "+00:00"))
                local_dt = dt_log.astimezone(local_tz)
                if first_seen is None or local_dt < first_seen:
                    first_seen = local_dt
                if last_seen is None or local_dt > last_seen:
                    last_seen = local_dt
            except (ValueError, TypeError):
                pass
        if first_seen and last_seen and first_seen != last_seen:
            total_seconds = int((last_seen - first_seen).total_seconds())
        present = len(logs) > 0
        if not present and not include_absent:
            continue
        result.append({
            "personnel_id": pid,
            "full_name": full_name,
            "detection_count": len(logs),
            "first_seen_local": first_seen.isoformat() if first_seen else None,
            "last_seen_local": last_seen.isoformat() if last_seen else None,
            "total_seconds": total_seconds,
            "present": present,
        })
    return result


@router.get("/monthly-summary")
def monthly_summary(
    jalali_year: int = Query(..., ge=1400, le=1500),
    jalali_month: int = Query(..., ge=1, le=12),
    personnel_id: int | None = Query(None),
    section_id: int | None = Query(None),
    shift_id: int | None = Query(None),
    include_daily_rows: bool = Query(False),
    move_days: int = Query(10),
    _: dict = Depends(require_role("operator")),
) -> list[dict]:
    utc_start, utc_end = jalali_month_utc_range(jalali_year, jalali_month)
    store = get_detection_log_store()
    records, _ = store.list_filter(
        personnel_id=personnel_id,
        section_id=section_id,
        from_date_utc=utc_start.isoformat(),
        to_date_utc=utc_end.isoformat(),
    )
    grouped: dict[int, list[DetectionLogRecord]] = defaultdict(list)
    for r in records:
        if r.personnel_id is not None:
            grouped[r.personnel_id].append(r)

    result: list[dict] = []
    for pid, logs in grouped.items():
        full_name = _resolve_personnel_name(pid)
        attendance_count = sum(1 for l in logs if l.counts_for_attendance)
        access_count = sum(1 for l in logs if l.access_granted)
        row = {
            "personnel_id": pid,
            "full_name": full_name,
            "total_detections": len(logs),
            "attendance_count": attendance_count,
            "access_count": access_count,
            "year": jalali_year,
            "month": jalali_month,
        }
        if include_daily_rows:
            daily: dict[str, list[dict]] = defaultdict(list)
            for l in logs:
                try:
                    dt_log = datetime.fromisoformat(l.detection_time.replace("Z", "+00:00"))
                    j_date = jdatetime.date.fromgregorian(date=dt_log.astimezone(_get_tehran_tz()).date())
                    day_key = f"{j_date.year:04d}-{j_date.month:02d}-{j_date.day:02d}"
                    daily[day_key].append(_build_response(l))
                except (ValueError, TypeError):
                    pass
            row["daily"] = dict(daily)
        result.append(row)
    return result


@router.get("/monthly-performance")
def monthly_performance(
    jalali_year: int = Query(..., ge=1400, le=1500),
    jalali_month: int = Query(..., ge=1, le=12),
    section_id: int | None = Query(None),
    _: dict = Depends(require_role("operator")),
) -> list[dict]:
    utc_start, utc_end = jalali_month_utc_range(jalali_year, jalali_month)
    store = get_detection_log_store()
    records, _ = store.list_filter(
        section_id=section_id,
        from_date_utc=utc_start.isoformat(),
        to_date_utc=utc_end.isoformat(),
    )
    grouped: dict[int, list[DetectionLogRecord]] = defaultdict(list)
    for r in records:
        if r.personnel_id is not None:
            grouped[r.personnel_id].append(r)

    result: list[dict] = []
    for pid, logs in grouped.items():
        full_name = _resolve_personnel_name(pid)
        avg_conf = sum(l.confidence for l in logs) / len(logs) if logs else 0.0
        attendance_count = sum(1 for l in logs if l.counts_for_attendance)
        score = round(avg_conf * 100, 2)
        result.append({
            "personnel_id": pid,
            "full_name": full_name,
            "total_detections": len(logs),
            "attendance_count": attendance_count,
            "average_confidence": round(avg_conf, 4),
            "score": score,
            "quality": "عالی" if score >= 90 else "خوب" if score >= 70 else "متوسط" if score >= 50 else "ضعیف",
        })
    return result


@router.get("/yearly-leave-summary")
def yearly_leave_summary(
    jalali_year: int = Query(..., ge=1400, le=1500),
    _: dict = Depends(require_role("operator")),
) -> list[dict]:
    store = get_detection_log_store()
    personnel_store = get_personnel_store()
    all_personnel = personnel_store.list(offset=0, limit=10000)[0]

    result: list[dict] = []
    for p in all_personnel:
        monthly_leave: dict[str, int] = {}
        for m in range(1, 13):
            utc_s, utc_e = jalali_month_utc_range(jalali_year, m)
            records, _ = store.list_filter(
                personnel_id=p.id,
                from_date_utc=utc_s.isoformat(),
                to_date_utc=utc_e.isoformat(),
            )
            monthly_leave[str(m)] = len(records)
        result.append({
            "personnel_id": p.id,
            "full_name": f"{p.fname} {p.lname}",
            "year": jalali_year,
            "monthly": monthly_leave,
        })
    return result


@router.get("/import-excel/template")
def import_excel_template(
    _: dict = Depends(require_role("operator")),
):
    import io
    wb = openpyxl.Workbook()

    ws = wb.active
    ws.title = "ورود اطلاعات"
    headers = [
        "کد ملی", "سال", "ماه", "روز", "ساعت", "دقیقه",
        "شناسه اتاق", "دسترسی مجاز", "لحاظ در حضور و غیاب",
    ]
    ws.append(headers)
    for cell in ws[1]:
        cell.font = openpyxl.styles.Font(bold=True)
        cell.alignment = Alignment(horizontal="right")
    col_widths = [15, 8, 8, 8, 8, 8, 12, 12, 12]
    for i, w in enumerate(col_widths, 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w
    ws.append(["0012345678", 1403, 6, 15, 8, 30, "", "1", "1"])

    ws_guide = wb.create_sheet("راهنما", 0)
    guide_lines = [
        "راهنمای واردسازی لاگ‌های تشخیص",
        "",
        "ستون‌ها:",
        "A: کد ملی (۱۰ رقمی، اجباری)",
        "B: سال شمسی (۴ رقمی، اجباری)",
        "C: ماه شمسی (۱-۱۲، اجباری)",
        "D: روز شمسی (۱-۳۱، اجباری)",
        "E: ساعت (۰-۲۳، پیش‌فرض ۰)",
        "F: دقیقه (۰-۵۹، پیش‌فرض ۰)",
        "G: شناسه اتاق (عددی، اختیاری)",
        "H: دسترسی مجاز (0 یا 1، پیش‌فرض 1)",
        "I: لحاظ در حضور و غیاب (0 یا 1، پیش‌فرض 1)",
        "",
        "توجه:",
        "- ردیف اول (سرستون) در واردسازی نادیده گرفته می‌شود",
        "- ردیف‌های خالی رد می‌شوند",
    ]
    ws_guide.append(["راهنمای واردسازی"])
    for i, line in enumerate(guide_lines, 1):
        ws_guide.cell(row=i, column=1, value=line)
    ws_guide.column_dimensions["A"].width = 60
    ws_guide.protection.sheet = True

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=import_template.xlsx"},
    )


@router.post("/import-excel")
def import_excel(
    file: UploadFile,
    calendar: str = Query("jalali", pattern="^(jalali|gregorian)$"),
    skip_duplicates: bool = Query(True),
    current_user: dict = Depends(require_role("operator")),
) -> dict:
    if file.filename and not (file.filename.endswith(".xlsx") or file.filename.endswith(".xlsm")):
        raise HTTPException(400, "Only .xlsx or .xlsm files are accepted")

    import io
    contents = file.file.read()
    wb = openpyxl.load_workbook(io.BytesIO(contents))
    ws = wb.active
    if ws is None:
        raise HTTPException(400, "Workbook has no active sheet")

    store = get_detection_log_store()
    ls = get_location_store()
    ps = get_personnel_store()
    user_id = _get_current_user_id(current_user)

    imported = 0
    skipped = 0
    failed = 0
    for row_idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        if not row or all(v is None for v in row):
            continue
        try:
            national_code = str(row[0]).strip() if row[0] is not None else ""
            jalali_year = int(row[1]) if row[1] is not None else None
            jalali_month = int(row[2]) if row[2] is not None else None
            jalali_day = int(row[3]) if row[3] is not None else None
            hour = int(row[4]) if row[4] is not None else 0
            minute = int(row[5]) if row[5] is not None else 0
            room_id = int(row[6]) if row[6] is not None else None
            access_granted = bool(int(row[7])) if row[7] is not None else False
            counts_for_attendance = bool(int(row[8])) if row[8] is not None else True
        except (ValueError, TypeError, IndexError):
            failed += 1
            continue

        if not national_code or jalali_year is None or jalali_month is None or jalali_day is None:
            failed += 1
            continue

        personnel = None
        try:
            personnel = ps.get_by_national_code(national_code)
        except Exception:
            pass

        if personnel is None:
            failed += 1
            continue

        try:
            j_dt = jdatetime.datetime(jalali_year, jalali_month, jalali_day, hour, minute)
            g_dt = j_dt.togregorian()
            detection_time = g_dt.replace(tzinfo=_get_tehran_tz()).astimezone(timezone.utc).isoformat()
        except (ValueError, TypeError):
            failed += 1
            continue

        if skip_duplicates and personnel.id is not None and room_id is not None:
            existing = store.find_dedup(
                personnel_id=personnel.id,
                room_id=room_id,
                detection_time_utc_str=detection_time,
                time_window_seconds=60,
            )
            if existing is not None:
                skipped += 1
                continue

        access = calculate_access(personnel.id, room_id, ls)

        try:
            store.create(
                source_system="excel_import",
                personnel_id=personnel.id,
                person=f"{personnel.fname} {personnel.lname}",
                confidence=1.0,
                detection_time=detection_time,
                room_id=room_id,
                camera_id=None,
                access_granted=access,
                counts_for_attendance=counts_for_attendance,
                log_type="excel_import",
                created_by=user_id,
            )
            imported += 1
        except Exception:
            failed += 1

    return {
        "imported_rows": imported,
        "skipped_rows": skipped,
        "failed_rows": failed,
    }


@router.delete("/delete-all-logs")
def delete_all_logs(
    current_user: dict = Depends(require_role("superuser")),
) -> dict:
    store = get_detection_log_store()
    count = store.delete_all()
    return {"deleted_count": count}


# ── Generate fake detections (admin only) ────────────────────────────


@router.post("/generate-fake", status_code=201)
def generate_fake_detections(
    count: int = Query(10, ge=1, le=200, description="تعداد لاگ آزمایشی برای ایجاد"),
    month: int | None = Query(None, ge=1, le=12, description="ماه (jalali) برای متمرکز کردن لاگ‌ها در یک ماه خاص"),
    year: int | None = Query(None, ge=1300, le=1500, description="سال (jalali) همراه با month"),
    from_date: str | None = Query(None, description="تاریخ شروع jalali (جایگزین month/year)"),
    to_date: str | None = Query(None, description="تاریخ پایان jalali (همراه from_date)"),
    personnel_id: int | None = Query(None, description="محدود کردن به پرسنل مشخص"),
    room_id: int | None = Query(None, description="محدود کردن به اتاق مشخص"),
    camera_id: str | None = Query(None, description="محدود کردن به دوربین مشخص"),
    pair_logs: bool = Query(False, description="ایجاد لاگ‌های جفتی (ورود+خروج) با فاصله چند دقیقه"),
    admin_user: Any = Depends(require_role("admin")),
) -> dict[str, Any]:
    """Generate fake detection logs for testing/demo purposes.

    Supports date range filtering (by jalali month/year or from_date/to_date),
    filtering by personnel/room/camera, and paired entry/exit log generation.
    """
    import random as _random

    store = get_detection_log_store()
    ps = get_personnel_store()
    ls = get_location_store()

    # ── Resolve date range ──────────────────────────────────────────
    base_time = datetime.now(timezone.utc)
    range_start: date | None = None
    range_end: date | None = None

    if from_date and to_date:
        try:
            from_g = parse_jalali_date(from_date)
            to_g = parse_jalali_date(to_date)
            range_start = from_g
            range_end = to_g
        except Exception:
            raise HTTPException(400, "فرمت تاریخ نامعتبر است (jalali: YYYY-MM-DD)")
    elif month is not None:
        y = year or jdatetime.date.today().year
        try:
            from_g = parse_jalali_date(f"{y}-{month:02d}-01")
        except Exception:
            raise HTTPException(400, "ماه یا سال نامعتبر است")
        to_g = from_g + timedelta(days=30)
        range_start = from_g
        range_end = to_g

    # ── Resolve personnel ───────────────────────────────────────────
    if personnel_id is not None:
        person = ps.get(personnel_id)
        if person is None:
            raise HTTPException(404, "پرسنل یافت نشد")
        selected_personnel = [person]
    else:
        all_p, _ = ps.list(limit=1000)
        if not all_p:
            raise HTTPException(404, "هیچ پرسنلی یافت نشد")
        selected_personnel = all_p

    # ── Resolve rooms ────────────────────────────────────────────────
    if room_id is not None:
        room = ls.get_room(room_id)
        if room is None:
            raise HTTPException(404, "اتاق یافت نشد")
        selected_room_ids = [room.id]
    else:
        all_rooms, _ = ls.list_rooms(limit=500)
        selected_room_ids = [r.id for r in all_rooms] if all_rooms else [None]

    # ── Resolve cameras ───────────────────────────────────────────────
    if camera_id is not None:
        cam = get_runtime().registry.get(camera_id)
        if cam is None:
            raise HTTPException(404, "دوربین یافت نشد")
        selected_camera_ids = [cam.source_id]
    else:
        all_cams = get_runtime().registry.list()
        selected_camera_ids = [c.source_id for c in all_cams] if all_cams else [None]

    created = 0

    for _ in range(count):
        person = _random.choice(selected_personnel)
        rid = _random.choice(selected_room_ids)
        cid = _random.choice(selected_camera_ids) if selected_camera_ids else None

        # ── Pick detection time ──────────────────────────────────────
        if range_start is not None and range_end is not None:
            delta_days = (range_end - range_start).days or 1
            det_date = range_start + timedelta(days=_random.randint(0, delta_days))
            det_time = datetime(
                det_date.year, det_date.month, det_date.day,
                hour=_random.randint(0, 23),
                minute=_random.randint(0, 59),
                second=_random.randint(0, 59),
                tzinfo=timezone.utc,
            )
        else:
            det_time = base_time - timedelta(
                days=_random.randint(0, 180),
                hours=_random.randint(0, 23),
                minutes=_random.randint(0, 59),
            )

        confidence = round(_random.uniform(0.5, 1.0), 4)

        if person.id is not None and rid is not None:
            access = calculate_access(person.id, rid, ls)
        else:
            access = _random.random() > 0.2

        def _create_one(dt: datetime, acc: bool) -> bool:
            try:
                store.create(
                    source_system="generate_fake",
                    personnel_id=person.id,
                    person=f"{person.fname} {person.lname}",
                    confidence=confidence,
                    detection_time=dt.isoformat(),
                    room_id=rid,
                    camera_id=cid,
                    access_granted=acc,
                    counts_for_attendance=_random.random() > 0.3,
                    log_type=_random.choice(["camera_rtsp", "tehran_door", "excel_import"]),
                )
                return True
            except (ValueError, Exception):
                return False

        if _create_one(det_time, access):
            created += 1

        # ── Paired (exit) log if requested ──────────────────────────
        if pair_logs:
            exit_time = det_time + timedelta(minutes=_random.randint(30, 480))
            exit_access = _random.random() > 0.2
            if _create_one(exit_time, exit_access):
                created += 1

    return {"message": f"{created} لاگ آزمایشی ایجاد شد", "count": created}


# ── Parameterized routes ──────────────────────────────────────────────


@router.get("/{log_id}")
def get_log(
    log_id: int,
    _: dict = Depends(require_role("operator")),
) -> dict:
    store = get_detection_log_store()
    record = store.get(log_id)
    if record is None:
        raise HTTPException(404, "لاگ یافت نشد")
    return _build_response(record, include_detail=True)


@router.patch("/{log_id}/person")
def patch_log_person(
    log_id: int,
    body: dict[str, Any],
    current_user: dict = Depends(require_role("admin")),
) -> dict:
    store = get_detection_log_store()
    record = store.get(log_id)
    if record is None:
        raise HTTPException(404, "لاگ یافت نشد")

    kwargs: dict[str, Any] = {}
    if "person" in body:
        kwargs["person"] = str(body["person"])
    if "personnel_id" in body:
        kwargs["personnel_id"] = int(body["personnel_id"])
    if "confidence" in body:
        kwargs["confidence"] = float(body["confidence"])
    kwargs["updated_by"] = _get_current_user_id(current_user)

    updated = store.update(log_id, **kwargs)
    if updated is None:
        raise HTTPException(404, "لاگ یافت نشد")
    _push_refresh_for_log(get_runtime(), log_id)
    return _build_response(updated, include_detail=True)


@router.patch("/{log_id}/attendance")
def patch_log_attendance(
    log_id: int,
    body: dict[str, Any],
    current_user: dict = Depends(require_role("admin")),
) -> dict:
    store = get_detection_log_store()
    record = store.get(log_id)
    if record is None:
        raise HTTPException(404, "لاگ یافت نشد")

    counts = bool(body.get("counts_for_attendance", False))
    updated = store.update(
        log_id,
        counts_for_attendance=counts,
        updated_by=_get_current_user_id(current_user),
    )
    if updated is None:
        raise HTTPException(404, "لاگ یافت نشد")
    _push_refresh_for_log(get_runtime(), log_id)
    return _build_response(updated, include_detail=True)


@router.get("/{log_id}/thumbnail")
def get_thumbnail(
    log_id: int,
    _: dict = Depends(require_role("operator")),
) -> FileResponse:
    store = get_detection_log_store()
    record = store.get(log_id)
    if record is None:
        raise HTTPException(404, "لاگ یافت نشد")
    fpath = _resolve_media_path(record.snapshot_image)
    if fpath is None or not fpath.exists():
        raise HTTPException(404, "پرونده یافت نشد")
    return FileResponse(str(fpath), filename=fpath.name)


@router.get("/{log_id}/face")
def get_face(
    log_id: int,
    _: dict = Depends(require_role("operator")),
) -> FileResponse:
    store = get_detection_log_store()
    record = store.get(log_id)
    if record is None:
        raise HTTPException(404, "لاگ یافت نشد")
    fpath = _resolve_media_path(record.face_image)
    if fpath is None or not fpath.exists():
        raise HTTPException(404, "پرونده یافت نشد")
    return FileResponse(str(fpath), filename=fpath.name)


@router.get("/{log_id}/body")
def get_body(
    log_id: int,
    _: dict = Depends(require_role("operator")),
) -> FileResponse:
    store = get_detection_log_store()
    record = store.get(log_id)
    if record is None:
        raise HTTPException(404, "لاگ یافت نشد")
    fpath = _resolve_media_path(record.body_image)
    if fpath is None or not fpath.exists():
        raise HTTPException(404, "پرونده یافت نشد")
    return FileResponse(str(fpath), filename=fpath.name)


@router.get("/{log_id}/snapshot")
def get_snapshot(
    log_id: int,
    _: dict = Depends(require_role("operator")),
) -> FileResponse:
    store = get_detection_log_store()
    record = store.get(log_id)
    if record is None:
        raise HTTPException(404, "لاگ یافت نشد")
    fpath = _resolve_media_path(record.snapshot_image)
    if fpath is None or not fpath.exists():
        raise HTTPException(404, "پرونده یافت نشد")
    return FileResponse(str(fpath), filename=fpath.name)


@router.get("/{log_id}/video")
def get_video(
    log_id: int,
    _: dict = Depends(require_role("operator")),
) -> FileResponse:
    store = get_detection_log_store()
    record = store.get(log_id)
    if record is None:
        raise HTTPException(404, "لاگ یافت نشد")
    fpath = _resolve_media_path(record.video)
    if fpath is None or not fpath.exists():
        raise HTTPException(404, "پرونده یافت نشد")
    return FileResponse(str(fpath), filename=fpath.name)


@router.get("/{log_id}/face-video")
def get_face_video(
    log_id: int,
    _: dict = Depends(require_role("operator")),
) -> FileResponse:
    store = get_detection_log_store()
    record = store.get(log_id)
    if record is None:
        raise HTTPException(404, "لاگ یافت نشد")
    fpath = _resolve_media_path(record.face_video_or_unknown_faces)
    if fpath is None or not fpath.exists():
        raise HTTPException(404, "پرونده یافت نشد")
    return FileResponse(str(fpath), filename=fpath.name)


@router.delete("/{log_id}")
def delete_log(
    log_id: int,
    _: dict = Depends(require_role("superuser")),
) -> dict:
    store = get_detection_log_store()
    record = store.get(log_id)
    if record is None:
        raise HTTPException(404, "لاگ یافت نشد")
    _delete_media_files(record)
    deleted = store.delete(log_id)
    if not deleted:
        raise HTTPException(404, "لاگ یافت نشد")
    return {
        "message": "لاگ با موفقیت حذف شد",
        "log_id": log_id,
    }
