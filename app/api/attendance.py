"""API router for Attendance Reports."""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import require_role
from app.core.attendance_service import AttendanceService

router = APIRouter(prefix="/api/v1/attendance", tags=["Attendance"])


def get_runtime() -> Any:
    from app.main import runtime
    return runtime


def get_attendance_service() -> AttendanceService:
    return get_runtime().attendance_service


@router.get("/daily/{personnel_id}")
def daily_summary(
    personnel_id: int,
    local_date: str = Query(..., description="Date in Gregorian ISO or Jalali format"),
    _: dict = Depends(require_role("operator")),
) -> dict[str, Any]:
    """Compute daily attendance summary for a personnel member."""
    svc = get_attendance_service()
    try:
        result = svc.compute_daily_summary(personnel_id, local_date)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


@router.get("/monthly/{personnel_id}")
def monthly_summary(
    personnel_id: int,
    year: int = Query(..., ge=1400, le=1500),
    month: int = Query(..., ge=1, le=12),
    calendar: str = Query("gregorian", pattern="^(gregorian|jalali)$"),
    _: dict = Depends(require_role("operator")),
) -> dict[str, Any]:
    """Compute monthly attendance summary for a personnel member."""
    svc = get_attendance_service()
    try:
        result = svc.compute_monthly_summary(personnel_id, year, month, calendar)
    except (ValueError, IndexError) as exc:
        raise HTTPException(400, str(exc))
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


@router.get("/monthly-performance")
def monthly_performance(
    year: int = Query(..., ge=1400, le=1500),
    month: int = Query(..., ge=1, le=12),
    calendar: str = Query("gregorian", pattern="^(gregorian|jalali)$"),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    _: dict = Depends(require_role("operator")),
) -> dict[str, Any]:
    """Compute monthly performance for all personnel."""
    svc = get_attendance_service()
    try:
        result = svc.compute_monthly_performance(year, month, calendar, offset, limit)
    except (ValueError, IndexError) as exc:
        raise HTTPException(400, str(exc))
    return result


@router.get("/yearly-leave/{personnel_id}")
def yearly_leave_summary(
    personnel_id: int,
    year: int = Query(..., ge=1400, le=1500),
    calendar: str = Query("gregorian", pattern="^(gregorian|jalali)$"),
    _: dict = Depends(require_role("operator")),
) -> dict[str, Any]:
    """Compute yearly leave/sick-leave usage summary."""
    svc = get_attendance_service()
    try:
        result = svc.compute_yearly_leave_summary(personnel_id, year, calendar)
    except (ValueError, IndexError) as exc:
        raise HTTPException(400, str(exc))
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


@router.put("/log/{log_id}/toggle-attendance")
def toggle_log_attendance(
    log_id: int,
    body: dict[str, Any],
    _: dict = Depends(require_role("admin")),
) -> dict[str, Any]:
    """Toggle whether a human_log counts for attendance."""
    counts = body.get("counts_for_attendance", True)
    svc = get_attendance_service()
    result = svc.set_counts_for_attendance(log_id, counts)
    if result is None:
        raise HTTPException(404, "Log not found")
    return result
