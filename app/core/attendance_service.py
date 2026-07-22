"""Attendance computation service using shifts, holidays, requests, and human logs."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

from app.core.holiday_store import HolidayStore
from app.core.human_log_store import HumanLogStore
from app.core.jalali_utils import (
    gregorian_to_jalali,
    gregorian_to_jalali_str,
    jalali_month_utc_range,
    local_day_utc_range,
    parse_jalali_date,
)
from app.core.personnel_store import PersonnelRecord, PersonnelStore
from app.core.request_store import RequestStore
from app.core.shift_store import (
    WEEKDAY_COLS,
    WEEKDAY_NAMES,
    WorkShiftRecord,
    _get_weekday_flag,
    _is_overnight,
    _weekday_from_local,
)

LOGGER = logging.getLogger(__name__)

# Minimum detection duration (in seconds) for counts_for_attendance
MIN_ATTENDANCE_SECONDS = 60


def _tehran_tz() -> timezone:
    from app.core.jalali_utils import _get_tehran_tz
    return _get_tehran_tz()


class AttendanceService:
    """Compute daily/monthly/yearly attendance records from logs and configuration."""

    def __init__(
        self,
        personnel_store: PersonnelStore,
        human_log_store: HumanLogStore,
        shift_store: Any,  # ShiftStore
        holiday_store: HolidayStore,
        request_store: RequestStore,
    ) -> None:
        self._personnel_store = personnel_store
        self._human_log_store = human_log_store
        self._shift_store = shift_store
        self._holiday_store = holiday_store
        self._request_store = request_store

    # ── Helpers ───────────────────────────────────────────────────────

    def _get_personnel(self, personnel_id: int) -> PersonnelRecord | None:
        return self._personnel_store.get(personnel_id)

    def _get_shift(self, shift_id: int | None) -> WorkShiftRecord | None:
        if shift_id is None:
            return None
        return self._shift_store.get(shift_id)

    @staticmethod
    def _is_working_day(shift: WorkShiftRecord, local_date: date) -> bool:
        """Check if the shift schedules work on a given local date."""
        weekday_idx = _weekday_from_local(local_date)
        return _get_weekday_flag(shift, weekday_idx)

    def _is_holiday(self, d: date, personnel: PersonnelRecord | None = None) -> bool:
        """Check if a date is a holiday (global or company-specific)."""
        return self._holiday_store.is_holiday(d)

    def _get_detection_logs(
        self, personnel_id: int, utc_start: datetime, utc_end: datetime,
    ) -> list[dict[str, Any]]:
        """Return human_logs rows for a personnel within UTC range."""
        return self._human_log_store.get_logs_for_personnel(
            personnel_id, utc_start.isoformat(), utc_end.isoformat(),
        )

    # ── Daily attendance ──────────────────────────────────────────────

    def compute_daily_summary(
        self, personnel_id: int, local_date: str | date,
    ) -> dict[str, Any]:
        """Compute one day's attendance summary for a personnel member.

        Args:
            personnel_id: Personnel ID
            local_date: Date in Gregorian ISO format or Jalali string

        Returns:
            dict with status, first_seen, last_seen, total_seconds,
            shift_seconds, delay_minutes, early_leave_minutes,
            overtime_minutes, is_holiday, day_type, logs.
        """
        # Resolve date
        if isinstance(local_date, str):
            try:
                d = date.fromisoformat(local_date)
            except ValueError:
                d = parse_jalali_date(local_date)
        else:
            d = local_date

        personnel = self._get_personnel(personnel_id)
        if personnel is None:
            return {"error": f"Personnel not found: {personnel_id}"}

        shift = self._get_shift(personnel.shift_id)
        is_holiday = self._is_holiday(d, personnel)

        # Determine day type
        if is_holiday:
            day_type = "holiday"
        elif shift is not None and not self._is_working_day(shift, d):
            day_type = "non_working_day"
        else:
            day_type = "working_day"

        # UTC range for the local date
        utc_start, utc_end = local_day_utc_range(d, _tehran_tz())

        # Get logs
        logs = self._get_detection_logs(personnel_id, utc_start, utc_end)

        # Compute presence
        if not logs:
            return {
                "personnel_id": personnel_id,
                "date": d.isoformat(),
                "jalali_date": gregorian_to_jalali_str(d),
                "day_type": day_type,
                "is_holiday": is_holiday,
                "status": "absent",
                "first_seen": None,
                "last_seen": None,
                "total_seconds": 0,
                "expected_seconds": 0,
                "delay_minutes": 0,
                "early_leave_minutes": 0,
                "overtime_minutes": 0,
                "log_count": 0,
            }

        first_seen = min(log["last_seen"] for log in logs)
        last_seen = max(log["last_seen"] for log in logs)
        total_seconds = self._compute_presence_seconds(logs, utc_start, utc_end)

        delay_minutes = 0
        early_leave_minutes = 0
        overtime_minutes = 0
        expected_seconds = 0

        if shift is not None and day_type == "working_day":
            expected_seconds = self._compute_expected_seconds(shift, d)
            delay_minutes = self._compute_delay_minutes(first_seen, shift, d)
            early_leave_minutes = self._compute_early_leave_minutes(last_seen, shift, d)
            overtime_minutes = self._compute_overtime_minutes(
                logs, shift, d, utc_start, utc_end, expected_seconds,
            )

        status = "present"
        if total_seconds < MIN_ATTENDANCE_SECONDS:
            status = "insufficient"

        return {
            "personnel_id": personnel_id,
            "date": d.isoformat(),
            "jalali_date": gregorian_to_jalali_str(d),
            "day_type": day_type,
            "is_holiday": is_holiday,
            "status": status,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "total_seconds": total_seconds,
            "expected_seconds": expected_seconds,
            "delay_minutes": max(0, delay_minutes),
            "early_leave_minutes": max(0, early_leave_minutes),
            "overtime_minutes": max(0, overtime_minutes),
            "log_count": len(logs),
        }

    # ── Monthly attendance ────────────────────────────────────────────

    def compute_monthly_summary(
        self, personnel_id: int, year: int, month: int,
        calendar: str = "gregorian",
    ) -> dict[str, Any]:
        """Compute monthly attendance summary.

        Args:
            personnel_id: Personnel ID
            year: Year
            month: Month (1-12)
            calendar: "gregorian" or "jalali"

        Returns:
            Monthly summary dict with daily breakdown, totals, and statistics.
        """
        personnel = self._get_personnel(personnel_id)
        if personnel is None:
            return {"error": f"Personnel not found: {personnel_id}"}

        if calendar == "jalali":
            utc_start, utc_end = jalali_month_utc_range(year, month)
            # Derive Gregorian month for display
            g_start = utc_start.date()
            display_label = f"{year:04d}-{month:02d} (Jalali)"
        else:
            import calendar as cal_mod
            last_day = cal_mod.monthrange(year, month)[1]
            utc_start, utc_end = local_day_utc_range(
                date(year, month, 1), _tehran_tz()
            )
            utc_end = local_day_utc_range(
                date(year, month, last_day), _tehran_tz()
            )[1]
            g_start = date(year, month, 1)
            display_label = f"{year:04d}-{month:02d}"

        # Get all logs for the month
        logs = self._get_detection_logs(personnel_id, utc_start, utc_end)

        # Group logs by local date
        logs_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for log in logs:
            # Convert UTC timestamp to local date
            log_utc = log["last_seen"]
            try:
                log_dt = datetime.fromisoformat(log_utc.replace("Z", "+00:00"))
                local_dt = log_dt.astimezone(_tehran_tz())
                local_date_str = local_dt.date().isoformat()
            except (ValueError, TypeError):
                local_date_str = log_utc[:10]
            logs_by_date[local_date_str].append(log)

        # Compute daily summaries
        daily: list[dict[str, Any]] = []
        current = g_start
        while current <= date.fromisoformat(utc_end.date().isoformat()):
            if current > utc_end.date():
                break
            day_logs = logs_by_date.get(current.isoformat(), [])
            daily.append(self._compute_single_day(personnel_id, personnel, current, day_logs))
            current += timedelta(days=1)

        # Aggregate
        present_days = sum(1 for d in daily if d["status"] == "present")
        absent_days = sum(1 for d in daily if d["status"] == "absent")
        insufficient_days = sum(1 for d in daily if d["status"] == "insufficient")
        total_seconds = sum(d["total_seconds"] for d in daily)
        total_overtime = sum(d["overtime_minutes"] for d in daily)
        total_delay = sum(d["delay_minutes"] for d in daily)
        total_early_leave = sum(d["early_leave_minutes"] for d in daily)

        return {
            "personnel_id": personnel_id,
            "personnel_name": f"{personnel.fname} {personnel.lname}",
            "period": display_label,
            "calendar": calendar,
            "total_days": len(daily),
            "present_days": present_days,
            "absent_days": absent_days,
            "insufficient_days": insufficient_days,
            "total_seconds": total_seconds,
            "total_overtime_minutes": total_overtime,
            "total_delay_minutes": total_delay,
            "total_early_leave_minutes": total_early_leave,
            "daily": daily,
        }

    def compute_monthly_performance(
        self, year: int, month: int, calendar: str = "gregorian",
        offset: int = 0, limit: int = 50,
    ) -> dict[str, Any]:
        """Compute monthly performance for all personnel."""
        all_personnel, total_count = self._personnel_store.list(
            offset=0, limit=10000
        )
        results: list[dict[str, Any]] = []
        for person in all_personnel:
            summary = self.compute_monthly_summary(person.id, year, month, calendar)
            if "error" in summary:
                continue
            results.append({
                "personnel_id": person.id,
                "personnel_name": f"{person.fname} {person.lname}",
                "present_days": summary["present_days"],
                "absent_days": summary["absent_days"],
                "total_seconds": summary["total_seconds"],
                "total_overtime_minutes": summary["total_overtime_minutes"],
                "total_delay_minutes": summary["total_delay_minutes"],
                "total_early_leave_minutes": summary["total_early_leave_minutes"],
            })
        # Sort by present_days descending, then by name
        results.sort(key=lambda r: (-r["present_days"], r["personnel_name"]))
        paged = results[offset:offset + limit]
        return {
            "year": year,
            "month": month,
            "calendar": calendar,
            "total_personnel": len(results),
            "limit": limit,
            "offset": offset,
            "results": paged,
        }

    # ── Yearly leave summary ──────────────────────────────────────────

    def compute_yearly_leave_summary(
        self, personnel_id: int, year: int, calendar: str = "gregorian",
    ) -> dict[str, Any]:
        """Compute yearly leave/sick-leave usage summary."""
        personnel = self._get_personnel(personnel_id)
        if personnel is None:
            return {"error": f"Personnel not found: {personnel_id}"}

        if calendar == "jalali":
            j_first = date(year, 1, 1)  # approximate — we use jalali range
            utc_start, utc_end = jalali_month_utc_range(year, 1)
            utc_s, utc_e = jalali_month_utc_range(year, 12)
            year_start = utc_start
            year_end = utc_e
            # Get approved requests in range
            range_start = year_start.date()
            range_end = year_end.date()
        else:
            range_start = date(year, 1, 1)
            range_end = date(year, 12, 31)
            year_start, year_end = local_day_utc_range(range_start, _tehran_tz())[0], \
                local_day_utc_range(range_end, _tehran_tz())[1]

        # Get approved requests
        requests = self._request_store.get_approved_requests_in_range(
            personnel_id, range_start, range_end,
        )

        total_leave_days = 0
        total_sick_days = 0
        total_mission_days = 0
        total_remote_days = 0
        total_personal_days = 0
        total_unpaid_days = 0
        total_overtime_hours = 0
        leave_details: list[dict[str, Any]] = []

        for req in requests:
            req_start = date.fromisoformat(req.start_date)
            req_end = date.fromisoformat(req.end_date)
            days = (req_end - req_start).days + 1
            rt = req.request_type
            if rt in ("leave", "earned_leave"):
                total_leave_days += days
            elif rt in ("sick_leave",):
                total_sick_days += days
            elif rt in ("mission",):
                total_mission_days += days
            elif rt in ("remote_work",):
                total_remote_days += days
            elif rt in ("personal",):
                total_personal_days += days
            elif rt in ("unpaid_leave",):
                total_unpaid_days += days
            elif rt in ("overtime",):
                total_overtime_hours += (req.duration_minutes or 0) / 60
            leave_details.append({
                "request_id": req.id,
                "request_type": rt,
                "start_date": req.start_date,
                "end_date": req.end_date,
                "days": days,
                "reason": req.reason,
            })

        total_days_off = total_leave_days + total_sick_days + total_personal_days

        return {
            "personnel_id": personnel_id,
            "personnel_name": f"{personnel.fname} {personnel.lname}",
            "year": year,
            "calendar": calendar,
            "total_leave_days": total_leave_days,
            "total_sick_days": total_sick_days,
            "total_mission_days": total_mission_days,
            "total_remote_days": total_remote_days,
            "total_personal_days": total_personal_days,
            "total_unpaid_days": total_unpaid_days,
            "total_overtime_hours": total_overtime_hours,
            "total_days_off": total_days_off,
            "details": leave_details,
        }

    # ── Log attendance toggle ─────────────────────────────────────────

    def set_counts_for_attendance(
        self, log_id: int, counts: bool,
    ) -> dict[str, Any] | None:
        """Toggle whether a specific human_log counts for attendance."""
        updated = self._human_log_store.set_counts_for_attendance(log_id, counts)
        if updated is None:
            return None
        return {
            "log_id": log_id,
            "counts_for_attendance": counts,
        }

    # ── Internal computation helpers ──────────────────────────────────

    def _compute_single_day(
        self, personnel_id: int, personnel: PersonnelRecord,
        d: date, logs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        shift = self._get_shift(personnel.shift_id)
        is_holiday = self._is_holiday(d, personnel)

        if is_holiday:
            day_type = "holiday"
        elif shift is not None and not self._is_working_day(shift, d):
            day_type = "non_working_day"
        else:
            day_type = "working_day"

        if not logs:
            return {
                "date": d.isoformat(),
                "jalali_date": gregorian_to_jalali_str(d),
                "day_type": day_type,
                "is_holiday": is_holiday,
                "status": "absent",
                "first_seen": None,
                "last_seen": None,
                "total_seconds": 0,
                "delay_minutes": 0,
                "early_leave_minutes": 0,
                "overtime_minutes": 0,
                "log_count": 0,
            }

        utc_start, utc_end = local_day_utc_range(d, _tehran_tz())
        first_seen = min(log["last_seen"] for log in logs)
        last_seen = max(log["last_seen"] for log in logs)
        total_seconds = self._compute_presence_seconds(logs, utc_start, utc_end)

        delay_minutes = 0
        early_leave_minutes = 0
        overtime_minutes = 0

        if shift is not None and day_type == "working_day":
            delay_minutes = self._compute_delay_minutes(first_seen, shift, d)
            early_leave_minutes = self._compute_early_leave_minutes(last_seen, shift, d)
            expected_seconds = self._compute_expected_seconds(shift, d)
            overtime_minutes = self._compute_overtime_minutes(
                logs, shift, d, utc_start, utc_end, expected_seconds,
            )

        status = "present"
        if total_seconds < MIN_ATTENDANCE_SECONDS:
            status = "insufficient"

        return {
            "date": d.isoformat(),
            "jalali_date": gregorian_to_jalali_str(d),
            "day_type": day_type,
            "is_holiday": is_holiday,
            "status": status,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "total_seconds": total_seconds,
            "delay_minutes": max(0, delay_minutes),
            "early_leave_minutes": max(0, early_leave_minutes),
            "overtime_minutes": max(0, overtime_minutes),
            "log_count": len(logs),
        }

    @staticmethod
    def _compute_presence_seconds(
        logs: list[dict[str, Any]],
        utc_start: datetime,
        utc_end: datetime,
    ) -> int:
        """Compute total presence seconds from overlapping detection intervals.

        This is a simplified approach:
        - Since logs are for unique (session, camera, track_id), each log
          covers the range [first_seen, last_seen].
        - We merge all overlapping intervals and sum their durations.
        - Clamp to [utc_start, utc_end].
        """
        intervals: list[tuple[float, float]] = []
        for log in logs:
            try:
                fs = datetime.fromisoformat(log["first_seen"].replace("Z", "+00:00")).timestamp()
                ls = datetime.fromisoformat(log["last_seen"].replace("Z", "+00:00")).timestamp()
            except (ValueError, TypeError):
                continue
            start_ts = max(fs, utc_start.timestamp())
            end_ts = min(ls, utc_end.timestamp())
            if start_ts < end_ts:
                intervals.append((start_ts, end_ts))

        if not intervals:
            return 0

        # Merge overlapping intervals
        intervals.sort()
        merged: list[tuple[float, float]] = [intervals[0]]
        for start, end in intervals[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        total = sum(max(0.0, end - start) for start, end in merged)
        return int(round(total))

    @staticmethod
    def _compute_expected_seconds(shift: WorkShiftRecord, d: date) -> int:
        """Compute expected working seconds for a shift on a given date."""
        start_h, start_m = (int(x) for x in shift.start_time.split(":"))
        end_h, end_m = (int(x) for x in shift.end_time.split(":"))
        start_min = start_h * 60 + start_m
        end_min = end_h * 60 + end_m
        if _is_overnight(shift.start_time, shift.end_time):
            end_min += 24 * 60  # next day
        return (end_min - start_min) * 60

    def _compute_delay_minutes(
        self, first_seen_utc: str, shift: WorkShiftRecord, d: date,
    ) -> int:
        """Compute delay in minutes (how late the person arrived)."""
        try:
            fs = datetime.fromisoformat(first_seen_utc.replace("Z", "+00:00"))
            local_fs = fs.astimezone(_tehran_tz())
        except (ValueError, TypeError):
            return 0

        start_h, start_m = (int(x) for x in shift.start_time.split(":"))
        shift_start_local = datetime(
            local_fs.year, local_fs.month, local_fs.day,
            start_h, start_m, 0, tzinfo=_tehran_tz(),
        )
        delay_seconds = (local_fs - shift_start_local).total_seconds()
        delay_minutes = int(delay_seconds // 60)
        return max(0, delay_minutes - shift.max_minutes_delay)

    def _compute_early_leave_minutes(
        self, last_seen_utc: str, shift: WorkShiftRecord, d: date,
    ) -> int:
        """Compute early leave in minutes."""
        try:
            ls = datetime.fromisoformat(last_seen_utc.replace("Z", "+00:00"))
            local_ls = ls.astimezone(_tehran_tz())
        except (ValueError, TypeError):
            return 0

        end_h, end_m = (int(x) for x in shift.end_time.split(":"))
        if _is_overnight(shift.start_time, shift.end_time):
            end_h += 24
        shift_end_local = datetime(
            local_ls.year, local_ls.month, local_ls.day,
            end_h % 24, end_m, 0, tzinfo=_tehran_tz(),
        )
        if _is_overnight(shift.start_time, shift.end_time):
            shift_end_local += timedelta(days=1)

        early_seconds = (shift_end_local - local_ls).total_seconds()
        early_minutes = int(early_seconds // 60)
        return max(0, early_minutes - shift.max_minutes_early)

    def _compute_overtime_minutes(
        self, logs: list[dict[str, Any]], shift: WorkShiftRecord,
        d: date, utc_start: datetime, utc_end: datetime,
        expected_seconds: int,
    ) -> int:
        """Compute overtime in minutes beyond the expected shift duration."""
        total_seconds = self._compute_presence_seconds(logs, utc_start, utc_end)
        overtime_seconds = max(0, total_seconds - expected_seconds)
        # Cap at max_overtime_hours
        max_overtime_seconds = shift.max_overtime_hours * 3600
        overtime_seconds = min(overtime_seconds, max_overtime_seconds)
        return int(overtime_seconds // 60)
