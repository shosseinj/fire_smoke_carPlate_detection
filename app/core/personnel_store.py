from __future__ import annotations

from app.database import Connection, Database, IntegrityError, OperationalError, Row, ensure_database
from app.time_utils import utc_now_text

import base64
import io
import json
import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings

LOGGER = logging.getLogger("uvicorn.error")

_VALID_EMPLOYEE_TYPES = frozenset({"contractor", "customer", "guest", "employee", "unknown"})
_NATIONAL_CODE_CONSTRAINT = "personnel_national_code_key"
_DEPARTMENT_CONSTRAINT = "personnel_department_id_fkey"
_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


@dataclass(frozen=True, slots=True)
class PersonnelRecord:
    id: int
    fname: str
    lname: str
    national_code: str
    employee_type: str
    degree: str | None
    shift_id: int | None
    department_id: int | None
    last_seen: str | None
    created_at_utc: str
    updated_at_utc: str
    created_by: int | None = None
    updated_by: int | None = None


@dataclass(frozen=True, slots=True)
class PersonnelImageRecord:
    id: int
    personnel_id: int
    storage_key: str
    description: str | None
    is_primary: bool
    uploaded_at_utc: str
    embedding_id: str | None


def normalize_national_code(raw: str) -> str:
    normalized = raw.strip().translate(_PERSIAN_DIGITS)
    normalized = re.sub(r"[^\d]", "", normalized)
    return normalized


def validate_national_code(code: str) -> bool:
    if not re.match(r"^\d{10}$", code):
        return False
    if len(set(code)) == 1:
        return False
    digits = [int(d) for d in code]
    checksum = digits[-1]
    total = sum(digits[i] * (10 - i) for i in range(9))
    remainder = total % 11
    if remainder < 2:
        expected = remainder
    else:
        expected = 11 - remainder
    return expected == checksum


def _personnel_integrity_message(
    exc: IntegrityError,
    *,
    national_code: str,
    department_id: int | None,
) -> str:
    original = getattr(exc, "orig", None)
    diagnostic = getattr(original, "diag", None)
    constraint_name = getattr(diagnostic, "constraint_name", None)
    if constraint_name == _NATIONAL_CODE_CONSTRAINT:
        return f"National code already exists: {national_code}"
    if constraint_name == _DEPARTMENT_CONSTRAINT:
        return f"Department not found: {department_id}"
    return "Personnel data violates a database constraint"


def _parse_zip_entry_personnel(name: str) -> tuple[str, str, str]:
    """Extract (fname, lname, national_code_candidate) from a ZIP entry path.

    Folder-per-person structure:  {national_code}/{filename}
      → (Unknown, Unknown, national_code)

    Underscore filename pattern:  {fname}_{lname}_{national_code}.ext
      → (fname, lname, national_code)

    Bare numeric stem:  {national_code}.ext
      → (Unknown, Unknown, national_code)

    Returns empty string for national_code when no candidate can be determined.
    """
    parts = Path(name).parts
    img_filename = Path(name).name
    stem = img_filename.rsplit(".", 1)[0] if "." in img_filename else img_filename

    if len(parts) >= 2:
        # Inside a directory — use the first path component as national_code
        # Expected structure: {national_code}/{image_file}
        nc_candidate = normalize_national_code(parts[0])
        return "Unknown", "Unknown", nc_candidate

    # Flat file — underscore pattern: fname_lname_nationalCode.ext
    parts_underscore = stem.split("_")
    if len(parts_underscore) >= 3:
        fc = parts_underscore[0]
        lc = "_".join(parts_underscore[1:-1])
        nc_candidate = normalize_national_code(parts_underscore[-1])
        return fc, lc, nc_candidate

    # Bare numeric stem
    if stem.isdigit():
        nc_candidate = normalize_national_code(stem)
        return "Unknown", "Unknown", nc_candidate

    return "Unknown", "Unknown", ""


class PersonnelStore:
    """PostgreSQL-backed store for personnel records and images."""

    def __init__(self, database: Database | str, saved_media_path: Path) -> None:
        self._database = ensure_database(database)
        self._media_root = saved_media_path.resolve()
        self._snapshot_dir = self._media_root / "personnel_snapshots"
        self._cropped_face_dir = self._media_root / "personnel_cropped_faces"
        self._zip_errors_dir = self._media_root / "personnel_zip_errors"
        self._cropped_face_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connection(self) -> Connection:
        return self._database.connection()

    def _init_db(self) -> None:
        return None

    def _now(self) -> str:
        return utc_now_text()

    # ── Helpers ─────────────────────────────────────────────────────

    def _resolve_shift_name(self, shift_id: int | None) -> str | None:
        if shift_id is None:
            return None
        try:
            with self._connection() as conn:
                row = conn.execute(
                    "SELECT shift_name FROM work_shifts WHERE id = ?", (shift_id,)
                ).fetchone()
                return str(row["shift_name"]) if row else None
        except (OperationalError, Exception):
            return None

    def _resolve_department_name(self, department_id: int | None) -> str | None:
        if department_id is None:
            return None
        try:
            with self._connection() as conn:
                row = conn.execute(
                    "SELECT name FROM sections WHERE id = ?", (department_id,)
                ).fetchone()
                return str(row["name"]) if row else None
        except (OperationalError, Exception):
            return None

    def _resolve_shift_names_bulk(self, shift_ids: set[int | None]) -> dict[int | None, str | None]:
        """Resolve multiple shift names in one query."""
        result: dict[int | None, str | None] = {None: None}
        ids = [sid for sid in shift_ids if sid is not None]
        if not ids:
            return result
        try:
            placeholders = ", ".join("?" for _ in ids)
            with self._connection() as conn:
                rows = conn.execute(
                    f"SELECT id, shift_name FROM work_shifts WHERE id IN ({placeholders})", ids
                ).fetchall()
                for row in rows:
                    result[int(row["id"])] = str(row["shift_name"])
                for sid in ids:
                    if sid not in result:
                        result[sid] = None
        except (OperationalError, Exception):
            for sid in ids:
                result[sid] = None
        return result

    def _resolve_department_names_bulk(self, dept_ids: set[int | None]) -> dict[int | None, str | None]:
        result: dict[int | None, str | None] = {None: None}
        ids = [did for did in dept_ids if did is not None]
        if not ids:
            return result
        try:
            placeholders = ", ".join("?" for _ in ids)
            with self._connection() as conn:
                rows = conn.execute(
                    f"SELECT id, name FROM sections WHERE id IN ({placeholders})", ids
                ).fetchall()
                for row in rows:
                    result[int(row["id"])] = str(row["name"])
                for did in ids:
                    if did not in result:
                        result[did] = None
        except (OperationalError, Exception):
            for did in ids:
                result[did] = None
        return result

    # ── Personnel CRUD ─────────────────────────────────────────────────

    def _row_to_personnel(self, row: Row) -> PersonnelRecord:
        last_seen: str | None = None
        if "last_seen" in row.keys():
            last_seen = row["last_seen"]
        shift_id: int | None = None
        if "shift_id" in row.keys():
            raw = row["shift_id"]
            shift_id = int(raw) if raw is not None else None
        department_id: int | None = None
        if "department_id" in row.keys():
            raw = row["department_id"]
            department_id = int(raw) if raw is not None else None
        return PersonnelRecord(
            id=row["id"],
            fname=row["fname"],
            lname=row["lname"],
            national_code=row["national_code"],
            employee_type=row["employee_type"],
            degree=row["degree"],
            shift_id=shift_id,
            department_id=department_id,
            last_seen=last_seen,
            created_at_utc=row["created_at_utc"],
            updated_at_utc=row["updated_at_utc"],
            created_by=row.get("created_by"),
            updated_by=row.get("updated_by"),
        )

    def _personnel_columns(self) -> str:
        return ("id, fname, lname, national_code, employee_type, degree, "
                "shift_id, department_id, last_seen, created_at_utc, updated_at_utc, "
                "created_by, updated_by")

    def create(
        self,
        fname: str,
        lname: str,
        national_code: str,
        employee_type: str = "unknown",
        degree: str | None = None,
        shift_id: int | None = None,
        department_id: int | None = None,
        created_by: int | None = None,
    ) -> PersonnelRecord:
        fname = fname.strip()
        lname = lname.strip()
        if not fname:
            raise ValueError("fname is required")
        raw_code = normalize_national_code(national_code)
        if not validate_national_code(raw_code):
            raise ValueError(f"Invalid Iranian national code: {national_code}")
        if employee_type not in _VALID_EMPLOYEE_TYPES:
            raise ValueError(
                f"Invalid employee_type '{employee_type}'. "
                f"Must be one of: {sorted(_VALID_EMPLOYEE_TYPES)}"
            )
        now = self._now()
        with self._lock, self._connection() as conn:
            try:
                cursor = conn.execute(
                    "INSERT INTO personnel "
                    "(fname, lname, national_code, employee_type, degree, shift_id, department_id, "
                    "created_at_utc, updated_at_utc, created_by) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (fname, lname, raw_code, employee_type, degree, shift_id, department_id, now, now, created_by),
                )
                row = conn.execute(
                    f"SELECT {self._personnel_columns()} FROM personnel WHERE id = ?", (cursor.lastrowid,)
                ).fetchone()
                if row is None:
                    raise RuntimeError("Failed to retrieve created personnel record")
                return self._row_to_personnel(row)
            except IntegrityError as exc:
                raise ValueError(
                    _personnel_integrity_message(
                        exc,
                        national_code=raw_code,
                        department_id=department_id,
                    )
                ) from exc

    def get(self, personnel_id: int) -> PersonnelRecord | None:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                f"SELECT {self._personnel_columns()} FROM personnel WHERE id = ?", (personnel_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_personnel(row)

    def get_by_national_code(self, national_code: str) -> PersonnelRecord | None:
        code = normalize_national_code(national_code)
        with self._lock, self._connection() as conn:
            row = conn.execute(
                f"SELECT {self._personnel_columns()} FROM personnel WHERE national_code = ?", (code,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_personnel(row)

    def get_by_name(self, fname: str, lname: str) -> PersonnelRecord | None:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                f"SELECT {self._personnel_columns()} FROM personnel WHERE fname = ? AND lname = ?",
                (fname.strip(), lname.strip()),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_personnel(row)

    def update(
        self,
        personnel_id: int,
        fname: str | None = None,
        lname: str | None = None,
        national_code: str | None = None,
        employee_type: str | None = None,
        degree: str | None = None,
        shift_id: int | None = None,
        department_id: int | None = None,
        updated_by: int | None = None,
    ) -> PersonnelRecord | None:
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                f"SELECT {self._personnel_columns()} FROM personnel WHERE id = ?", (personnel_id,)
            ).fetchone()
            if existing is None:
                return None
            new_fname = fname.strip() if fname else existing["fname"]
            new_lname = lname.strip() if lname is not None else existing["lname"]
            if fname is not None and not new_fname:
                raise ValueError("fname cannot be blank")
            raw_code = existing["national_code"]
            if national_code is not None:
                raw_code = normalize_national_code(national_code)
                if not validate_national_code(raw_code):
                    raise ValueError(f"Invalid Iranian national code: {national_code}")
            new_employee_type = employee_type if employee_type is not None else existing["employee_type"]
            if employee_type is not None and new_employee_type not in _VALID_EMPLOYEE_TYPES:
                raise ValueError(
                    f"Invalid employee_type '{new_employee_type}'. "
                    f"Must be one of: {sorted(_VALID_EMPLOYEE_TYPES)}"
                )
            new_degree = degree if degree is not None else existing["degree"]
            new_shift_id = shift_id if shift_id is not None else existing["shift_id"]
            new_department_id = department_id if department_id is not None else existing["department_id"]
            now = self._now()
            try:
                conn.execute(
                    "UPDATE personnel SET fname=?, lname=?, national_code=?, "
                    "employee_type=?, degree=?, shift_id=?, department_id=?, "
                    "updated_at_utc=?, updated_by=? WHERE id=?",
                    (new_fname, new_lname, raw_code, new_employee_type, new_degree,
                     new_shift_id, new_department_id, now, updated_by, personnel_id),
                )
                row = conn.execute(
                    f"SELECT {self._personnel_columns()} FROM personnel WHERE id = ?", (personnel_id,)
                ).fetchone()
                return self._row_to_personnel(row)
            except IntegrityError as exc:
                raise ValueError(
                    _personnel_integrity_message(
                        exc,
                        national_code=raw_code,
                        department_id=new_department_id,
                    )
                ) from exc

    def delete(self, personnel_id: int) -> bool:
        """Delete a personnel record. Cascades images (DB + files) and sets NULL
        on human_logs and detection_room_matches. Returns True if deleted."""
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                "SELECT id FROM personnel WHERE id = ?", (personnel_id,)
            ).fetchone()
            if existing is None:
                return False
            existing = conn.execute(
                f"SELECT {self._personnel_columns()} FROM personnel WHERE id = ?", (personnel_id,)
            ).fetchone()
            if existing is None:
                return False
            image_rows = conn.execute(
                "SELECT storage_key FROM personnel_images WHERE personnel_id = ?",
                (personnel_id,),
            ).fetchall()
            for img_row in image_rows:
                self._delete_storage_file(img_row["storage_key"])
            self._delete_personnel_files(personnel_id)
            img_ids = conn.execute(
                "SELECT embedding_id FROM personnel_images WHERE personnel_id = ? AND embedding_id IS NOT NULL",
                (personnel_id,),
            ).fetchall()
            for img_id_row in img_ids:
                eid = img_id_row["embedding_id"]
                if eid:
                    try:
                        conn.execute(
                            "DELETE FROM face_embeddings WHERE id = ?", (eid,)
                        )
                    except OperationalError:
                        pass
            try:
                conn.execute(
                    "UPDATE detection_room_matches SET personnel_id = NULL WHERE personnel_id = ?",
                    (personnel_id,),
                )
            except OperationalError:
                pass
            try:
                conn.execute(
                    "UPDATE human_logs SET personnel_id = NULL WHERE personnel_id = ?",
                    (personnel_id,),
                )
            except OperationalError:
                pass
            conn.execute(
                "DELETE FROM personnel_images WHERE personnel_id = ?",
                (personnel_id,),
            )
            cursor = conn.execute(
                "DELETE FROM personnel WHERE id = ?", (personnel_id,)
            )
            return cursor.rowcount > 0

    def list(
        self,
        offset: int = 0,
        limit: int = 50,
        employee_type: str | None = None,
        search: str | None = None,
    ) -> tuple[list[PersonnelRecord], int]:
        where_clauses: list[str] = []
        params: list[Any] = []
        if employee_type is not None:
            where_clauses.append("employee_type = ?")
            params.append(employee_type)
        if search is not None:
            where_clauses.append("(fname LIKE ? OR lname LIKE ? OR national_code LIKE ?)")
            pattern = f"%{search}%"
            params.extend([pattern, pattern, pattern])
        where = ""
        if where_clauses:
            where = " WHERE " + " AND ".join(where_clauses)
        with self._lock, self._connection() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM personnel{where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT {self._personnel_columns()} FROM personnel{where} ORDER BY id DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            records = [self._row_to_personnel(r) for r in rows]
            return records, int(total)

    def list_with_images(
        self,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        with self._lock, self._connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM personnel").fetchone()[0]
            rows = conn.execute(
                f"SELECT {self._personnel_columns()} FROM personnel ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            result: list[dict[str, Any]] = []
            for row in rows:
                person = self._row_to_personnel(row)
                images = [
                    self._row_to_image(img)
                    for img in conn.execute(
                        "SELECT * FROM personnel_images WHERE personnel_id = ? ORDER BY is_primary DESC, uploaded_at_utc DESC",
                        (person.id,),
                    ).fetchall()
                ]
                result.append({
                    **dataclass_to_dict(person),
                    "images": [dataclass_to_dict(img) for img in images],
                })
            return result, int(total)

    def touch_last_seen(self, personnel_id: int, seen_at: str | None = None) -> None:
        if seen_at is None:
            seen_at = self._now()
        with self._lock, self._connection() as conn:
            conn.execute(
                "UPDATE personnel SET last_seen = ?, updated_at_utc = ? WHERE id = ?",
                (seen_at, seen_at, personnel_id),
            )

    # ── Personnel Image operations ──────────────────────────────────────

    def _row_to_image(self, row: Row) -> PersonnelImageRecord:
        embedding_id: str | None = None
        if "embedding_id" in row.keys():
            embedding_id = row["embedding_id"]
        return PersonnelImageRecord(
            id=row["id"],
            personnel_id=row["personnel_id"],
            storage_key=row["storage_key"],
            description=row["description"],
            is_primary=bool(row["is_primary"]),
            uploaded_at_utc=row["uploaded_at_utc"],
            embedding_id=embedding_id,
        )

    def create_image(
        self,
        personnel_id: int,
        storage_key: str,
        description: str | None = None,
        embedding_id: str | None = None,
        is_primary: bool | None = None,
    ) -> PersonnelImageRecord:
        now = self._now()
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                "SELECT id FROM personnel WHERE id = ?", (personnel_id,)
            ).fetchone()
            if existing is None:
                raise ValueError(f"Personnel not found: {personnel_id}")
            if is_primary is True:
                conn.execute(
                    "UPDATE personnel_images SET is_primary = 0 "
                    "WHERE personnel_id = ? AND is_primary = 1",
                    (personnel_id,),
                )
                primary_flag = 1
            elif is_primary is False:
                primary_flag = 0
            else:
                image_count = conn.execute(
                    "SELECT COUNT(*) FROM personnel_images WHERE personnel_id = ?",
                    (personnel_id,),
                ).fetchone()[0]
                primary_flag = 1 if image_count == 0 else 0
            cursor = conn.execute(
                "INSERT INTO personnel_images "
                "(personnel_id, storage_key, description, is_primary, uploaded_at_utc, embedding_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (personnel_id, storage_key, description, primary_flag, now, embedding_id),
            )
            row = conn.execute(
                "SELECT * FROM personnel_images WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            if row is None:
                raise RuntimeError("Failed to retrieve created image record")
            return self._row_to_image(row)

    def get_image(self, image_id: int) -> PersonnelImageRecord | None:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM personnel_images WHERE id = ?", (image_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_image(row)

    def update_image_embedding(self, image_id: int, embedding_id: str | None) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                "UPDATE personnel_images SET embedding_id = ? WHERE id = ?",
                (embedding_id, image_id),
            )

    def list_images(self, personnel_id: int) -> list[PersonnelImageRecord]:
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM personnel_images WHERE personnel_id = ? "
                "ORDER BY is_primary DESC, uploaded_at_utc DESC",
                (personnel_id,),
            ).fetchall()
            return [self._row_to_image(r) for r in rows]

    def delete_image(self, image_id: int) -> bool:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM personnel_images WHERE id = ?", (image_id,)
            ).fetchone()
            if row is None:
                return False
            was_primary = bool(row["is_primary"])
            personnel_id = int(row["personnel_id"])
            # Delete embedding from face_embeddings if present
            embedding_id = row["embedding_id"]
            if embedding_id:
                try:
                    conn.execute(
                        "DELETE FROM face_embeddings WHERE id = ?", (embedding_id,)
                    )
                except OperationalError:
                    pass
            # Delete the image file from disk
            self._delete_storage_file(row["storage_key"])
            conn.execute("DELETE FROM personnel_images WHERE id = ?", (image_id,))
            if was_primary:
                oldest = conn.execute(
                    "SELECT id FROM personnel_images WHERE personnel_id = ? "
                    "ORDER BY uploaded_at_utc ASC LIMIT 1",
                    (personnel_id,),
                ).fetchone()
                if oldest is not None:
                    conn.execute(
                        "UPDATE personnel_images SET is_primary = 1 WHERE id = ?",
                        (oldest["id"],),
                    )
            return True

    def set_primary_image(self, image_id: int) -> PersonnelImageRecord | None:
        with self._lock, self._connection() as conn:
            target = conn.execute(
                "SELECT * FROM personnel_images WHERE id = ?", (image_id,)
            ).fetchone()
            if target is None:
                return None
            personnel_id = int(target["personnel_id"])
            conn.execute(
                "UPDATE personnel_images SET is_primary = 0 WHERE personnel_id = ? AND is_primary = 1",
                (personnel_id,),
            )
            conn.execute(
                "UPDATE personnel_images SET is_primary = 1 WHERE id = ?",
                (image_id,),
            )
            row = conn.execute(
                "SELECT * FROM personnel_images WHERE id = ?", (image_id,)
            ).fetchone()
            return self._row_to_image(row)

    def _delete_storage_file(self, storage_key: str) -> None:
        try:
            path = self._media_root / storage_key
            if path.is_file():
                path.unlink()
        except OSError:
            pass

    def _delete_personnel_files(self, personnel_id: int) -> None:
        """Delete snapshot and cropped-face files owned by one personnel record.

        Files may be stored directly in the flat directory or inside
        {national_code}/ subdirectories — search recursively.
        """
        filename_patterns = (
            f"**/{personnel_id}_*",
            f"**/personnel_{personnel_id}_*",
        )
        for directory in (self._snapshot_dir, self._cropped_face_dir):
            try:
                candidates = {
                    path
                    for pattern in filename_patterns
                    for path in directory.glob(pattern)
                }
            except OSError:
                continue
            for path in candidates:
                try:
                    if path.is_file():
                        path.unlink()
                except OSError:
                    continue

    # ── Import / Export ─────────────────────────────────────────────────

    def generate_import_template(self) -> bytes:
        import openpyxl
        from openpyxl.styles import Alignment, Font, PatternFill, Protection, Border, Side
        wb = openpyxl.Workbook()

        # ── Fetch live data ──────────────────────────────────────────
        shifts: list[tuple[int, str]] = []
        sections: list[tuple[int, str]] = []
        try:
            with self._connection() as conn:
                shift_rows = conn.execute(
                    "SELECT id, shift_name FROM work_shifts ORDER BY id"
                ).fetchall()
                shifts = [(int(r["id"]), str(r["shift_name"])) for r in shift_rows]
                section_rows = conn.execute(
                    "SELECT id, name FROM sections ORDER BY id"
                ).fetchall()
                sections = [(int(r["id"]), str(r["name"])) for r in section_rows]
        except Exception:
            pass

        # ── Data entry sheet ─────────────────────────────────────────
        ws = wb.active
        ws.title = "ورود اطلاعات پرسنل"
        ws.sheet_view.rightToLeft = True
        headers = [
            "نام", "نام خانوادگی",
            "کد ملی", "نوع کارمند (کد)", "دپارتمان (شناسه)",
            "شیفت کاری (شناسه)", "مدرک تحصیلی",
        ]
        ws.append(headers)
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="right")
        col_widths = [16, 20, 16, 18, 18, 18, 20]
        for i, w in enumerate(col_widths, 1):
            ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w

        # ── Guidance sheet ───────────────────────────────────────────
        ws_guide = wb.create_sheet("راهنما")
        ws_guide.sheet_view.rightToLeft = True
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF", size=11)
        thin_border = Border(
            left=Side(style="thin"), right=Side(style="thin"),
            top=Side(style="thin"), bottom=Side(style="thin"),
        )
        right_align = Alignment(horizontal="right", vertical="top", wrap_text=True)
        center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

        r = 1
        ws_guide.cell(row=r, column=1, value="راهنمای واردسازی پرسنل").font = Font(bold=True, size=14)
        r += 2

        # ── Column descriptions ──────────────────────────────────────
        ws_guide.cell(row=r, column=1, value="راهنمای ستون‌ها").font = Font(bold=True, size=12)
        r += 1
        col_guide = [
            ("A: نام", "نام شخص (اجباری)"),
            ("B: نام خانوادگی", "نام خانوادگی شخص (اجباری)"),
            ("C: کد ملی", "کد ملی ۱۰ رقمی معتبر (اجباری) — صفرهای ابتدا را حتماً وارد کنید، مثال: 0012345678"),
            ("D: نوع کارمند", "1=پیمانکار, 2=مشتری, 3=مهمان, 4=کارمند, 5=نامشخص"),
            ("E: دپارتمان", "شناسه دپارتمان از جدول دپارتمان‌های زیر (اختیاری)"),
            ("F: شیفت کاری", "شناسه شیفت از جدول شیفت‌های زیر (اختیاری)"),
            ("G: مدرک تحصیلی", "1=بی‌سواد, 2=ابتدایی, 3=سیکل, 4=دیپلم, 5=فوق‌دیپلم, 6=لیسانس, 7=فوق‌لیسانس, 8=دکتری"),
        ]
        for col, desc in col_guide:
            ws_guide.cell(row=r, column=1, value=col).font = Font(bold=True)
            ws_guide.cell(row=r, column=2, value=desc).alignment = right_align
            r += 1
        r += 1

        # ── Employee type code table ──────────────────────────────────
        ws_guide.cell(row=r, column=1, value="کدهای نوع کارمند").font = Font(bold=True, size=12)
        r += 1
        et_header = ["کد", "عنوان فارسی", "عنوان انگلیسی"]
        for c, val in enumerate(et_header, 1):
            cell = ws_guide.cell(row=r, column=c, value=val)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center_align
            cell.border = thin_border
        r += 1
        for code, fa, en in [("1", "پیمانکار", "contractor"), ("2", "مشتری", "customer"),
                              ("3", "مهمان", "guest"), ("4", "کارمند", "employee"),
                              ("5", "نامشخص", "unknown")]:
            ws_guide.cell(row=r, column=1, value=code).alignment = center_align
            ws_guide.cell(row=r, column=2, value=fa).alignment = right_align
            ws_guide.cell(row=r, column=3, value=en).alignment = Alignment(horizontal="left")
            for c in range(1, 4):
                ws_guide.cell(row=r, column=c).border = thin_border
            r += 1
        r += 1

        # ── Degree code table ────────────────────────────────────────
        ws_guide.cell(row=r, column=1, value="کدهای مدرک تحصیلی").font = Font(bold=True, size=12)
        r += 1
        deg_header = ["کد", "عنوان"]
        for c, val in enumerate(deg_header, 1):
            cell = ws_guide.cell(row=r, column=c, value=val)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center_align
            cell.border = thin_border
        r += 1
        for code, label in [("1", "بی‌سواد"), ("2", "ابتدایی"), ("3", "سیکل"),
                             ("4", "دیپلم"), ("5", "فوق‌دیپلم"), ("6", "لیسانس"),
                             ("7", "فوق‌لیسانس"), ("8", "دکتری")]:
            ws_guide.cell(row=r, column=1, value=code).alignment = center_align
            ws_guide.cell(row=r, column=2, value=label).alignment = right_align
            for c in range(1, 3):
                ws_guide.cell(row=r, column=c).border = thin_border
            r += 1
        r += 1

        # ── Departments table ────────────────────────────────────────
        if sections:
            ws_guide.cell(row=r, column=1, value="دپارتمان‌های موجود").font = Font(bold=True, size=12)
            r += 1
            sec_header = ["شناسه", "نام دپارتمان"]
            for c, val in enumerate(sec_header, 1):
                cell = ws_guide.cell(row=r, column=c, value=val)
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = center_align
                cell.border = thin_border
            r += 1
            for sid, sname in sections:
                ws_guide.cell(row=r, column=1, value=sid).alignment = center_align
                ws_guide.cell(row=r, column=2, value=sname).alignment = right_align
                for c in range(1, 3):
                    ws_guide.cell(row=r, column=c).border = thin_border
                r += 1
            r += 1

        # ── Shifts table ─────────────────────────────────────────────
        if shifts:
            ws_guide.cell(row=r, column=1, value="شیفت‌های کاری موجود").font = Font(bold=True, size=12)
            r += 1
            sh_header = ["شناسه", "نام شیفت"]
            for c, val in enumerate(sh_header, 1):
                cell = ws_guide.cell(row=r, column=c, value=val)
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = center_align
                cell.border = thin_border
            r += 1
            for sid, sname in shifts:
                ws_guide.cell(row=r, column=1, value=sid).alignment = center_align
                ws_guide.cell(row=r, column=2, value=sname).alignment = right_align
                for c in range(1, 3):
                    ws_guide.cell(row=r, column=c).border = thin_border
                r += 1
            r += 1

        # ── Notes ────────────────────────────────────────────────────
        ws_guide.cell(row=r, column=1, value="نکات مهم").font = Font(bold=True, size=12)
        r += 1
        notes = [
            "ردیف اول (سرستون) در واردسازی نادیده گرفته می‌شود",
            "ردیف‌های خالی رد می‌شوند",
            "کد ملی باید ۱۰ رقمی و معتبر باشد — صفرهای ابتدایی حتماً حفظ شوند",
        ]
        for note in notes:
            ws_guide.cell(row=r, column=1, value=note).alignment = right_align
            r += 1

        ws_guide.column_dimensions["A"].width = 30
        ws_guide.column_dimensions["B"].width = 50
        ws_guide.column_dimensions["C"].width = 25
        ws_guide.protection.sheet = True
        ws_guide.protection.set_password("readonly")
        ws.protection.sheet = False
        wb.security.lockStructure = True
        wb.security.set_workbook_password("readonly")

        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        return buf.getvalue()

    def import_from_excel(
        self,
        data: bytes,
        *,
        update_existing: bool = False,
        skip_invalid_rows: bool = True,
    ) -> dict[str, Any]:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(data))
        ws = wb.active
        if ws is None:
            raise ValueError("فایل اکسل کاربرگ فعالی ندارد")

        # ── Pre-fetch valid reference IDs ──────────────────────────
        valid_shift_ids: set[int] = set()
        valid_section_ids: set[int] = set()
        try:
            with self._connection() as conn:
                valid_shift_ids = {
                    int(r["id"])
                    for r in conn.execute("SELECT id FROM work_shifts").fetchall()
                }
                valid_section_ids = {
                    int(r["id"])
                    for r in conn.execute("SELECT id FROM sections").fetchall()
                }
        except Exception:
            pass

        employee_type_map = {
            "1": "contractor", "2": "customer", "3": "guest",
            "4": "employee", "5": "unknown",
            "contractor": "contractor", "customer": "customer",
            "guest": "guest", "employee": "employee", "unknown": "unknown",
            "پیمانکار": "contractor", "مشتری": "customer", "مهمان": "guest",
            "کارمند": "employee", "نامشخص": "unknown",
        }
        degree_map = {
            "1": "illiterate", "2": "below_diploma", "3": "diploma",
            "4": "associate", "5": "bachelor", "6": "master",
            "7": "doctorate", "8": "unknown",
        }
        valid_degree_codes = set(degree_map.keys())

        rows_iter = ws.iter_rows(min_row=2, values_only=True)
        created = 0
        skipped = 0
        errors: list[dict[str, Any]] = []
        successful_rows: list[dict[str, Any]] = []
        skipped_rows: list[dict[str, Any]] = []

        for row_idx, row in enumerate(rows_iter, start=2):
            if not row or all(cell is None for cell in row):
                continue
            row_field_errors: list[dict[str, str]] = []
            try:
                # ── Parse fname ────────────────────────────────────
                fname_val = row[0] if len(row) > 0 else None
                fname = str(fname_val).strip() if fname_val is not None else ""

                # ── Parse lname ────────────────────────────────────
                lname_val = row[1] if len(row) > 1 else None
                lname = str(lname_val).strip() if lname_val is not None else ""

                # ── Parse national_code ────────────────────────────
                raw_nc_val = row[2] if len(row) > 2 else None
                if raw_nc_val is None:
                    national_code = ""
                elif isinstance(raw_nc_val, (int, float)):
                    national_code = str(int(float(raw_nc_val))).zfill(10)
                else:
                    national_code = str(raw_nc_val).strip()

                # ── Validate required fields ───────────────────────
                if not fname:
                    row_field_errors.append({"field": "fname", "message": "نام (فیلد A) اجباری است"})
                if not lname:
                    row_field_errors.append({"field": "lname", "message": "نام خانوادگی (فیلد B) اجباری است"})
                if not national_code:
                    row_field_errors.append({"field": "national_code", "message": "کد ملی (فیلد C) اجباری است"})

                # ── Parse & validate employee_type ─────────────────
                et_raw = row[3] if len(row) > 3 else None
                employee_type: str | None = None
                if et_raw is not None and str(et_raw).strip():
                    et_key = str(et_raw).strip().lower()
                    employee_type = employee_type_map.get(et_key)
                    if employee_type is None:
                        row_field_errors.append({
                            "field": "employee_type",
                            "message": (
                                f"کد نوع کارمند نامعتبر: '{et_raw}'. "
                                "کدهای مجاز: 1=پیمانکار, 2=مشتری, 3=مهمان, 4=کارمند, 5=نامشخص"
                            ),
                        })
                else:
                    employee_type = "unknown"

                # ── Parse & validate department_id ────────────────
                dept_val = row[4] if len(row) > 4 else None
                department_id: int | None = None
                if dept_val is not None and str(dept_val).strip() not in ("", "nan", "None"):
                    try:
                        dept_parsed = int(float(str(dept_val)))
                        if valid_section_ids and dept_parsed not in valid_section_ids:
                            row_field_errors.append({
                                "field": "department_id",
                                "message": f"شناسه دپارتمان '{dept_parsed}' در سیستم وجود ندارد",
                            })
                        else:
                            department_id = dept_parsed
                    except (ValueError, TypeError):
                        row_field_errors.append({
                            "field": "department_id",
                            "message": f"شناسه دپارتمان نامعتبر: '{dept_val}'",
                        })

                # ── Parse & validate shift_id ─────────────────────
                shift_val = row[5] if len(row) > 5 else None
                shift_id: int | None = None
                if shift_val is not None and str(shift_val).strip() not in ("", "nan", "None"):
                    try:
                        shift_parsed = int(float(str(shift_val)))
                        if valid_shift_ids and shift_parsed not in valid_shift_ids:
                            row_field_errors.append({
                                "field": "shift_id",
                                "message": f"شناسه شیفت کاری '{shift_parsed}' در سیستم وجود ندارد",
                            })
                        else:
                            shift_id = shift_parsed
                    except (ValueError, TypeError):
                        row_field_errors.append({
                            "field": "shift_id",
                            "message": f"شناسه شیفت کاری نامعتبر: '{shift_val}'",
                        })

                # ── Parse & validate degree ───────────────────────
                deg_raw = row[6] if len(row) > 6 else None
                degree: str | None = None
                if deg_raw is not None and str(deg_raw).strip() not in ("", "nan", "None"):
                    deg_key = str(deg_raw).strip()
                    mapped = degree_map.get(deg_key)
                    if mapped is not None:
                        degree = mapped
                    else:
                        row_field_errors.append({
                            "field": "degree",
                            "message": (
                                f"کد مدرک تحصیلی نامعتبر: '{deg_raw}'. "
                                "کدهای مجاز: 1=بی‌سواد, 2=ابتدایی, 3=سیکل, 4=دیپلم, "
                                "5=فوق‌دیپلم, 6=لیسانس, 7=فوق‌لیسانس, 8=دکتری"
                            ),
                        })

                if row_field_errors:
                    if skip_invalid_rows:
                        skipped += 1
                        skipped_rows.append({
                            "row": row_idx,
                            "national_code": national_code,
                            "field_errors": row_field_errors,
                        })
                        continue
                    else:
                        raise ValueError("; ".join(e["message"] for e in row_field_errors))

                # ── Upsert logic ──────────────────────────────────
                existing = self.get_by_national_code(national_code)
                if existing is not None:
                    if not update_existing:
                        skipped += 1
                        skipped_rows.append({
                            "row": row_idx,
                            "national_code": national_code,
                            "field_errors": [{"field": "national_code", "message": "کد ملی تکراری است"}],
                        })
                        continue
                    self.update(
                        existing.id, fname, lname, national_code,
                        employee_type, degree, shift_id, department_id,
                    )
                    action = "updated"
                else:
                    self.create(
                        fname, lname, national_code, employee_type,
                        degree, shift_id, department_id,
                    )
                    action = "created"
                created += 1
                successful_rows.append({
                    "row": row_idx,
                    "action": action,
                    "national_code": national_code,
                    "fname": fname,
                    "lname": lname,
                    "employee_type": employee_type,
                    "degree": degree,
                    "shift_id": shift_id,
                    "department_id": department_id,
                })
            except ValueError as exc:
                nc = locals().get("national_code", "")
                if skip_invalid_rows:
                    skipped += 1
                    skipped_rows.append({
                        "row": row_idx,
                        "national_code": nc,
                        "field_errors": [{"field": None, "message": str(exc)}],
                    })
                errors.append({
                    "row": row_idx,
                    "national_code": nc,
                    "field_errors": [{"field": None, "message": str(exc)}],
                })
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                nc = locals().get("national_code", "")
                if skip_invalid_rows:
                    skipped += 1
                    skipped_rows.append({
                        "row": row_idx,
                        "national_code": nc,
                        "field_errors": [{"field": None, "message": msg}],
                    })
                errors.append({
                    "row": row_idx,
                    "national_code": nc,
                    "field_errors": [{"field": None, "message": msg}],
                })
        return {
            "created": created,
            "skipped": skipped,
            "errors": errors,
            "successful_rows": successful_rows,
            "skipped_rows": skipped_rows,
        }

    def upload_personnel_zip(
        self,
        data: bytes,
        face_processor: Any | None = None,
        enable_cropping: bool = False,
    ) -> dict[str, Any]:
        """Process a ZIP file containing personnel data and images.

        Supported structures (in priority order):
          1. Folder-per-person:  {national_code}/{image_file}
             Each folder name is the person's 10-digit national code.
             fname/lname default to "Unknown".
          2. Filename pattern:  {fname}_{lname}_{national_code}.ext
             Underscore-separated parts, last part is national code.
          3. Flat numeric stem:  {national_code}.ext
             Only when the filename stem is a valid national code.
          4. metadata.xlsx or metadata.json at ZIP root (optional).

        When face_processor is provided, runs face detection and reports
        face_status per image: 0 = no face, 1 = one face (enrolled),
        2 = multiple faces. When enable_cropping is True and exactly one
        face is detected, the aligned face crop is saved to disk.

        Failed images (no face, multiple faces, enrollment errors) are
        saved to the personnel_zip_errors/ folder for manual review.
        Faces whose person name is "Unknown Unknown" are NOT enrolled
        in the vector store to avoid polluting Qdrant with unknowns.

        Returns summary dict with per-image details and error reasons.
        """
        import cv2 as cv2_mod
        import numpy as np_mod
        import zipfile
        from io import BytesIO

        created_personnel = 0
        created_images = 0
        qdrant_enrolled = 0
        image_details: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        face_counts = {"no_face": 0, "multiple_faces": 0, "enrolled": 0, "skipped_unknown_name": 0, "errors": 0}

        total_images_in_zip = 0

        with zipfile.ZipFile(BytesIO(data)) as zf:
            names = zf.namelist()

            metadata: list[dict[str, str]] = []
            if "metadata.xlsx" in names:
                excel_data = zf.read("metadata.xlsx")
                result = self.import_from_excel(excel_data)
                created_personnel = result["created"]
                for e in result.get("errors", []):
                    e["file"] = "metadata.xlsx"
                    errors.append(e)
            elif "metadata.json" in names:
                json_data = zf.read("metadata.json")
                records = json.loads(json_data)
                for rec in records:
                    try:
                        self.create(
                            fname=rec.get("fname", ""),
                            lname=rec.get("lname", ""),
                            national_code=rec.get("national_code", ""),
                            employee_type=rec.get("employee_type", "unknown"),
                            degree=rec.get("degree"),
                        )
                        created_personnel += 1
                    except (ValueError, KeyError) as exc:
                        errors.append({"file": "metadata.json", "error": str(exc)})

            for name in names:
                if name.startswith("__MACOSX") or name.startswith("."):
                    continue
                if name.endswith((".xlsx", ".json", "/")):
                    continue
                if name.startswith("metadata."):
                    continue

                total_images_in_zip += 1
                file_data = zf.read(name)
                img_filename = Path(name).name

                # Determine personnel info from folder or filename
                fc, lc, nc_candidate = _parse_zip_entry_personnel(name)
                error_reason: str | None = None
                saved_to_error_folder: str | None = None
                person_name: str = "Unknown Unknown"

                if validate_national_code(nc_candidate):
                    person = self.get_by_national_code(nc_candidate)
                    if person is None:
                        try:
                            # When the parsed name is "Unknown" (folder/bare-numeric
                            # pattern with no real name info), use a descriptive label.
                            display_fname = f"person_{nc_candidate}" if fc == "Unknown" else fc
                            display_lname = "" if fc == "Unknown" else lc
                            person = self.create(
                                fname=display_fname,
                                lname=display_lname,
                                national_code=nc_candidate,
                                employee_type="unknown",
                            )
                            created_personnel += 1
                        except ValueError as exc:
                            error_reason = str(exc)
                            errors.append({"file": name, "error": error_reason})
                            saved_to_error_folder = self._save_error_image(nc_candidate, img_filename, file_data)
                            image_details.append({
                                "file": name,
                                "status": "failed",
                                "error_reason": error_reason,
                                "saved_to_error_folder": saved_to_error_folder,
                                "face_status": -1,
                                "enrolled_in_qdrant": False,
                            })
                            face_counts["errors"] += 1
                            continue
                else:
                    error_reason = "Cannot determine personnel from filename"
                    errors.append({"file": name, "error": error_reason})
                    saved_to_error_folder = self._save_error_image("unknown", img_filename, file_data)
                    image_details.append({
                        "file": name,
                        "status": "failed",
                        "error_reason": error_reason,
                        "saved_to_error_folder": saved_to_error_folder,
                        "face_status": -1,
                        "enrolled_in_qdrant": False,
                    })
                    face_counts["errors"] += 1
                    continue

                if person is not None:
                    raw_name = f"{person.fname} {person.lname}".strip() or "Unknown Unknown"
                    # If the person has an unknown name (e.g. from a previous
                    # upload that could not determine the real name), update it
                    # to a descriptive label so vector enrollment can proceed.
                    if raw_name.lower() in ("unknown unknown", "unknown"):
                        updated = self.update(
                            person.id,
                            fname=f"person_{nc_candidate}",
                            lname="",
                        )
                        if updated is not None:
                            person = updated
                    person_name = f"{person.fname} {person.lname}".strip() 
                    try:
                        storage_key = self._save_image_file(person.id, file_data, img_filename, nc_candidate)
                        image_record = self.create_image(
                            person.id,
                            storage_key,
                            embedding_id=None,
                        )

                        # Face processing
                        face_status = -1
                        cropped_face_key: str | None = None
                        embedding_id: str | None = None
                        enrolled_in_qdrant = False

                        if face_processor is not None:
                            np_arr = np_mod.frombuffer(file_data, dtype=np_mod.uint8)
                            image = cv2_mod.imdecode(np_arr, cv2_mod.IMREAD_COLOR)
                            if image is not None:
                                try:
                                    raw_count = face_processor.count_faces(image)
                                except Exception:
                                    raw_count = 0

                                if raw_count == 0:
                                    face_status = 0
                                    error_reason = "No face detected in image"
                                    face_counts["no_face"] += 1
                                elif raw_count >= 2:
                                    face_status = 2
                                    error_reason = f"Multiple faces detected ({raw_count})"
                                    face_counts["multiple_faces"] += 1
                                else:
                                    # Exactly one face detected — attempt enrollment
                                    try:
                                        # Skip enrollment when person name is unknown
                                        if person_name.lower() in ("unknown unknown", "unknown"):
                                            face_status = 1
                                            error_reason = "Skipped vector enrollment: person name is unknown"
                                            face_counts["skipped_unknown_name"] += 1
                                            LOGGER.info(
                                                "Skipped Qdrant enrollment for '%s' (%s): name is unknown",
                                                person_name, nc_candidate,
                                            )
                                        else:
                                            enroll_result = face_processor.enroll(
                                                image,
                                                person=person.national_code,
                                                ref_img_id=str(image_record.id),
                                            )
                                            embedding_id = enroll_result.get("point_id")
                                            face_status = 1
                                            enrolled_in_qdrant = True
                                            qdrant_enrolled += 1
                                            face_counts["enrolled"] += 1

                                            if enable_cropping:
                                                success, aligned = face_processor.get_aligned_face(image)
                                                if success and aligned is not None:
                                                    ok_enc, encoded = cv2_mod.imencode(".jpg", aligned)
                                                    if ok_enc:
                                                        cropped_face_key = self._save_cropped_face_file(
                                                            person.id, encoded.tobytes(), img_filename, nc_candidate
                                                        )

                                            LOGGER.info(
                                                "Enrolled face for '%s' (%s) in Qdrant: point_id=%s",
                                                person_name, nc_candidate, embedding_id,
                                            )
                                    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
                                        face_status = 0
                                        error_reason = f"Face enrollment failed: {exc}"
                                        face_counts["errors"] += 1
                            else:
                                error_reason = "Could not decode image"
                                face_counts["errors"] += 1
                        else:
                            # No face processor — image saved without face check
                            face_status = -1

                        self.update_image_embedding(image_record.id, embedding_id)
                        created_images += 1

                        # Save to error folder if there was a problem
                        if error_reason and embedding_id is None:
                            saved_to_error_folder = self._save_error_image(nc_candidate, img_filename, file_data)

                        image_details.append({
                            "file": name,
                            "status": "failed" if error_reason else "ok",
                            "error_reason": error_reason,
                            "saved_to_error_folder": saved_to_error_folder,
                            "face_status": face_status,
                            "enrolled_in_qdrant": enrolled_in_qdrant,
                            "cropped_face_key": cropped_face_key,
                            "embedding_id": embedding_id,
                        })
                        if error_reason:
                            errors.append({"file": name, "error": error_reason})
                    except Exception as exc:
                        error_reason = f"{type(exc).__name__}: {exc}"
                        errors.append({"file": name, "error": error_reason})
                        saved_to_error_folder = self._save_error_image(
                            nc_candidate, img_filename, file_data
                        )
                        image_details.append({
                            "file": name,
                            "status": "failed",
                            "error_reason": error_reason,
                            "saved_to_error_folder": saved_to_error_folder,
                            "face_status": -1,
                            "enrolled_in_qdrant": False,
                        })
                        face_counts["errors"] += 1

        total_failed = sum(1 for d in image_details if d["status"] == "failed")

        LOGGER.info(
            "ZIP upload complete: %d images in zip | %d persons | %d images saved | "
            "%d enrolled in Qdrant | %d failed | "
            "face_stats: no_face=%d multiple_faces=%d enrolled=%d skipped_unknown_name=%d errors=%d",
            total_images_in_zip,
            created_personnel,
            created_images,
            qdrant_enrolled,
            total_failed,
            face_counts["no_face"],
            face_counts["multiple_faces"],
            face_counts["enrolled"],
            face_counts["skipped_unknown_name"],
            face_counts["errors"],
        )

        return {
            "created_personnel": created_personnel,
            "created_images": created_images,
            "qdrant_enrolled": qdrant_enrolled,
            "image_details": image_details,
            "errors": errors,
            "face_counts": face_counts,
            "total_images_in_zip": total_images_in_zip,
            "total_failed": total_failed,
        }

    def _save_image_file(self, personnel_id: int, data: bytes, original_filename: str, national_code: str = "") -> str:
        """Save an image to personnel_snapshots/{national_code}/ and return the relative storage_key.

        When national_code is empty the file is saved directly under personnel_snapshots/
        for backward compatibility.
        """
        ext = Path(original_filename).suffix if "." in original_filename else ".jpg"
        unique_name = f"{personnel_id}_{uuid.uuid4().hex}{ext}"
        if national_code:
            relative_path = f"personnel_snapshots/{national_code}/{unique_name}"
        else:
            relative_path = f"personnel_snapshots/{unique_name}"
        dest = self._media_root / relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return relative_path

    def _save_cropped_face_file(self, personnel_id: int, data: bytes, original_filename: str, national_code: str = "") -> str:
        """Save a cropped face image to personnel_cropped_faces/{national_code}/ and return the storage_key.

        When national_code is empty the file is saved directly under personnel_cropped_faces/
        for backward compatibility.
        """
        ext = Path(original_filename).suffix if "." in original_filename else ".jpg"
        unique_name = f"{personnel_id}_{uuid.uuid4().hex}{ext}"
        if national_code:
            relative_path = f"personnel_cropped_faces/{national_code}/{unique_name}"
        else:
            relative_path = f"personnel_cropped_faces/{unique_name}"
        dest = self._media_root / relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return relative_path

    def _save_error_image(self, national_code: str, filename: str, data: bytes) -> str:
        """Save a copy of a failed image to the zip errors directory for manual review.

        Returns the relative path (for logging/reporting).
        """
        safe_nc = re.sub(r"[^\dA-Za-z_-]", "_", national_code) if national_code else "unknown"
        dest_dir = self._zip_errors_dir / safe_nc
        dest_dir.mkdir(parents=True, exist_ok=True)
        unique_name = f"{Path(filename).stem}_{uuid.uuid4().hex}{Path(filename).suffix or '.jpg'}"
        dest = dest_dir / unique_name
        dest.write_bytes(data)
        relative = str(dest.relative_to(self._media_root))
        LOGGER.warning("Saved failed image to error folder: %s", relative)
        return relative

    def get_image_path(self, storage_key: str) -> Path:
        return self._media_root / storage_key

    def read_image_base64(self, storage_key: str) -> str | None:
        """Read the image file at the given storage_key and return it as a
        base64-encoded data URI string. Returns None if the file is missing."""
        try:
            path = self._media_root / storage_key
            if not path.is_file():
                return None
            data = path.read_bytes()
            ext = path.suffix.lower()
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png", "bmp": "image/bmp"}.get(ext.lstrip("."), "image/jpeg")
            encoded = base64.b64encode(data).decode("ascii")
            return f"data:{mime};base64,{encoded}"
        except (OSError, FileNotFoundError):
            return None

    def count(self) -> int:
        with self._lock, self._connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM personnel").fetchone()[0])


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    return {
        f.name: getattr(obj, f.name)
        for f in type(obj).__dataclass_fields__.values()
    }
