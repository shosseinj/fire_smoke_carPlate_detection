from __future__ import annotations

from app.database import Connection, Database, IntegrityError, OperationalError, Row, ensure_database
from app.time_utils import utc_now_text

import io
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings


_VALID_EMPLOYEE_TYPES = frozenset({"contractor", "customer", "guest", "employee", "unknown"})

# Persian, Arabic, and English digit mappings for national code normalization
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
    last_seen: str | None
    created_at_utc: str
    updated_at_utc: str


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
    """Normalize Persian/Arabic/English digits and strip whitespace."""
    normalized = raw.strip().translate(_PERSIAN_DIGITS)
    # Keep only ASCII digits
    normalized = re.sub(r"[^\d]", "", normalized)
    return normalized


def validate_national_code(code: str) -> bool:
    """Validate an Iranian national code (کد ملی) using the checksum algorithm."""
    if not re.match(r"^\d{10}$", code):
        return False
    # All identical digits are invalid
    if len(set(code)) == 1:
        return False
    digits = [int(d) for d in code]
    checksum = digits[-1]
    # Multiply each of the first 9 digits by its position weight (10 down to 2)
    total = sum(digits[i] * (10 - i) for i in range(9))
    remainder = total % 11
    if remainder < 2:
        expected = remainder
    else:
        expected = 11 - remainder
    return expected == checksum


class PersonnelStore:
    """PostgreSQL-backed store for personnel records and images."""

    def __init__(self, database: Database | str, saved_media_path: Path) -> None:
        self.database = ensure_database(database)
        self._media_root = saved_media_path.resolve()
        self._snapshot_dir = self._media_root / "personnel_snapshots"
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._cropped_face_dir = self._media_root / "personnel_cropped_faces"
        self._cropped_face_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connection(self) -> Connection:
        return self.database.connection()

    def _init_db(self) -> None:
        # Alembic owns the PostgreSQL schema; runtime startup validates it.
        return None

    # ── Personnel CRUD ─────────────────────────────────────────────────

    def _row_to_personnel(self, row: Row) -> PersonnelRecord:
        last_seen: str | None = None
        if "last_seen" in row.keys():
            last_seen = row["last_seen"]
        shift_id: int | None = None
        if "shift_id" in row.keys():
            raw = row["shift_id"]
            shift_id = int(raw) if raw is not None else None
        return PersonnelRecord(
            id=row["id"],
            fname=row["fname"],
            lname=row["lname"],
            national_code=row["national_code"],
            employee_type=row["employee_type"],
            degree=row["degree"],
            shift_id=shift_id,
            last_seen=last_seen,
            created_at_utc=row["created_at_utc"],
            updated_at_utc=row["updated_at_utc"],
        )

    def _now(self) -> str:
        return utc_now_text()

    def create(
        self,
        fname: str,
        lname: str,
        national_code: str,
        employee_type: str = "unknown",
        degree: str | None = None,
    ) -> PersonnelRecord:
        """Insert a new personnel record. Raises ValueError on invalid data."""
        fname = fname.strip()
        lname = lname.strip()
        if not fname or not lname:
            raise ValueError("fname and lname are required")
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
                    "(fname, lname, national_code, employee_type, degree, created_at_utc, updated_at_utc) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (fname, lname, raw_code, employee_type, degree, now, now),
                )
                row = conn.execute(
                    "SELECT * FROM personnel WHERE id = ?", (cursor.lastrowid,)
                ).fetchone()
                if row is None:
                    raise RuntimeError("Failed to retrieve created personnel record")
                return self._row_to_personnel(row)
            except IntegrityError:
                raise ValueError(f"National code already exists: {raw_code}")

    def get(self, personnel_id: int) -> PersonnelRecord | None:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM personnel WHERE id = ?", (personnel_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_personnel(row)

    def get_by_national_code(self, national_code: str) -> PersonnelRecord | None:
        code = normalize_national_code(national_code)
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM personnel WHERE national_code = ?", (code,)
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
    ) -> PersonnelRecord | None:
        """Update personnel fields. Returns updated record or None if not found."""
        with self._lock, self._connection() as conn:
            existing = conn.execute(
                "SELECT * FROM personnel WHERE id = ?", (personnel_id,)
            ).fetchone()
            if existing is None:
                return None
            new_fname = fname.strip() if fname else existing["fname"]
            new_lname = lname.strip() if lname else existing["lname"]
            if fname is not None and not new_fname:
                raise ValueError("fname cannot be blank")
            if lname is not None and not new_lname:
                raise ValueError("lname cannot be blank")
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
            now = self._now()
            try:
                conn.execute(
                    "UPDATE personnel SET fname=?, lname=?, national_code=?, "
                    "employee_type=?, degree=?, updated_at_utc=? WHERE id=?",
                    (new_fname, new_lname, raw_code, new_employee_type, new_degree, now, personnel_id),
                )
                row = conn.execute(
                    "SELECT * FROM personnel WHERE id = ?", (personnel_id,)
                ).fetchone()
                return self._row_to_personnel(row)
            except IntegrityError:
                raise ValueError(f"National code already exists: {raw_code}")

    def delete(self, personnel_id: int) -> bool:
        """Delete a personnel record and cascade images. Returns True if deleted."""
        with self._lock, self._connection() as conn:
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
        """Return (records, total_count) with optional filters."""
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
                f"SELECT * FROM personnel{where} ORDER BY id DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            records = [self._row_to_personnel(r) for r in rows]
            return records, int(total)

    def list_with_images(
        self,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return personnel records with their images as nested list."""
        with self._lock, self._connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM personnel").fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM personnel ORDER BY id DESC LIMIT ? OFFSET ?",
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
        """Update the last_seen timestamp for a personnel record."""
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
    ) -> PersonnelImageRecord:
        """Insert an image record. The first image for a personnel becomes primary."""
        now = self._now()
        with self._lock, self._connection() as conn:
            # Verify personnel exists
            existing = conn.execute(
                "SELECT id FROM personnel WHERE id = ?", (personnel_id,)
            ).fetchone()
            if existing is None:
                raise ValueError(f"Personnel not found: {personnel_id}")
            # First image auto-becomes primary
            image_count = conn.execute(
                "SELECT COUNT(*) FROM personnel_images WHERE personnel_id = ?",
                (personnel_id,),
            ).fetchone()[0]
            is_primary = 1 if image_count == 0 else 0
            cursor = conn.execute(
                "INSERT INTO personnel_images "
                "(personnel_id, storage_key, description, is_primary, uploaded_at_utc, embedding_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (personnel_id, storage_key, description, is_primary, now, embedding_id),
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

    def list_images(self, personnel_id: int) -> list[PersonnelImageRecord]:
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM personnel_images WHERE personnel_id = ? "
                "ORDER BY is_primary DESC, uploaded_at_utc DESC",
                (personnel_id,),
            ).fetchall()
            return [self._row_to_image(r) for r in rows]

    def delete_image(self, image_id: int) -> bool:
        """Delete an image record. If it was primary, promote the oldest remaining."""
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM personnel_images WHERE id = ?", (image_id,)
            ).fetchone()
            if row is None:
                return False
            was_primary = bool(row["is_primary"])
            personnel_id = int(row["personnel_id"])
            # Delete the image file from disk
            self._delete_storage_file(row["storage_key"])
            conn.execute("DELETE FROM personnel_images WHERE id = ?", (image_id,))
            # If deleted was primary, promote the oldest remaining image
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
        """Atomically set one image as primary, unsetting any other primary for the same personnel."""
        with self._lock, self._connection() as conn:
            target = conn.execute(
                "SELECT * FROM personnel_images WHERE id = ?", (image_id,)
            ).fetchone()
            if target is None:
                return None
            personnel_id = int(target["personnel_id"])
            # Unset current primary for this personnel
            conn.execute(
                "UPDATE personnel_images SET is_primary = 0 WHERE personnel_id = ? AND is_primary = 1",
                (personnel_id,),
            )
            # Set new primary
            conn.execute(
                "UPDATE personnel_images SET is_primary = 1 WHERE id = ?",
                (image_id,),
            )
            row = conn.execute(
                "SELECT * FROM personnel_images WHERE id = ?", (image_id,)
            ).fetchone()
            return self._row_to_image(row)

    def _delete_storage_file(self, storage_key: str) -> None:
        """Delete the physical file for a storage key. Failures are non-fatal."""
        try:
            path = self._media_root / storage_key
            if path.is_file():
                path.unlink()
        except OSError:
            pass

    # ── Import / Export ─────────────────────────────────────────────────

    def generate_import_template(self) -> bytes:
        """Generate an Excel template with the expected columns."""
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Personnel Import"
        headers = ["fname", "lname", "national_code", "employee_type", "degree"]
        ws.append(headers)
        ws.append(["Example", "User", "0012345678", "employee", "Bachelor"])
        # Style headers
        for cell in ws[1]:
            cell.font = openpyxl.styles.Font(bold=True)
        # Set column widths
        for i, header in enumerate(headers, 1):
            ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = 20
        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        return buf.getvalue()

    def import_from_excel(self, data: bytes) -> dict[str, Any]:
        """Parse an Excel file and import personnel records.

        Returns summary dict with created, skipped, errors counts.
        """
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(data))
        ws = wb.active
        if ws is None:
            raise ValueError("Excel file has no active worksheet")
        rows_iter = ws.iter_rows(min_row=2, values_only=True)
        created = 0
        skipped = 0
        errors: list[dict[str, Any]] = []
        for row_idx, row in enumerate(rows_iter, start=2):
            if not row or all(cell is None for cell in row):
                continue
            try:
                fname = str(row[0]).strip() if row[0] is not None else ""
                lname = str(row[1]).strip() if row[1] is not None else ""
                national_code = str(row[2]).strip() if row[2] is not None else ""
                employee_type = str(row[3]).strip().lower() if row[3] is not None else "unknown"
                degree = str(row[4]).strip() if len(row) > 4 and row[4] is not None else None
                if not fname or not lname or not national_code:
                    skipped += 1
                    errors.append({"row": row_idx, "error": "Missing required fields (fname, lname, national_code)"})
                    continue
                self.create(
                    fname=fname,
                    lname=lname,
                    national_code=national_code,
                    employee_type=employee_type,
                    degree=degree,
                )
                created += 1
            except ValueError as exc:
                skipped += 1
                errors.append({"row": row_idx, "error": str(exc)})
            except Exception as exc:
                skipped += 1
                errors.append({"row": row_idx, "error": f"{type(exc).__name__}: {exc}"})
        return {"created": created, "skipped": skipped, "errors": errors}

    def upload_personnel_zip(
        self,
        data: bytes,
        face_processor: Any | None = None,
    ) -> dict[str, Any]:
        """Process a ZIP file containing personnel metadata + images.

        Expected structure:
          - metadata.xlsx or metadata.json (optional — personnel records)
          - images/ directory with image files
          - Images are matched to personnel via metadata or filename pattern

        When face_processor is provided, runs face detection and saves
        cropped faces. Each image entry includes face_status:
          0 = no face, 1 = one face (enrolled), 2 = multiple faces.

        Returns summary dict.
        """
        import cv2 as cv2_mod
        import numpy as np_mod
        import zipfile
        from io import BytesIO

        created_personnel = 0
        created_images = 0
        image_results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        with zipfile.ZipFile(BytesIO(data)) as zf:
            # Check for metadata files
            metadata: list[dict[str, str]] = []
            if "metadata.xlsx" in zf.namelist():
                excel_data = zf.read("metadata.xlsx")
                result = self.import_from_excel(excel_data)
                created_personnel = result["created"]
                errors.extend(result["errors"])
            elif "metadata.json" in zf.namelist():
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
            else:
                # No metadata — create personnel from image filenames
                # Files named like: fname_lname_nationalcode.ext
                pass  # handled by file loop below

            # Process image files
            for name in zf.namelist():
                if name.startswith("__MACOSX") or name.startswith("."):
                    continue
                if name.endswith((".xlsx", ".json", "/")):
                    continue
                # Extract personnel info from filename or use auto-generated entries
                file_data = zf.read(name)
                # Try to extract personnel info from path
                parts = Path(name).parts
                # Support: images/fname_lname_nationalcode.ext or images/nationalcode.ext
                img_filename = Path(name).name
                stem = img_filename.rsplit(".", 1)[0] if "." in img_filename else img_filename
                parts_underscore = stem.split("_")
                if len(parts_underscore) >= 3:
                    # Assume fname_lname_nationalcode format
                    fc = parts_underscore[0]
                    lc = "_".join(parts_underscore[1:-1])
                    nc_candidate = normalize_national_code(parts_underscore[-1])
                else:
                    # Use stem directly as identifier
                    fc = "Unknown"
                    lc = "Unknown"
                    nc_candidate = normalize_national_code(stem) if stem.isdigit() else f"auto-{uuid.uuid4().hex[:8]}"

                # Find or create personnel
                if validate_national_code(nc_candidate):
                    person = self.get_by_national_code(nc_candidate)
                    if person is None:
                        try:
                            person = self.create(
                                fname=fc,
                                lname=lc,
                                national_code=nc_candidate,
                                employee_type="unknown",
                            )
                            created_personnel += 1
                        except ValueError as exc:
                            errors.append({"file": name, "error": str(exc)})
                            continue
                else:
                    # Auto-generate placeholder if no valid national code found
                    try:
                        # Use a placeholder — only if we can't find by other means
                        # Skip if we can't identify personnel
                        errors.append({"file": name, "error": "Cannot determine personnel from filename"})
                        continue
                    except ValueError as exc:
                        errors.append({"file": name, "error": str(exc)})
                        continue

                # Save image
                if person is not None:
                    try:
                        storage_key = self._save_image_file(person.id, file_data, img_filename)

                        # Face processing
                        face_status = 0
                        cropped_face_key: str | None = None
                        embedding_id: str | None = None

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
                                elif raw_count >= 2:
                                    face_status = 2
                                else:
                                    try:
                                        enroll_result = face_processor.enroll(
                                            image,
                                            person=f"{person.fname} {person.lname}",
                                            ref_img_id=f"personnel_{person.id}",
                                        )
                                        embedding_id = enroll_result.get("point_id")
                                        face_status = 1

                                        success, aligned = face_processor.get_aligned_face(image)
                                        if success and aligned is not None:
                                            ok_enc, encoded = cv2_mod.imencode(".jpg", aligned)
                                            if ok_enc:
                                                cropped_face_key = self._save_cropped_face_file(
                                                    person.id, encoded.tobytes(), img_filename
                                                )
                                    except (ValueError, FileNotFoundError, RuntimeError, ImportError):
                                        face_status = 0

                        self.create_image(
                            person.id, storage_key,
                            embedding_id=embedding_id,
                        )
                        created_images += 1
                        image_results.append({
                            "file": name,
                            "face_status": face_status,
                            "cropped_face_key": cropped_face_key,
                        })
                    except Exception as exc:
                        errors.append({"file": name, "error": f"{type(exc).__name__}: {exc}"})

        return {
            "created_personnel": created_personnel,
            "created_images": created_images,
            "image_results": image_results,
            "errors": errors,
        }

    def _save_image_file(self, personnel_id: int, data: bytes, original_filename: str) -> str:
        """Save an image file to disk and return the storage_key (relative to media_root)."""
        ext = Path(original_filename).suffix if "." in original_filename else ".jpg"
        unique_name = f"personnel_{personnel_id}_{uuid.uuid4().hex}{ext}"
        relative_path = f"personnel_snapshots/{unique_name}"
        dest = self._media_root / relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return relative_path

    def _save_cropped_face_file(self, personnel_id: int, data: bytes, original_filename: str) -> str:
        """Save a cropped face image to disk and return the storage_key (relative to media_root)."""
        ext = Path(original_filename).suffix if "." in original_filename else ".jpg"
        unique_name = f"personnel_{personnel_id}_{uuid.uuid4().hex}{ext}"
        relative_path = f"personnel_cropped_faces/{unique_name}"
        dest = self._media_root / relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return relative_path

    def get_image_path(self, storage_key: str) -> Path:
        """Return the absolute path for a storage_key."""
        return self._media_root / storage_key

    def count(self) -> int:
        with self._lock, self._connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM personnel").fetchone()[0])


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass instance to a dict, handling frozen=True."""
    # This avoids asdict() which has issues with frozen dataclasses with slots
    return {
        f.name: getattr(obj, f.name)
        for f in type(obj).__dataclass_fields__.values()  # type: ignore[attr-defined]
    }
