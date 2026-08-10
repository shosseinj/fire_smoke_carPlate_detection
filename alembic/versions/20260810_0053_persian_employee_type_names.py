"""Normalize existing default employee-type names to Persian.

Revision ID: 20260810_0053
Revises: 20260810_0052
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260810_0053"
down_revision = "20260810_0052"
branch_labels = None
depends_on = None


_NAME_MAP = (
    ("contractor", "پیمانکار", False),
    ("customer", "مشتری", False),
    ("guest", "مهمان", False),
    ("employee", "کارمند", True),
    ("unknown", "نامشخص", False),
)


def upgrade() -> None:
    bind = op.get_bind()
    for english_name, persian_name, report_flag in _NAME_MAP:
        source = bind.execute(
            sa.text("SELECT id, include_in_attendance_reports FROM employee_types WHERE LOWER(name) = :name"),
            {"name": english_name},
        ).mappings().first()
        target = bind.execute(
            sa.text("SELECT id, include_in_attendance_reports FROM employee_types WHERE name = :name"),
            {"name": persian_name},
        ).mappings().first()

        if source is not None and target is not None and source["id"] != target["id"]:
            # Merge safely if an administrator already created the Persian row.
            bind.execute(
                sa.text("UPDATE personnel SET employee_type_id = :target_id WHERE employee_type_id = :source_id"),
                {"target_id": target["id"], "source_id": source["id"]},
            )
            if bool(source["include_in_attendance_reports"]):
                bind.execute(
                    sa.text("UPDATE employee_types SET include_in_attendance_reports = TRUE WHERE id = :target_id"),
                    {"target_id": target["id"]},
                )
            bind.execute(
                sa.text("DELETE FROM employee_types WHERE id = :source_id"),
                {"source_id": source["id"]},
            )
        elif source is not None:
            bind.execute(
                sa.text("UPDATE employee_types SET name = :persian_name WHERE id = :source_id"),
                {"persian_name": persian_name, "source_id": source["id"]},
            )

        if report_flag:
            bind.execute(
                sa.text("UPDATE employee_types SET include_in_attendance_reports = TRUE WHERE name = :name"),
                {"name": persian_name},
            )


def downgrade() -> None:
    bind = op.get_bind()
    for english_name, persian_name, _ in _NAME_MAP:
        # Downgrade only renames unambiguous Persian defaults back to legacy names.
        bind.execute(
            sa.text("UPDATE employee_types SET name = :english_name WHERE name = :persian_name"),
            {"english_name": english_name, "persian_name": persian_name},
        )
