"""Add attendance-report eligibility to employee types.

Revision ID: 20260810_0052
Revises: 20260810_0051
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260810_0052"
down_revision = "20260810_0051"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "employee_types",
        sa.Column(
            "include_in_attendance_reports",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
    )
    # Preserve the intended legacy business rule without coupling future report
    # selection to a hardcoded ID/name: only the existing employee type starts
    # eligible, and administrators can change eligibility through CRUD later.
    op.execute(
        "UPDATE employee_types "
        "SET include_in_attendance_reports = TRUE "
        "WHERE LOWER(name) = 'employee' OR name = 'کارمند'"
    )


def downgrade() -> None:
    op.drop_column("employee_types", "include_in_attendance_reports")
