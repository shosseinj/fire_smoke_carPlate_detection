"""Add dated personnel shift assignments.

Revision ID: 20260802_0047
Revises: 20260730_0046
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260802_0047"
down_revision = "20260730_0046"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS btree_gist"))
    op.create_table(
        "personnel_shift_assignments",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("personnel_id", sa.Integer(), nullable=False),
        sa.Column("shift_id", sa.Integer(), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=False),
        sa.Column("created_at_utc", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at_utc", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.CheckConstraint("end_date >= start_date", name="ck_personnel_shift_assignment_dates"),
        sa.ForeignKeyConstraint(["personnel_id"], ["personnel.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["shift_id"], ["work_shifts.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_personnel_shift_assignments_personnel", "personnel_shift_assignments", ["personnel_id"])
    op.create_index("idx_personnel_shift_assignments_shift", "personnel_shift_assignments", ["shift_id"])
    op.create_index("idx_personnel_shift_assignments_dates", "personnel_shift_assignments", ["start_date", "end_date"])
    op.execute(sa.text(
        "ALTER TABLE personnel_shift_assignments ADD CONSTRAINT "
        "ex_personnel_shift_assignments_no_overlap EXCLUDE USING gist "
        "(personnel_id WITH =, daterange(start_date, end_date, '[]') WITH &&)"
    ))
    op.execute(sa.text(
        "INSERT INTO personnel_shift_assignments "
        "(personnel_id, shift_id, start_date, end_date) "
        "SELECT id, shift_id, DATE '2026-03-21', DATE '2027-03-20' "
        "FROM personnel WHERE shift_id IS NOT NULL"
    ))


def downgrade() -> None:
    op.drop_table("personnel_shift_assignments")
