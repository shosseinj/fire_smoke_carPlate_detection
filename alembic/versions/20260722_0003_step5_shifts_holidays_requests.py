"""Step 5 schema: shift FK, holiday unique index, request fields.

- Add FK from personnel.shift_id to work_shifts.id ON DELETE SET NULL
- Add partial unique index on holidays (date_value, every_year) WHERE is_active
- Add columns to personnel_requests: duration_type, start_time, end_time,
  duration_days, duration_minutes, admin_notes, reviewed_by, reviewed_at

Revision ID: 20260722_0003
Revises: 20260722_0002
Create Date: 2026-07-22
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260722_0003"
down_revision: Union[str, Sequence[str], None] = "20260722_0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Shift FK ──────────────────────────────────────────────────────
    op.execute(
        sa.text(
            "DO $$ "
            "BEGIN "
            "  IF NOT EXISTS ( "
            "    SELECT 1 FROM pg_constraint "
            "    WHERE conname = 'fk_personnel_shift_id' "
            "  ) THEN "
            "    ALTER TABLE personnel "
            "    ADD CONSTRAINT fk_personnel_shift_id "
            "    FOREIGN KEY (shift_id) REFERENCES work_shifts(id) "
            "    ON DELETE SET NULL; "
            "  END IF; "
            "END $$;"
        )
    )

    # ── Holiday partial unique index ───────────────────────────────────
    op.execute(
        sa.text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_holidays_active_date_every_year "
            "ON holidays (date_value, every_year) "
            "WHERE is_active = 1"
        )
    )

    # ── Personnel request new columns ──────────────────────────────────
    op.execute(
        sa.text(
            "ALTER TABLE personnel_requests "
            "ADD COLUMN IF NOT EXISTS duration_type VARCHAR(16)"
        )
    )
    op.execute(
        sa.text(
            "ALTER TABLE personnel_requests "
            "ADD COLUMN IF NOT EXISTS start_time TIME WITHOUT TIME ZONE"
        )
    )
    op.execute(
        sa.text(
            "ALTER TABLE personnel_requests "
            "ADD COLUMN IF NOT EXISTS end_time TIME WITHOUT TIME ZONE"
        )
    )
    op.execute(
        sa.text(
            "ALTER TABLE personnel_requests "
            "ADD COLUMN IF NOT EXISTS duration_days FLOAT"
        )
    )
    op.execute(
        sa.text(
            "ALTER TABLE personnel_requests "
            "ADD COLUMN IF NOT EXISTS duration_minutes INTEGER"
        )
    )
    op.execute(
        sa.text(
            "ALTER TABLE personnel_requests "
            "ADD COLUMN IF NOT EXISTS admin_notes TEXT"
        )
    )
    op.execute(
        sa.text(
            "ALTER TABLE personnel_requests "
            "ADD COLUMN IF NOT EXISTS reviewed_by INTEGER "
            "REFERENCES users(id) ON DELETE SET NULL"
        )
    )
    op.execute(
        sa.text(
            "ALTER TABLE personnel_requests "
            "ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ"
        )
    )

    # Add indexes for new columns
    op.create_index(
        "idx_requests_duration_type",
        "personnel_requests", ["duration_type"],
    )
    op.create_index(
        "idx_requests_reviewed_by",
        "personnel_requests", ["reviewed_by"],
    )


def downgrade() -> None:
    # Remove new indexes
    op.drop_index("idx_requests_duration_type", table_name="personnel_requests")
    op.drop_index("idx_requests_reviewed_by", table_name="personnel_requests")

    # Remove new columns
    op.execute(sa.text("ALTER TABLE personnel_requests DROP COLUMN reviewed_at"))
    op.execute(sa.text("ALTER TABLE personnel_requests DROP COLUMN reviewed_by"))
    op.execute(sa.text("ALTER TABLE personnel_requests DROP COLUMN admin_notes"))
    op.execute(sa.text("ALTER TABLE personnel_requests DROP COLUMN duration_minutes"))
    op.execute(sa.text("ALTER TABLE personnel_requests DROP COLUMN duration_days"))
    op.execute(sa.text("ALTER TABLE personnel_requests DROP COLUMN end_time"))
    op.execute(sa.text("ALTER TABLE personnel_requests DROP COLUMN start_time"))
    op.execute(sa.text("ALTER TABLE personnel_requests DROP COLUMN duration_type"))

    # Drop holiday unique index
    op.execute(
        sa.text("DROP INDEX IF EXISTS uq_holidays_active_date_every_year")
    )

    # Drop shift FK
    op.execute(
        sa.text(
            "ALTER TABLE personnel DROP CONSTRAINT IF EXISTS fk_personnel_shift_id"
        )
    )
