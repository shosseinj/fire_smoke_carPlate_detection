"""Reconcile official-holiday columns for databases stamped at revision 0036.

Revision ID: 20260728_0037
Revises: 20260728_0036
"""

from __future__ import annotations

from alembic import op


revision = "20260728_0037"
down_revision = "20260728_0036"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Some live databases were stamped at 0036 before that revision acquired
    # the official-holiday DDL.  Keep this repair idempotent so it is harmless
    # when 0036 was applied with its current contents.
    op.execute(
        "ALTER TABLE holidays "
        "ADD COLUMN IF NOT EXISTS is_official INTEGER NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE holidays "
        "ADD COLUMN IF NOT EXISTS official_jalali_year INTEGER NULL"
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'ck_holidays_official_year'
                  AND conrelid = 'holidays'::regclass
            ) THEN
                ALTER TABLE holidays
                ADD CONSTRAINT ck_holidays_official_year CHECK (
                    (is_official = 0 AND official_jalali_year IS NULL) OR
                    (is_official = 1 AND official_jalali_year BETWEEN 1200 AND 1600)
                );
            END IF;
        END
        $$
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_holidays_official_year "
        "ON holidays (is_official, official_jalali_year)"
    )
    op.execute("DROP INDEX IF EXISTS uq_holidays_active_date_every_year")
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_holidays_active_identity "
        "ON holidays (date_value, every_year, name, is_official) "
        "WHERE is_active = 1"
    )


def downgrade() -> None:
    # Revision 0036 owns this schema.  Downgrading only the reconciliation
    # revision must leave the 0036 contract intact.
    pass
