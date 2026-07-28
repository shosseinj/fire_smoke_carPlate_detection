"""Official holiday imports and multiple occasions per date.

Revision ID: 20260728_0036
Revises: 20260727_0035
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260728_0036"
down_revision = "20260727_0035"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "holidays",
        sa.Column("is_official", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "holidays",
        sa.Column("official_jalali_year", sa.Integer(), nullable=True),
    )
    op.create_check_constraint(
        "ck_holidays_official_year",
        "holidays",
        "(is_official = 0 AND official_jalali_year IS NULL) OR "
        "(is_official = 1 AND official_jalali_year BETWEEN 1200 AND 1600)",
    )
    op.create_index(
        "idx_holidays_official_year",
        "holidays",
        ["is_official", "official_jalali_year"],
        unique=False,
    )

    op.execute(sa.text("DROP INDEX IF EXISTS uq_holidays_active_date_every_year"))
    op.create_index(
        "uq_holidays_active_identity",
        "holidays",
        ["date_value", "every_year", "name", "is_official"],
        unique=True,
        postgresql_where=sa.text("is_active = 1"),
    )


def downgrade() -> None:
    # Refuse to recreate the old one-row-per-date rule when active data would
    # violate it.  This avoids silently deleting or deactivating holidays.
    op.execute(sa.text("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM holidays
                WHERE is_active = 1
                GROUP BY date_value, every_year
                HAVING COUNT(*) > 1
            ) THEN
                RAISE EXCEPTION
                    'Cannot downgrade: multiple active holidays share a date';
            END IF;
        END
        $$
    """))

    op.drop_index("uq_holidays_active_identity", table_name="holidays")
    op.create_index(
        "uq_holidays_active_date_every_year",
        "holidays",
        ["date_value", "every_year"],
        unique=True,
        postgresql_where=sa.text("is_active = 1"),
    )
    op.drop_index("idx_holidays_official_year", table_name="holidays")
    op.drop_constraint("ck_holidays_official_year", "holidays", type_="check")
    op.drop_column("holidays", "official_jalali_year")
    op.drop_column("holidays", "is_official")
