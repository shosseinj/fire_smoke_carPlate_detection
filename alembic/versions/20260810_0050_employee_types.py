"""Replace personnel.employee_type choice text with a protected lookup table.

Revision ID: 20260810_0050
Revises: 20260803_0049
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260810_0050"
down_revision = "20260803_0049"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "employee_types",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("code", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("TRUE")),
        sa.Column("created_at_utc", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at_utc", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("created_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("updated_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.UniqueConstraint("code", name="uq_employee_types_code"),
    )
    op.create_index("idx_employee_types_active", "employee_types", ["is_active"])
    op.create_index("idx_employee_types_name", "employee_types", ["name"])

    # This is migration of pre-existing personnel data, not default seeding.
    # On a fresh/empty database no employee-type rows are inserted here;
    # init_db.py owns all normal initial/default seed data.
    op.execute(
        "UPDATE personnel SET employee_type = 'unknown' "
        "WHERE employee_type IS NULL OR BTRIM(employee_type) = ''"
    )
    op.execute(
        "INSERT INTO employee_types (code, name, is_active) "
        "SELECT DISTINCT BTRIM(p.employee_type), "
        "CASE LOWER(BTRIM(p.employee_type)) "
        "WHEN 'contractor' THEN 'پیمانکار' "
        "WHEN 'customer' THEN 'مشتری' "
        "WHEN 'guest' THEN 'مهمان' "
        "WHEN 'employee' THEN 'کارمند' "
        "WHEN 'unknown' THEN 'نامشخص' "
        "ELSE BTRIM(p.employee_type) END, TRUE "
        "FROM personnel AS p "
        "WHERE p.employee_type IS NOT NULL AND BTRIM(p.employee_type) <> '' "
        "ON CONFLICT (code) DO NOTHING"
    )

    op.add_column("personnel", sa.Column("employee_type_id", sa.Integer(), nullable=True))
    op.execute(
        "UPDATE personnel AS p SET employee_type_id = et.id "
        "FROM employee_types AS et WHERE et.code = BTRIM(p.employee_type)"
    )
    op.alter_column("personnel", "employee_type_id", existing_type=sa.Integer(), nullable=False)
    op.create_foreign_key(
        "personnel_employee_type_id_fkey",
        "personnel",
        "employee_types",
        ["employee_type_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index("idx_personnel_employee_type", "personnel", ["employee_type_id"])
    op.drop_column("personnel", "employee_type")


def downgrade() -> None:
    op.add_column(
        "personnel",
        sa.Column("employee_type", sa.Text(), nullable=True, server_default="unknown"),
    )
    op.execute(
        "UPDATE personnel AS p SET employee_type = et.code "
        "FROM employee_types AS et WHERE et.id = p.employee_type_id"
    )
    op.execute("UPDATE personnel SET employee_type = 'unknown' WHERE employee_type IS NULL")
    op.alter_column("personnel", "employee_type", existing_type=sa.Text(), nullable=False)
    op.drop_index("idx_personnel_employee_type", table_name="personnel")
    op.drop_constraint("personnel_employee_type_id_fkey", "personnel", type_="foreignkey")
    op.drop_column("personnel", "employee_type_id")
    op.drop_table("employee_types")
