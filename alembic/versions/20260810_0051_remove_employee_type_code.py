"""Remove the redundant employee_types.code column.

Revision ID: 20260810_0051
Revises: 20260810_0050
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260810_0051"
down_revision = "20260810_0050"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Keep the human-readable name populated by the data migration and remove
    # the temporary compatibility code used only while converting old rows.
    op.drop_index("idx_employee_types_name", table_name="employee_types")
    op.drop_constraint("uq_employee_types_code", "employee_types", type_="unique")
    op.create_unique_constraint("uq_employee_types_name", "employee_types", ["name"])
    op.drop_column("employee_types", "code")


def downgrade() -> None:
    op.add_column("employee_types", sa.Column("code", sa.String(length=64), nullable=True))
    op.execute("UPDATE employee_types SET code = name")
    op.alter_column("employee_types", "code", existing_type=sa.String(length=64), nullable=False)
    op.create_unique_constraint("uq_employee_types_code", "employee_types", ["code"])
    op.drop_constraint("uq_employee_types_name", "employee_types", type_="unique")
    op.create_index("idx_employee_types_name", "employee_types", ["name"])
