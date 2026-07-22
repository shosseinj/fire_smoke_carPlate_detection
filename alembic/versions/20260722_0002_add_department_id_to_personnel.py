"""Add department_id column to personnel table.

Maps to the existing sections table so that the old "department" concept
is represented by a section reference without a separate Department model.

Revision ID: 20260722_0002
Revises: 20260722_0001
Create Date: 2026-07-22
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260722_0002"
down_revision: Union[str, Sequence[str], None] = "20260722_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        sa.text(
            "ALTER TABLE personnel "
            "ADD COLUMN department_id INTEGER "
            "REFERENCES sections(id) ON DELETE SET NULL"
        )
    )
    op.create_index("idx_personnel_department", "personnel", ["department_id"])


def downgrade() -> None:
    op.drop_index("idx_personnel_department", table_name="personnel")
    op.execute(sa.text("ALTER TABLE personnel DROP COLUMN department_id"))
