"""Add created_by and updated_by audit FK columns to core tables.

Adds ``created_by`` and ``updated_by`` ``INTEGER`` columns referencing
``users(id) ON DELETE SET NULL`` to ``buildings``, ``sections``,
``rooms``, ``cam``, ``personnel``, and ``holidays``.

This enables per-row user-ownership tracking for admin-managed entities.

Revision ID: 20260727_0027
Revises: 20260726_0026
Create Date: 2026-07-27
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '20260727_0027'
down_revision: Union[str, Sequence[str], None] = '20260726_0026'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_TABLES = ("buildings", "sections", "rooms", "cam", "personnel", "holidays")


def upgrade() -> None:
    for table in _TABLES:
        op.add_column(
            table,
            sa.Column("created_by", sa.Integer, sa.ForeignKey("users.id", ondelete="SET NULL")),
        )
        op.add_column(
            table,
            sa.Column("updated_by", sa.Integer, sa.ForeignKey("users.id", ondelete="SET NULL")),
        )
        op.create_index(f"idx_{table}_created_by", table, ["created_by"])
        op.create_index(f"idx_{table}_updated_by", table, ["updated_by"])


def downgrade() -> None:
    for table in reversed(_TABLES):
        op.drop_index(f"idx_{table}_updated_by", table_name=table)
        op.drop_index(f"idx_{table}_created_by", table_name=table)
        op.drop_column(table, "updated_by")
        op.drop_column(table, "created_by")
