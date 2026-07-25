"""Add per-source loop column to sources table.

Revision ID: 20260725_0020
Revises: 20260725_0019
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260725_0020"
down_revision = "20260725_0019"
branch_labels = None
depends_on = None


def _column_names() -> set[str]:
    return {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns("sources")
    }


def upgrade() -> None:
    columns = _column_names()
    if "loop" not in columns:
        op.add_column(
            "sources",
            sa.Column("loop", sa.Integer(), server_default="1"),
        )


def downgrade() -> None:
    columns = _column_names()
    if "loop" in columns:
        op.drop_column("sources", "loop")
