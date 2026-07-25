"""Add per-source FPS override to sources table.

Revision ID: 20260725_0022
Revises: 20260725_0021
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260725_0022"
down_revision = "20260725_0021"
branch_labels = None
depends_on = None


def _column_names() -> set[str]:
    return {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns("sources")
    }


def upgrade() -> None:
    columns = _column_names()
    if "fps" not in columns:
        op.add_column("sources", sa.Column("fps", sa.Float()))


def downgrade() -> None:
    columns = _column_names()
    if "fps" in columns:
        op.drop_column("sources", "fps")
