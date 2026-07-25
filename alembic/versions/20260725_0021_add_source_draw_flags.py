"""Add per-source draw flags to sources table.

Revision ID: 20260725_0021
Revises: 20260725_0020
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260725_0021"
down_revision = "20260725_0020"
branch_labels = None
depends_on = None


def _column_names() -> set[str]:
    return {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns("sources")
    }


def upgrade() -> None:
    columns = _column_names()
    for name in (
        "draw_human",
        "draw_zone",
        "draw_fire",
        "draw_smoke",
        "draw_vehicle",
        "draw_plate",
    ):
        if name not in columns:
            op.add_column("sources", sa.Column(name, sa.Integer(), server_default="1"))


def downgrade() -> None:
    columns = _column_names()
    for name in (
        "draw_plate",
        "draw_vehicle",
        "draw_smoke",
        "draw_fire",
        "draw_zone",
        "draw_human",
    ):
        if name in columns:
            op.drop_column("sources", name)
