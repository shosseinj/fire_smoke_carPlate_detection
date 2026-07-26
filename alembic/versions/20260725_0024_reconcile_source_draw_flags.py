"""Reconcile source draw flags on databases stamped before schema completion.

Revision ID: 20260725_0024
Revises: 20260725_0023
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260725_0024"
down_revision = "20260725_0023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns("sources")
    }
    for name in (
        "draw_human",
        "draw_zone",
        "draw_fire",
        "draw_smoke",
        "draw_vehicle",
        "draw_plate",
    ):
        if name not in columns:
            op.add_column(
                "sources",
                sa.Column(name, sa.Integer(), server_default="1"),
            )


def downgrade() -> None:
    # Revision 0021 owns these columns; this repair only restores missing schema.
    pass
