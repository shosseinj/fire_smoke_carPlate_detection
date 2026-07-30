"""Add per-source attendance counting policy.

Revision ID: 20260730_0046
Revises: 20260729_0045
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260730_0046"
down_revision = "20260729_0045"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "sources",
        sa.Column(
            "counts_for_attendance",
            sa.Integer(),
            nullable=False,
            server_default="1",
        ),
    )


def downgrade() -> None:
    op.drop_column("sources", "counts_for_attendance")
