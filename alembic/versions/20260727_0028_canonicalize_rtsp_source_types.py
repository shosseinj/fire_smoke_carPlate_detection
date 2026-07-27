"""Canonicalize RTSP source types.

Revision ID: 20260727_0028
Revises: 20260727_0027
"""
from __future__ import annotations

from alembic import op


revision = "20260727_0028"
down_revision = "20260727_0027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE sources
        SET source_type = 'rtsp'
        WHERE lower(trim(source_uri)) LIKE 'rtsp://%'
           OR lower(trim(source_uri)) LIKE 'rtsps://%'
        """
    )


def downgrade() -> None:
    return None
