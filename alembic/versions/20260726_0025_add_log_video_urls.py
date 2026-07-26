"""Add saved video URLs to fire and plate logs.

Revision ID: 20260726_0025
Revises: 20260725_0024
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260726_0025"
down_revision = "20260725_0024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("fire_smoke_logs", sa.Column("video_url", sa.Text(), nullable=False, server_default=""))
    op.add_column("plate_logs", sa.Column("video_url", sa.Text(), nullable=False, server_default=""))


def downgrade() -> None:
    op.drop_column("plate_logs", "video_url")
    op.drop_column("fire_smoke_logs", "video_url")
