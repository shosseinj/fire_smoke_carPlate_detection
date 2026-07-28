"""Add static-video processing lifecycle state.

Revision ID: 20260728_0040
Revises: 20260728_0039
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260728_0040"
down_revision = "20260728_0039"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("static_videos", sa.Column("processing_status", sa.String(16), nullable=False, server_default="queued"))
    op.add_column("static_videos", sa.Column("processing_error", sa.Text(), nullable=True))
    op.add_column("static_videos", sa.Column("processing_started_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("static_videos", sa.Column("processing_completed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("static_videos", sa.Column("processing_attempts", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("static_videos", sa.Column("is_processed", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("static_videos", sa.Column("loop", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("static_videos", sa.Column("source_config_json", sa.Text(), nullable=True))
    op.create_check_constraint(
        "ck_static_videos_processing_status",
        "static_videos",
        "processing_status IN ('queued', 'processing', 'completed', 'failed')",
    )
    op.create_check_constraint(
        "ck_static_videos_processing_attempts",
        "static_videos",
        "processing_attempts >= 0",
    )
    op.execute(
        "UPDATE static_videos v SET loop = COALESCE(s.loop, 0) "
        "FROM sources s WHERE s.source_uri = v.source_uri"
    )


def downgrade() -> None:
    op.drop_constraint("ck_static_videos_processing_attempts", "static_videos", type_="check")
    op.drop_constraint("ck_static_videos_processing_status", "static_videos", type_="check")
    op.drop_column("static_videos", "source_config_json")
    op.drop_column("static_videos", "loop")
    op.execute("ALTER TABLE static_videos DROP COLUMN IF EXISTS is_processed")
    op.drop_column("static_videos", "processing_attempts")
    op.drop_column("static_videos", "processing_completed_at")
    op.drop_column("static_videos", "processing_started_at")
    op.drop_column("static_videos", "processing_error")
    op.drop_column("static_videos", "processing_status")
