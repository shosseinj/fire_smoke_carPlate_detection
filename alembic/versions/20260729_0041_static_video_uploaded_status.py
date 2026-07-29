"""Add the pre-queue uploaded status for static videos.

Revision ID: 20260729_0041
Revises: 20260728_0040
"""
from __future__ import annotations

from alembic import op


revision = "20260729_0041"
down_revision = "20260728_0040"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "ck_static_videos_processing_status",
        "static_videos",
        type_="check",
    )
    op.create_check_constraint(
        "ck_static_videos_processing_status",
        "static_videos",
        "processing_status IN "
        "('uploaded', 'queued', 'processing', 'completed', 'failed')",
    )
    op.alter_column(
        "static_videos",
        "processing_status",
        server_default="uploaded",
    )
    op.execute(
        "UPDATE static_videos v SET processing_status = 'uploaded' "
        "WHERE v.processing_status = 'queued' "
        "AND NOT EXISTS ("
        "SELECT 1 FROM sources s WHERE s.source_uri = v.source_uri"
        ")"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE static_videos SET processing_status = 'queued' "
        "WHERE processing_status = 'uploaded'"
    )
    op.drop_constraint(
        "ck_static_videos_processing_status",
        "static_videos",
        type_="check",
    )
    op.create_check_constraint(
        "ck_static_videos_processing_status",
        "static_videos",
        "processing_status IN ('queued', 'processing', 'completed', 'failed')",
    )
    op.alter_column(
        "static_videos",
        "processing_status",
        server_default="queued",
    )
