"""Add human pose and quality-weighted recognition policy settings.

Revision ID: 20260727_0030
Revises: 20260727_0027
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260727_0030"
down_revision = "20260727_0027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "face_quality_settings",
        sa.Column("human_pose_enabled", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "face_quality_settings",
        sa.Column("human_pose_min_keypoints", sa.Integer(), nullable=False, server_default="4"),
    )
    op.add_column(
        "face_quality_settings",
        sa.Column(
            "human_pose_keypoint_confidence",
            sa.Float(),
            nullable=False,
            server_default="0.25",
        ),
    )
    op.add_column(
        "face_quality_settings",
        sa.Column("recognition_quality_weight", sa.Float(), nullable=False, server_default="0.5"),
    )


def downgrade() -> None:
    op.drop_column("face_quality_settings", "recognition_quality_weight")
    op.drop_column("face_quality_settings", "human_pose_keypoint_confidence")
    op.drop_column("face_quality_settings", "human_pose_min_keypoints")
    op.drop_column("face_quality_settings", "human_pose_enabled")
