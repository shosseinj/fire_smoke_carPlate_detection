"""Drop the old singleton sources table and recreate per-source.

The previous migration created a singleton ``sources`` table with ``id``
as PK, a singleton CHECK constraint, ``rtsp_source_count``,
``video_loop``, and NOT NULL defaults for the 9 confidence fields.

This migration drops that table and creates the new per-source layout
keyed on ``source_uri`` with nullable typed columns.

Revision ID: 20260725_0015
Revises: 20260725_0014
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260725_0015"
down_revision = "20260725_0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("sources")
    op.create_table(
        "sources",
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column("fire_confidence", sa.Float(), nullable=True),
        sa.Column("smoke_confidence", sa.Float(), nullable=True),
        sa.Column("plate_confidence", sa.Float(), nullable=True),
        sa.Column("plate_iou", sa.Float(), nullable=True),
        sa.Column("vehicle_confidence", sa.Float(), nullable=True),
        sa.Column("vehicle_iou", sa.Float(), nullable=True),
        sa.Column("face_human_confidence", sa.Float(), nullable=True),
        sa.Column("face_detection_confidence", sa.Float(), nullable=True),
        sa.Column("face_recognition_threshold", sa.Float(), nullable=True),
        sa.Column(
            "updated_at_utc",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("source_uri"),
    )
    op.execute(
        "INSERT INTO sources (source_uri, updated_at_utc) "
        "VALUES ('__default__', CURRENT_TIMESTAMP)"
    )


def downgrade() -> None:
    op.drop_table("sources")
    op.create_table(
        "sources",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("fire_confidence", sa.Float(), nullable=False, server_default="0.3"),
        sa.Column("smoke_confidence", sa.Float(), nullable=False, server_default="0.3"),
        sa.Column("plate_confidence", sa.Float(), nullable=False, server_default="0.3"),
        sa.Column("plate_iou", sa.Float(), nullable=False, server_default="0.45"),
        sa.Column("vehicle_confidence", sa.Float(), nullable=False, server_default="0.35"),
        sa.Column("vehicle_iou", sa.Float(), nullable=False, server_default="0.45"),
        sa.Column("face_human_confidence", sa.Float(), nullable=False, server_default="0.4"),
        sa.Column("face_detection_confidence", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("face_recognition_threshold", sa.Float(), nullable=False, server_default="0.45"),
        sa.Column("rtsp_source_count", sa.Integer(), nullable=False, server_default="128"),
        sa.Column("video_loop", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("updated_at_utc", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("id = 1", name="ck_sources_singleton"),
    )
    op.execute(
        "INSERT INTO sources (id, updated_at_utc) VALUES (1, CURRENT_TIMESTAMP)"
    )
