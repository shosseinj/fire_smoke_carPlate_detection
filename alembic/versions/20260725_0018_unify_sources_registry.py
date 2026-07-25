"""Unify the source registry into the ``sources`` table and drop ``cameras``.

The sources table becomes the single source-of-truth for both task-manager
selection and per-source confidence settings. Existing camera rows are copied
into the new unified layout, while the old cameras table is removed.

Revision ID: 20260725_0018
Revises: 20260725_0017
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260725_0018"
down_revision = "20260725_0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sources_new",
        sa.Column("id", sa.Integer(), nullable=True),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("enabled", sa.Integer(), nullable=True, server_default="1"),
        sa.Column("tasks_json", sa.Text(), nullable=True, server_default="[]"),
        sa.Column("frame_width", sa.Integer(), nullable=True, server_default="640"),
        sa.Column("frame_height", sa.Integer(), nullable=True, server_default="640"),
        sa.Column("room_id", sa.Integer(), nullable=True),
        sa.Column("source_type", sa.String(length=16), nullable=True, server_default="rtsp"),
        sa.Column("metadata_json", sa.Text(), nullable=True, server_default="{}"),
        sa.Column("created_at_utc", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at_utc",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("fire_confidence", sa.Float(), nullable=True),
        sa.Column("smoke_confidence", sa.Float(), nullable=True),
        sa.Column("plate_confidence", sa.Float(), nullable=True),
        sa.Column("plate_iou", sa.Float(), nullable=True),
        sa.Column("vehicle_confidence", sa.Float(), nullable=True),
        sa.Column("vehicle_iou", sa.Float(), nullable=True),
        sa.Column("face_human_confidence", sa.Float(), nullable=True),
        sa.Column("face_detection_confidence", sa.Float(), nullable=True),
        sa.Column("face_recognition_threshold", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("source_uri"),
        sa.UniqueConstraint("id", name="uq_sources_id"),
        sa.ForeignKeyConstraint(["room_id"], ["rooms.id"], ondelete="SET NULL"),
    )
    op.create_index("idx_sources_new_enabled", "sources_new", ["enabled"], unique=False)
    op.create_index("idx_sources_new_room", "sources_new", ["room_id"], unique=False)

    bind = op.get_bind()
    # Preserve default settings rows first.
    bind.execute(
        sa.text(
            """
            INSERT INTO sources_new (
                source_uri, fire_confidence, smoke_confidence, plate_confidence,
                plate_iou, vehicle_confidence, vehicle_iou, face_human_confidence,
                face_detection_confidence, face_recognition_threshold, updated_at_utc
            )
            SELECT
                source_uri, fire_confidence, smoke_confidence, plate_confidence,
                plate_iou, vehicle_confidence, vehicle_iou, face_human_confidence,
                face_detection_confidence, face_recognition_threshold, updated_at_utc
            FROM sources
            """
        )
    )
    # Merge camera/task-manager rows into the unified table while preserving
    # any confidence overrides already present in sources.
    bind.execute(
        sa.text(
            """
            INSERT INTO sources_new (
                id, source_uri, name, enabled, tasks_json, frame_width,
                frame_height, room_id, source_type, metadata_json,
                created_at_utc, updated_at_utc
            )
            SELECT
                c.id, c.source_uri, c.name, c.enabled, c.tasks_json, c.frame_width,
                c.frame_height, c.room_id, c.source_type, c.metadata_json,
                c.created_at_utc, c.updated_at_utc
            FROM cameras AS c
            ON CONFLICT(source_uri) DO UPDATE SET
                id = COALESCE(sources_new.id, excluded.id),
                name = excluded.name,
                enabled = excluded.enabled,
                tasks_json = excluded.tasks_json,
                frame_width = excluded.frame_width,
                frame_height = excluded.frame_height,
                room_id = excluded.room_id,
                source_type = excluded.source_type,
                metadata_json = excluded.metadata_json,
                created_at_utc = COALESCE(sources_new.created_at_utc, excluded.created_at_utc),
                updated_at_utc = excluded.updated_at_utc
            """
        )
    )
    op.drop_table("cameras")
    op.drop_table("sources")
    op.rename_table("sources_new", "sources")


def downgrade() -> None:
    op.create_table(
        "cameras",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("enabled", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("tasks_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("frame_width", sa.Integer(), nullable=False, server_default="640"),
        sa.Column("frame_height", sa.Integer(), nullable=False, server_default="640"),
        sa.Column("room_id", sa.Integer(), nullable=True),
        sa.Column("source_type", sa.String(length=16), nullable=False, server_default="rtsp"),
        sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at_utc", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_cameras_enabled", "cameras", ["enabled"], unique=False)
    op.create_index("idx_cameras_room", "cameras", ["room_id"], unique=False)

    op.drop_table("sources")
    op.execute(
        """
        CREATE TABLE sources (
            source_uri TEXT NOT NULL PRIMARY KEY,
            fire_confidence FLOAT,
            smoke_confidence FLOAT,
            plate_confidence FLOAT,
            plate_iou FLOAT,
            vehicle_confidence FLOAT,
            vehicle_iou FLOAT,
            face_human_confidence FLOAT,
            face_detection_confidence FLOAT,
            face_recognition_threshold FLOAT,
            updated_at_utc DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
