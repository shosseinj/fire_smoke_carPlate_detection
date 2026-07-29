"""Normalize plate log identity, time, and media fields.

Revision ID: 20260729_0043
Revises: 20260729_0042
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260729_0043"
down_revision = "20260729_0042"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "UPDATE plate_logs SET plate_full_number = COALESCE(NULLIF(plate_full_number, ''), plate), "
        "detection_time = COALESCE(detection_time, time), "
        "camera_id = COALESCE(NULLIF(camera_id, ''), camera), "
        "snapshot_path = COALESCE(NULLIF(snapshot_path, ''), NULLIF(snapshot_url, '')), "
        "created_at = COALESCE(created_at, detection_time, time, CURRENT_TIMESTAMP), "
        "updated_at = COALESCE(updated_at, created_at, detection_time, time, CURRENT_TIMESTAMP)"
    )
    op.execute(
        "UPDATE plate_logs p SET source_type = 'static_video' "
        "WHERE EXISTS (SELECT 1 FROM static_videos s WHERE s.source_uri = p.camera_id)"
    )
    op.execute(
        "UPDATE plate_logs SET source_type = 'camera' "
        "WHERE source_type IS NULL OR source_type NOT IN ('camera', 'static_video', 'manual') "
        "OR (source_type = 'manual' AND created_by_user_id IS NULL)"
    )
    op.add_column("plate_logs", sa.Column("static_video_id", sa.Integer(), nullable=True))
    op.add_column("plate_logs", sa.Column("updated_by_user_id", sa.Integer(), nullable=True))
    op.execute(
        "UPDATE plate_logs p SET static_video_id = s.id, camera_id = NULL "
        "FROM static_videos s WHERE p.source_type = 'static_video' AND s.source_uri = p.camera_id"
    )
    op.execute("UPDATE plate_logs SET camera_id = NULL WHERE source_type = 'manual'")

    for name in (
        "idx_plate_logs_time", "idx_plate_logs_camera", "idx_plate_logs_plate",
        "idx_plate_logs_full_number", "idx_plate_logs_camera_id",
        "idx_plate_logs_detection_time", "idx_plate_logs_direction",
    ):
        op.drop_index(name, table_name="plate_logs")

    op.alter_column("plate_logs", "plate_full_number", new_column_name="plate_number")
    op.alter_column("plate_logs", "camera_id", new_column_name="source_uri")
    op.alter_column("plate_logs", "snapshot_path", new_column_name="snapshot_key")
    op.alter_column("plate_logs", "video_url", new_column_name="video_key")
    op.alter_column("plate_logs", "video_key", nullable=True, server_default=None)
    op.execute(
        "UPDATE plate_logs SET snapshot_key = NULLIF(regexp_replace(snapshot_key, '^/media/', ''), ''), "
        "video_key = NULLIF(regexp_replace(video_key, '^/media/', ''), '')"
    )
    op.alter_column("plate_logs", "plate_number", nullable=False)
    op.alter_column("plate_logs", "detection_time", nullable=False)
    op.alter_column("plate_logs", "source_type", nullable=False, server_default="camera")
    op.alter_column("plate_logs", "created_at", nullable=False, server_default=sa.text("CURRENT_TIMESTAMP"))
    op.alter_column("plate_logs", "updated_at", nullable=False, server_default=sa.text("CURRENT_TIMESTAMP"))

    for column in (
        "camera", "time", "plate", "snapshot_url", "details_json", "direction",
        "plate_crop_path", "is_verified", "verified_by_user_id", "verified_at",
    ):
        op.drop_column("plate_logs", column)

    op.create_check_constraint(
        "ck_plate_logs_source_identity",
        "plate_logs",
        "(source_type = 'camera' AND source_uri IS NOT NULL AND static_video_id IS NULL AND created_by_user_id IS NULL) OR "
        "(source_type = 'static_video' AND source_uri IS NULL AND static_video_id IS NOT NULL AND created_by_user_id IS NULL) OR "
        "(source_type = 'manual' AND source_uri IS NULL AND static_video_id IS NULL AND created_by_user_id IS NOT NULL)",
    )
    op.create_check_constraint(
        "ck_plate_logs_confidence", "plate_logs",
        "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
    )
    op.create_index("idx_plate_logs_plate_number", "plate_logs", ["plate_number"])
    op.create_index("idx_plate_logs_source_uri", "plate_logs", ["source_uri"])
    op.create_index("idx_plate_logs_static_video_id", "plate_logs", ["static_video_id"])
    op.create_index("idx_plate_logs_detection_time", "plate_logs", ["detection_time"])
    op.create_index("idx_plate_logs_source_type", "plate_logs", ["source_type"])


def downgrade() -> None:
    op.drop_constraint("ck_plate_logs_source_identity", "plate_logs", type_="check")
    op.drop_constraint("ck_plate_logs_confidence", "plate_logs", type_="check")
    for name in (
        "idx_plate_logs_plate_number", "idx_plate_logs_source_uri",
        "idx_plate_logs_static_video_id", "idx_plate_logs_detection_time",
        "idx_plate_logs_source_type",
    ):
        op.drop_index(name, table_name="plate_logs")
    op.add_column("plate_logs", sa.Column("camera", sa.Text(), nullable=True))
    op.add_column("plate_logs", sa.Column("time", sa.DateTime(timezone=True), nullable=True))
    op.add_column("plate_logs", sa.Column("plate", sa.Text(), nullable=True))
    op.add_column("plate_logs", sa.Column("snapshot_url", sa.Text(), nullable=False, server_default=""))
    op.add_column("plate_logs", sa.Column("details_json", sa.Text(), nullable=False, server_default="{}"))
    op.add_column("plate_logs", sa.Column("direction", sa.String(16), server_default="unknown"))
    op.add_column("plate_logs", sa.Column("plate_crop_path", sa.String(512)))
    op.add_column("plate_logs", sa.Column("is_verified", sa.Integer(), server_default="0"))
    op.add_column("plate_logs", sa.Column("verified_by_user_id", sa.Integer()))
    op.add_column("plate_logs", sa.Column("verified_at", sa.DateTime(timezone=True)))
    op.alter_column("plate_logs", "plate_number", new_column_name="plate_full_number")
    op.alter_column("plate_logs", "source_uri", new_column_name="camera_id")
    op.alter_column("plate_logs", "snapshot_key", new_column_name="snapshot_path")
    op.alter_column("plate_logs", "video_key", new_column_name="video_url")
    op.execute(
        "UPDATE plate_logs SET camera = COALESCE(camera_id, 'manual'), time = detection_time, "
        "plate = plate_full_number, snapshot_url = COALESCE(snapshot_path, '')"
    )
    op.execute("UPDATE plate_logs SET video_url = '' WHERE video_url IS NULL")
    op.alter_column("plate_logs", "video_url", nullable=False, server_default="")
    op.alter_column("plate_logs", "camera", nullable=False)
    op.alter_column("plate_logs", "time", nullable=False)
    op.alter_column("plate_logs", "plate", nullable=False)
    op.drop_column("plate_logs", "updated_by_user_id")
    op.drop_column("plate_logs", "static_video_id")
    op.create_index("idx_plate_logs_time", "plate_logs", ["time"])
    op.create_index("idx_plate_logs_camera", "plate_logs", ["camera"])
    op.create_index("idx_plate_logs_plate", "plate_logs", ["plate"])
    op.create_index("idx_plate_logs_full_number", "plate_logs", ["plate_full_number"])
    op.create_index("idx_plate_logs_camera_id", "plate_logs", ["camera_id"])
    op.create_index("idx_plate_logs_detection_time", "plate_logs", ["detection_time"])
    op.create_index("idx_plate_logs_direction", "plate_logs", ["direction"])
