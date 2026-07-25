"""Extend plate_logs table with missing columns.

Adds columns and indexes defined in the SQLAlchemy table definition
but never added to the database. Fixes column "detection_time" does
not exist error on GET /api/v1/plate-logs.

Revision ID: 20260725_0010
Revises: 20260724_0009
Create Date: 2026-07-25
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260725_0010"
down_revision: Union[str, Sequence[str], None] = "20260724_0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("plate_logs", sa.Column("plate_id", sa.Integer))
    op.add_column("plate_logs", sa.Column("plate_full_number", sa.String(32)))
    op.add_column("plate_logs", sa.Column("raw_plate_text", sa.String(64)))
    op.add_column("plate_logs", sa.Column("detection_time", sa.DateTime(timezone=True)))
    op.add_column("plate_logs", sa.Column("camera_id", sa.String(200)))
    op.add_column("plate_logs", sa.Column("confidence", sa.Float))
    op.add_column("plate_logs", sa.Column("direction", sa.String(16), server_default="unknown"))
    op.add_column("plate_logs", sa.Column("source_type", sa.String(16), server_default="camera"))
    op.add_column("plate_logs", sa.Column("snapshot_path", sa.String(512)))
    op.add_column("plate_logs", sa.Column("plate_crop_path", sa.String(512)))
    op.add_column("plate_logs", sa.Column("is_verified", sa.Integer, server_default="0"))
    op.add_column("plate_logs", sa.Column("created_by_user_id", sa.Integer))
    op.add_column("plate_logs", sa.Column("verified_by_user_id", sa.Integer))
    op.add_column("plate_logs", sa.Column("verified_at", sa.DateTime(timezone=True)))
    op.add_column("plate_logs", sa.Column("notes", sa.Text))
    op.add_column("plate_logs", sa.Column("created_at", sa.DateTime(timezone=True)))
    op.add_column("plate_logs", sa.Column("updated_at", sa.DateTime(timezone=True)))

    op.create_index("idx_plate_logs_plate_id", "plate_logs", ["plate_id"])
    op.create_index("idx_plate_logs_full_number", "plate_logs", ["plate_full_number"])
    op.create_index("idx_plate_logs_camera_id", "plate_logs", ["camera_id"])
    op.create_index("idx_plate_logs_detection_time", "plate_logs", ["detection_time"])
    op.create_index("idx_plate_logs_direction", "plate_logs", ["direction"])


def downgrade() -> None:
    op.drop_index("idx_plate_logs_direction", table_name="plate_logs")
    op.drop_index("idx_plate_logs_detection_time", table_name="plate_logs")
    op.drop_index("idx_plate_logs_camera_id", table_name="plate_logs")
    op.drop_index("idx_plate_logs_full_number", table_name="plate_logs")
    op.drop_index("idx_plate_logs_plate_id", table_name="plate_logs")

    op.drop_column("plate_logs", "updated_at")
    op.drop_column("plate_logs", "created_at")
    op.drop_column("plate_logs", "notes")
    op.drop_column("plate_logs", "verified_at")
    op.drop_column("plate_logs", "verified_by_user_id")
    op.drop_column("plate_logs", "created_by_user_id")
    op.drop_column("plate_logs", "is_verified")
    op.drop_column("plate_logs", "plate_crop_path")
    op.drop_column("plate_logs", "snapshot_path")
    op.drop_column("plate_logs", "source_type")
    op.drop_column("plate_logs", "direction")
    op.drop_column("plate_logs", "confidence")
    op.drop_column("plate_logs", "camera_id")
    op.drop_column("plate_logs", "detection_time")
    op.drop_column("plate_logs", "raw_plate_text")
    op.drop_column("plate_logs", "plate_full_number")
    op.drop_column("plate_logs", "plate_id")
