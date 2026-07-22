"""Step 6 schema: detection_logs table.

Create the detection_logs table for storing face recognition,
plate recognition, and fire/smoke detection events.

Revision ID: 20260722_0004
Revises: 20260722_0003
Create Date: 2026-07-22
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260722_0004"
down_revision: Union[str, Sequence[str], None] = "20260722_0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        sa.text(
            "CREATE TABLE IF NOT EXISTS detection_logs ( "
            "  id SERIAL PRIMARY KEY, "
            "  source_system VARCHAR(64) NOT NULL DEFAULT 'face_recognition', "
            "  source_event_key VARCHAR(255) UNIQUE, "
            "  source_human_log_id INTEGER REFERENCES human_logs(id) ON DELETE SET NULL, "
            "  personnel_id INTEGER REFERENCES personnel(id) ON DELETE SET NULL, "
            "  person VARCHAR(255) NOT NULL DEFAULT 'Unknown', "
            "  confidence FLOAT NOT NULL DEFAULT 0.0, "
            "  detection_time TIMESTAMPTZ NOT NULL, "
            "  ref_img_id VARCHAR(255), "
            "  room_id INTEGER REFERENCES rooms(id) ON DELETE SET NULL, "
            "  camera_id TEXT REFERENCES cameras(camera_id) ON DELETE SET NULL, "
            "  access_granted INTEGER NOT NULL DEFAULT 0, "
            "  counts_for_attendance INTEGER NOT NULL DEFAULT 1, "
            "  log_type VARCHAR(64) NOT NULL DEFAULT 'real_time', "
            "  import_source_parts JSONB, "
            "  face_image TEXT, "
            "  body_image TEXT, "
            "  snapshot_image TEXT, "
            "  video TEXT, "
            "  face_video_or_unknown_faces TEXT, "
            "  created_by INTEGER REFERENCES users(id) ON DELETE SET NULL, "
            "  updated_by INTEGER REFERENCES users(id) ON DELETE SET NULL, "
            "  created_at_utc TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, "
            "  updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP "
            ")"
        )
    )

    op.execute(
        sa.text(
            "CREATE INDEX IF NOT EXISTS idx_detection_logs_time "
            "ON detection_logs (detection_time)"
        )
    )
    op.execute(
        sa.text(
            "CREATE INDEX IF NOT EXISTS idx_detection_logs_personnel "
            "ON detection_logs (personnel_id)"
        )
    )
    op.execute(
        sa.text(
            "CREATE INDEX IF NOT EXISTS idx_detection_logs_person "
            "ON detection_logs (person)"
        )
    )
    op.execute(
        sa.text(
            "CREATE INDEX IF NOT EXISTS idx_detection_logs_room "
            "ON detection_logs (room_id)"
        )
    )
    op.execute(
        sa.text(
            "CREATE INDEX IF NOT EXISTS idx_detection_logs_camera "
            "ON detection_logs (camera_id)"
        )
    )
    op.execute(
        sa.text(
            "CREATE INDEX IF NOT EXISTS idx_detection_logs_attendance "
            "ON detection_logs (counts_for_attendance)"
        )
    )
    op.execute(
        sa.text(
            "CREATE INDEX IF NOT EXISTS idx_detection_logs_type "
            "ON detection_logs (log_type)"
        )
    )
    op.execute(
        sa.text(
            "CREATE INDEX IF NOT EXISTS idx_detection_logs_source_event "
            "ON detection_logs (source_event_key)"
        )
    )


def downgrade() -> None:
    op.execute(
        sa.text("DROP INDEX IF EXISTS idx_detection_logs_source_event")
    )
    op.execute(
        sa.text("DROP INDEX IF EXISTS idx_detection_logs_type")
    )
    op.execute(
        sa.text("DROP INDEX IF EXISTS idx_detection_logs_attendance")
    )
    op.execute(
        sa.text("DROP INDEX IF EXISTS idx_detection_logs_camera")
    )
    op.execute(
        sa.text("DROP INDEX IF EXISTS idx_detection_logs_room")
    )
    op.execute(
        sa.text("DROP INDEX IF EXISTS idx_detection_logs_person")
    )
    op.execute(
        sa.text("DROP INDEX IF EXISTS idx_detection_logs_personnel")
    )
    op.execute(
        sa.text("DROP INDEX IF EXISTS idx_detection_logs_time")
    )
    op.execute(
        sa.text("DROP TABLE IF EXISTS detection_logs CASCADE")
    )
