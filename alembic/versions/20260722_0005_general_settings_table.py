"""Create general_settings table for legacy compatibility.

Revision ID: 20260722_0005
Revises: 20260722_0004
Create Date: 2026-07-22
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260722_0005"
down_revision: Union[str, Sequence[str], None] = "20260722_0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        sa.text(
            "CREATE TABLE IF NOT EXISTS general_settings ( "
            "  id INTEGER PRIMARY KEY CHECK (id = 1), "
            "  enable_processing INTEGER NOT NULL DEFAULT 1, "
            "  process_fire INTEGER NOT NULL DEFAULT 0, "
            "  process_plate INTEGER NOT NULL DEFAULT 0, "
            "  counts_for_attendance INTEGER NOT NULL DEFAULT 1, "
            "  margin_level REAL NOT NULL DEFAULT 1.0, "
            "  draw_box INTEGER NOT NULL DEFAULT 1, "
            "  draw_face INTEGER NOT NULL DEFAULT 1, "
            "  draw_skeleton INTEGER NOT NULL DEFAULT 0, "
            "  draw_zones INTEGER NOT NULL DEFAULT 1, "
            "  face_rec_score REAL NOT NULL DEFAULT 0.4, "
            "  face_det_score REAL NOT NULL DEFAULT 0.4, "
            "  human_det_score REAL NOT NULL DEFAULT 0.4, "
            "  confirmation_threshold REAL NOT NULL DEFAULT 0.6, "
            "  created_by INTEGER REFERENCES users(id) ON DELETE SET NULL, "
            "  updated_by INTEGER REFERENCES users(id) ON DELETE SET NULL, "
            "  created_at_utc TEXT NOT NULL, "
            "  updated_at_utc TEXT NOT NULL "
            ")"
        )
    )

    op.execute(
        sa.text(
            "INSERT INTO general_settings (id, enable_processing, process_fire, process_plate, "
            "counts_for_attendance, margin_level, draw_box, draw_face, draw_skeleton, draw_zones, "
            "face_rec_score, face_det_score, human_det_score, confirmation_threshold, "
            "created_at_utc, updated_at_utc) "
            "SELECT 1, 1, 0, 0, 1, 1.0, 1, 1, 0, 1, 0.4, 0.4, 0.4, 0.6, "
            "datetime('now'), datetime('now') "
            "WHERE NOT EXISTS (SELECT 1 FROM general_settings WHERE id = 1)"
        )
    )


def downgrade() -> None:
    op.execute(sa.text("DROP TABLE IF EXISTS general_settings CASCADE"))
