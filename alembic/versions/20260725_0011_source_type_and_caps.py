"""Add source_type column to cameras table.

Revision ID: 20260725_0011
Revises: 20260725_0010
Create Date: 2026-07-25
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "20260725_0011"
down_revision: str = "20260725_0010"
branch_labels = None
depends_on = None


def _column_exists(table: str, column: str) -> bool:
    conn = op.get_bind()
    inspector = inspect(conn)
    columns = {c["name"] for c in inspector.get_columns(table)}
    return column in columns


def _constraint_exists(table: str, constraint: str) -> bool:
    conn = op.get_bind()
    inspector = inspect(conn)
    for c in inspector.get_check_constraints(table):
        if c["name"] == constraint:
            return True
    return False


def upgrade() -> None:
    if not _column_exists("cameras", "source_type"):
        op.execute(sa.text("""
            ALTER TABLE cameras
            ADD COLUMN source_type VARCHAR(16) NOT NULL DEFAULT 'rtsp'
        """))
    if not _constraint_exists("cameras", "ck_cameras_source_type"):
        op.execute(sa.text("""
            ALTER TABLE cameras
            ADD CONSTRAINT ck_cameras_source_type
            CHECK (source_type IN ('rtsp', 'static_video'))
        """))


def downgrade() -> None:
    if _constraint_exists("cameras", "ck_cameras_source_type"):
        op.execute(sa.text("ALTER TABLE cameras DROP CONSTRAINT ck_cameras_source_type"))
    if _column_exists("cameras", "source_type"):
        op.execute(sa.text("ALTER TABLE cameras DROP COLUMN source_type"))
