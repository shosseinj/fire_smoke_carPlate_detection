"""Add stable numeric identifiers to static videos.

Revision ID: 20260729_0042
Revises: 20260729_0041
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260729_0042"
down_revision = "20260729_0041"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "static_videos",
        sa.Column("id", sa.Integer(), nullable=True),
    )
    op.execute("CREATE SEQUENCE static_videos_id_seq OWNED BY static_videos.id")
    op.execute(
        "ALTER TABLE static_videos ALTER COLUMN id "
        "SET DEFAULT nextval('static_videos_id_seq')"
    )
    op.execute(
        "UPDATE static_videos SET id = nextval('static_videos_id_seq') "
        "WHERE id IS NULL"
    )
    op.alter_column("static_videos", "id", nullable=False)
    op.create_unique_constraint(
        "uq_static_videos_source_uri",
        "static_videos",
        ["source_uri"],
    )
    op.drop_constraint("static_videos_pkey", "static_videos", type_="primary")
    op.create_primary_key("static_videos_pkey", "static_videos", ["id"])


def downgrade() -> None:
    op.drop_constraint("static_videos_pkey", "static_videos", type_="primary")
    op.create_primary_key(
        "static_videos_pkey",
        "static_videos",
        ["source_uri"],
    )
    op.drop_constraint(
        "uq_static_videos_source_uri",
        "static_videos",
        type_="unique",
    )
    op.drop_column("static_videos", "id")
