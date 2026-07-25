"""Create static_videos table for uploaded static video file metadata.

Revision ID: 20260725_0012
Revises: 20260725_0011
"""
from alembic import op
import sqlalchemy as sa

revision = "20260725_0012"
down_revision = "20260725_0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "static_videos",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(16), nullable=False, server_default="static_video"),
        sa.Column("created_at_utc", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at_utc", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("static_videos")
