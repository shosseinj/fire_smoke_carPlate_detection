"""Create sources table for per-source confidence-field overrides.

The ``sources`` table stores 9 nullable confidence fields keyed by
``source_uri`` (TEXT PK).  A well-known row with ``source_uri = '__default__'``
holds global defaults.  Per-source rows override individual fields.

Revision ID: 20260725_0014
Revises: 20260725_0013
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260725_0014"
down_revision = "20260725_0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
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
    # Insert the global-defaults row
    op.execute(
        "INSERT INTO sources (source_uri, updated_at_utc) "
        "VALUES ('__default__', CURRENT_TIMESTAMP)"
    )


def downgrade() -> None:
    op.drop_table("sources")
