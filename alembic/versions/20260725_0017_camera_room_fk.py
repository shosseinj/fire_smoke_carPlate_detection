"""Replace camera section association with a room foreign key.

Revision ID: 20260725_0017
Revises: 20260725_0016
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260725_0017"
down_revision = "20260725_0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("cameras", sa.Column("room_id", sa.Integer(), nullable=True))
    op.execute(
        """
        UPDATE cameras AS c SET room_id = candidate.room_id
        FROM (
            SELECT section_id, MIN(id) AS room_id
            FROM rooms
            WHERE section_id IS NOT NULL
            GROUP BY section_id
            HAVING COUNT(*) = 1
        ) AS candidate
        WHERE c.section_id = candidate.section_id
        """
    )
    op.create_foreign_key(
        "fk_cameras_room_id", "cameras", "rooms", ["room_id"], ["id"], ondelete="SET NULL"
    )
    op.create_index("idx_cameras_room", "cameras", ["room_id"])
    op.drop_column("cameras", "section_id")


def downgrade() -> None:
    op.add_column("cameras", sa.Column("section_id", sa.Integer(), nullable=True))
    op.execute(
        "UPDATE cameras AS c SET section_id = r.section_id FROM rooms AS r WHERE c.room_id = r.id"
    )
    op.drop_index("idx_cameras_room", table_name="cameras")
    op.drop_constraint("fk_cameras_room_id", "cameras", type_="foreignkey")
    op.drop_column("cameras", "room_id")
