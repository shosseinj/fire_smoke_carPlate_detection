from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260725_0017"
down_revision = "20260725_0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("cameras_pkey", "cameras", type_="primary")

    op.add_column(
        "cameras",
        sa.Column("id", sa.Integer(), nullable=True),
    )

    op.execute("CREATE SEQUENCE IF NOT EXISTS cameras_id_seq START WITH 1 OWNED BY cameras.id")
    op.execute("UPDATE cameras SET id = nextval('cameras_id_seq')")
    op.execute("ALTER TABLE cameras ALTER COLUMN id SET DEFAULT nextval('cameras_id_seq')")

    op.alter_column("cameras", "id", nullable=False)
    op.create_primary_key("cameras_pkey", "cameras", ["id"])

    op.create_unique_constraint("uq_cameras_source_uri", "cameras", ["source_uri"])

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

    op.drop_constraint("uq_cameras_source_uri", "cameras", type_="unique")
    op.drop_constraint("cameras_pkey", "cameras", type_="primary")
    op.drop_column("cameras", "id")
    op.execute("DROP SEQUENCE IF EXISTS cameras_id_seq")
    op.create_primary_key("cameras_pkey", "cameras", ["source_uri"])
