"""Create cam table and link rooms to cams.

Hierarchy: building -> section -> cam -> room.

Revision ID: 20260725_0023
Revises: 20260725_0022
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260725_0023"
down_revision = "20260725_0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cam",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("camera_name", sa.Text(), nullable=False),
        sa.Column("camera_number", sa.Integer(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("high", sa.Integer(), nullable=False),
        sa.Column("source_type", sa.String(length=16), nullable=False),
        sa.Column("section_id", sa.Integer(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column(
            "created_at_utc",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at_utc",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint("camera_number > 0", name="ck_cam_camera_number_positive"),
        sa.CheckConstraint("width > 0", name="ck_cam_width_positive"),
        sa.CheckConstraint("high > 0", name="ck_cam_high_positive"),
        sa.CheckConstraint(
            "source_type IN ('usb', 'rtsp', 'other')",
            name="ck_cam_source_type",
        ),
        sa.ForeignKeyConstraint(
            ["section_id"],
            ["sections.id"],
            name="fk_cam_section_id",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "section_id", "camera_number", name="uq_cam_section_camera_number"
        ),
    )
    op.create_index("idx_cam_section", "cam", ["section_id"])
    op.create_index("idx_cam_source_type", "cam", ["source_type"])

    # Nullable keeps existing rooms valid until they are assigned to a cam.
    # New/updated room records use this FK as their parent relationship.
    op.add_column("rooms", sa.Column("cam_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_rooms_cam_id",
        "rooms",
        "cam",
        ["cam_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index("idx_rooms_cam", "rooms", ["cam_id"])


def downgrade() -> None:
    op.drop_index("idx_rooms_cam", table_name="rooms")
    op.drop_constraint("fk_rooms_cam_id", "rooms", type_="foreignkey")
    op.drop_column("rooms", "cam_id")
    op.drop_index("idx_cam_source_type", table_name="cam")
    op.drop_index("idx_cam_section", table_name="cam")
    op.drop_table("cam")
