"""Add room fields supported by the room PATCH endpoint."""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260727_0028"
down_revision: Union[str, Sequence[str], None] = "20260727_0027"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("rooms", sa.Column("room_number", sa.Text(), nullable=True))
    op.add_column("rooms", sa.Column("room_type", sa.Text(), nullable=True))
    op.add_column("rooms", sa.Column("is_active", sa.Integer(), nullable=False, server_default="1"))


def downgrade() -> None:
    op.drop_column("rooms", "is_active")
    op.drop_column("rooms", "room_type")
    op.drop_column("rooms", "room_number")
