"""Add floor and is_active columns to sections table."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260727_0035"
down_revision = "20260727_0034"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("sections", sa.Column("floor", sa.Text(), nullable=True))
    op.add_column("sections", sa.Column("is_active", sa.Integer(), nullable=False, server_default="1"))


def downgrade() -> None:
    op.drop_column("sections", "is_active")
    op.drop_column("sections", "floor")
