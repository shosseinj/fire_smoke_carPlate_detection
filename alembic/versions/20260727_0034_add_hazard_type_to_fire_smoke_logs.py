"""Add hazard type to fire and smoke logs."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260727_0034"
down_revision = "20260727_0033"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("fire_smoke_logs", sa.Column("hazard_type", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("fire_smoke_logs", "hazard_type")
