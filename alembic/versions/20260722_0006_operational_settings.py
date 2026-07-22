"""Move runtime operational configuration into database-backed settings.

Revision ID: 20260722_0006
Revises: 20260722_0005
"""
from alembic import op
import sqlalchemy as sa

revision = "20260722_0006"
down_revision = "20260722_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "general_settings",
        sa.Column("operational_json", sa.Text(), nullable=False, server_default="{}"),
    )


def downgrade() -> None:
    op.drop_column("general_settings", "operational_json")
