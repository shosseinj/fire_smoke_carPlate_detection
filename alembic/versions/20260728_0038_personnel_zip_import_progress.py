"""Store final personnel ZIP import results.

Revision ID: 20260728_0038
Revises: 20260728_0037
"""

from alembic import op
import sqlalchemy as sa


revision = "20260728_0038"
down_revision = "20260728_0037"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("import_progress", sa.Column("result_json", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("import_progress", "result_json")
