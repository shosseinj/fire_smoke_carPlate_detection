"""Add audit user columns to registered car plates.

Revision ID: 20260725_0019
Revises: 20260725_0018
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260725_0019"
down_revision = "20260725_0018"
branch_labels = None
depends_on = None


def _column_names() -> set[str]:
    return {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns("car_plates")
    }


def upgrade() -> None:
    columns = _column_names()
    if "created_by" not in columns:
        op.add_column("car_plates", sa.Column("created_by", sa.Integer()))
    if "updated_by" not in columns:
        op.add_column("car_plates", sa.Column("updated_by", sa.Integer()))


def downgrade() -> None:
    columns = _column_names()
    if "updated_by" in columns:
        op.drop_column("car_plates", "updated_by")
    if "created_by" in columns:
        op.drop_column("car_plates", "created_by")
