"""Add auto-increment integer id primary key to cameras table.

Steps:
1. Drop the existing PK constraint on ``source_uri``.
2. Add ``id`` as a SERIAL column.
3. Backfill existing rows with sequential IDs.
4. Make ``id`` NOT NULL and add PK constraint on ``id``.
5. Add a UNIQUE constraint on ``source_uri``.

Revision ID: 20260725_0017
Revises: 20260725_0016
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260725_0017"
down_revision = "20260725_0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Drop PK on source_uri
    op.drop_constraint("cameras_pkey", "cameras", type_="primary")

    # 2. Add id column (nullable first)
    op.add_column(
        "cameras",
        sa.Column("id", sa.Integer(), nullable=True),
    )

    # 3. Create owned sequence and backfill existing rows
    op.execute("CREATE SEQUENCE IF NOT EXISTS cameras_id_seq START WITH 1 OWNED BY cameras.id")
    op.execute("UPDATE cameras SET id = nextval('cameras_id_seq')")
    op.execute("ALTER TABLE cameras ALTER COLUMN id SET DEFAULT nextval('cameras_id_seq')")

    # 4. Make id NOT NULL and add PK
    op.alter_column("cameras", "id", nullable=False)
    op.create_primary_key("cameras_pkey", "cameras", ["id"])

    # 5. Add UNIQUE constraint on source_uri
    op.create_unique_constraint("uq_cameras_source_uri", "cameras", ["source_uri"])


def downgrade() -> None:
    # 1. Drop UNIQUE constraint on source_uri
    op.drop_constraint("uq_cameras_source_uri", "cameras", type_="unique")

    # 2. Drop PK on id
    op.drop_constraint("cameras_pkey", "cameras", type_="primary")

    # 3. Drop id column
    op.drop_column("cameras", "id")

    # 4. Re-add PK on source_uri
    op.create_primary_key("cameras_pkey", "cameras", ["source_uri"])

    # 5. Drop the sequence
    op.execute("DROP SEQUENCE IF EXISTS cameras_id_seq")
