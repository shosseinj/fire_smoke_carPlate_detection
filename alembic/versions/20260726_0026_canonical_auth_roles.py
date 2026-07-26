"""Canonicalize authentication roles.

Revision ID: 20260726_0026
Revises: 20260726_0025
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260726_0026"
down_revision = "20260726_0025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        sa.text(
            "UPDATE users SET role = 'superadmin' WHERE role = 'superuser'"
        )
    )
    op.execute(
        sa.text(
            "UPDATE users SET role = 'user' WHERE role IN ('operator', 'viewer')"
        )
    )
    op.execute(sa.text("ALTER TABLE users ALTER COLUMN role SET DEFAULT 'user'"))
    op.create_check_constraint(
        "ck_users_role",
        "users",
        "role IN ('superadmin', 'admin', 'user')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_users_role", "users", type_="check")
    op.execute(sa.text("ALTER TABLE users ALTER COLUMN role SET DEFAULT 'viewer'"))
    op.execute(sa.text("UPDATE users SET role = 'viewer' WHERE role = 'user'"))
    op.execute(sa.text("UPDATE users SET role = 'superuser' WHERE role = 'superadmin'"))
