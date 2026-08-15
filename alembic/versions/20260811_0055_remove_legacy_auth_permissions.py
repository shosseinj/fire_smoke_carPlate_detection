"""Replace legacy authorization permissions with current permission keys.

Revision ID: 20260811_0055
Revises: 20260811_0054
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260811_0055"
down_revision = "20260811_0054"
branch_labels = None
depends_on = None


_PERMISSION_RENAMES = (
    ("legacy.user", "app.read"),
    ("legacy.admin", "app.manage"),
    ("legacy.superadmin", "app.system"),
)


def _copy_grants(source_key: str, target_key: str) -> None:
    op.execute(
        sa.text(
            "INSERT INTO role_permissions (role_id, permission_id) "
            "SELECT rp.role_id, target.id "
            "FROM role_permissions rp "
            "JOIN permissions source ON source.id = rp.permission_id "
            "JOIN permissions target ON target.key = :target_key "
            "WHERE source.key = :source_key "
            "ON CONFLICT DO NOTHING"
        ).bindparams(source_key=source_key, target_key=target_key)
    )


def upgrade() -> None:
    for source_key, target_key in _PERMISSION_RENAMES:
        _copy_grants(source_key, target_key)
    op.execute(
        sa.text(
            "DELETE FROM permissions "
            "WHERE key IN ('legacy.user', 'legacy.admin', 'legacy.superadmin')"
        )
    )


def downgrade() -> None:
    for legacy_key, current_key in _PERMISSION_RENAMES:
        op.execute(
            sa.text(
                "INSERT INTO permissions (key, description, is_system) "
                "VALUES (:key, :description, TRUE) ON CONFLICT (key) DO NOTHING"
            ).bindparams(
                key=legacy_key,
                description=f"Compatibility permission restored from {current_key}",
            )
        )
        _copy_grants(current_key, legacy_key)
