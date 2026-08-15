"""Add hierarchical building, section and camera permission scopes.

Revision ID: 20260811_0056
Revises: 20260811_0055
"""

from alembic import op
import sqlalchemy as sa


revision = "20260811_0056"
down_revision = "20260811_0055"
branch_labels = None
depends_on = None


CRUD_PERMISSIONS = tuple(
    (f"{resource}.{action}", f"{action.title()} {resource}")
    for resource in ("buildings", "sections", "cameras")
    for action in ("create", "read", "edit", "delete")
)


def upgrade() -> None:
    op.create_table(
        "role_permission_scopes",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("role_id", sa.Integer(), nullable=False),
        sa.Column("permission_id", sa.Integer(), nullable=False),
        sa.Column("scope_type", sa.String(length=20), nullable=False),
        sa.Column("scope_id", sa.Integer(), server_default="0", nullable=False),
        sa.Column("assigned_by", sa.Integer(), nullable=True),
        sa.Column("created_at_utc", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.CheckConstraint(
            "(scope_type = 'global' AND scope_id = 0) OR "
            "(scope_type IN ('building', 'section', 'camera') AND scope_id > 0)",
            name="ck_role_permission_scopes_target",
        ),
        sa.ForeignKeyConstraint(["assigned_by"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["permission_id"], ["permissions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("role_id", "permission_id", "scope_type", "scope_id", name="uq_role_permission_scopes_grant"),
    )
    op.create_index("idx_role_permission_scopes_lookup", "role_permission_scopes", ["permission_id", "scope_type", "scope_id"])

    connection = op.get_bind()
    for key, description in CRUD_PERMISSIONS:
        connection.execute(
            sa.text(
                "INSERT INTO permissions (key, description, is_system) "
                "VALUES (:key, :description, TRUE) ON CONFLICT (key) DO NOTHING"
            ),
            {"key": key, "description": description},
        )

    # Preserve current behavior while making future assignments scope-aware.
    for role, actions in (("user", ("read",)), ("admin", ("create", "read", "edit", "delete"))):
        for resource in ("buildings", "sections", "cameras"):
            for action in actions:
                connection.execute(
                    sa.text(
                        "INSERT INTO role_permission_scopes "
                        "(role_id, permission_id, scope_type, scope_id) "
                        "SELECT r.id, p.id, 'global', 0 FROM roles r, permissions p "
                        "WHERE r.name = :role AND p.key = :key ON CONFLICT DO NOTHING"
                    ),
                    {"role": role, "key": f"{resource}.{action}"},
                )


def downgrade() -> None:
    op.drop_index("idx_role_permission_scopes_lookup", table_name="role_permission_scopes")
    op.drop_table("role_permission_scopes")
    op.execute(
        "DELETE FROM permissions WHERE key IN ("
        + ", ".join(f"'{key}'" for key, _ in CRUD_PERMISSIONS)
        + ")"
    )
