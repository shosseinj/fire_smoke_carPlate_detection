"""Add database-driven roles and permissions.

Revision ID: 20260811_0054
Revises: 20260810_0053
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260811_0054"
down_revision = "20260810_0053"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("auth_version", sa.Integer(), server_default="0", nullable=False),
    )
    op.create_table(
        "roles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("display_name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("is_system", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("TRUE"), nullable=False),
        sa.Column("created_at_utc", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at_utc", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "permissions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("key", sa.String(length=150), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("is_system", sa.Boolean(), server_default=sa.text("TRUE"), nullable=False),
        sa.Column("created_at_utc", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at_utc", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key"),
    )
    op.create_table(
        "role_permissions",
        sa.Column("role_id", sa.Integer(), nullable=False),
        sa.Column("permission_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["permission_id"], ["permissions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("role_id", "permission_id"),
    )
    op.create_table(
        "user_roles",
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("role_id", sa.Integer(), nullable=False),
        sa.Column("is_primary", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
        sa.Column("assigned_at_utc", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("assigned_by", sa.Integer()),
        sa.ForeignKeyConstraint(["assigned_by"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id", "role_id"),
    )
    op.create_index("idx_user_roles_role_id", "user_roles", ["role_id"])
    op.create_table(
        "auth_audit_log",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("actor_user_id", sa.Integer()),
        sa.Column("action", sa.String(length=100), nullable=False),
        sa.Column("target_type", sa.String(length=50), nullable=False),
        sa.Column("target_id", sa.String(length=150), nullable=False),
        sa.Column("details", sa.Text()),
        sa.Column("created_at_utc", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["actor_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_auth_audit_created", "auth_audit_log", ["created_at_utc"])

    op.execute(sa.text("""
        INSERT INTO roles (name, display_name, description, is_system)
        VALUES
          ('superadmin', 'Super administrator', 'Protected recovery role', TRUE),
          ('admin', 'Administrator', 'Default administrative role', TRUE),
          ('user', 'User', 'Default authenticated-user role', TRUE)
    """))
    op.execute(sa.text("""
        INSERT INTO permissions (key, description, is_system)
        VALUES
          ('app.read', 'Use authenticated read operations', TRUE),
          ('app.manage', 'Use administrative write operations', TRUE),
          ('app.system', 'Use protected system operations', TRUE),
          ('auth.users.manage', 'Create, update, assign roles to, and delete users', TRUE),
          ('auth.roles.manage', 'Create and configure roles and permissions', TRUE)
    """))
    op.execute(sa.text("""
        INSERT INTO role_permissions (role_id, permission_id)
        SELECT r.id, p.id
        FROM roles r CROSS JOIN permissions p
        WHERE (r.name = 'user' AND p.key = 'app.read')
           OR (r.name = 'admin' AND p.key IN ('app.read', 'app.manage', 'auth.users.manage'))
           OR (r.name = 'superadmin')
    """))
    op.execute(sa.text("""
        INSERT INTO user_roles (user_id, role_id, is_primary)
        SELECT u.id, r.id, TRUE
        FROM users u JOIN roles r ON r.name = u.role
    """))
    op.drop_constraint("ck_users_role", "users", type_="check")
    op.drop_column("users", "role")


def downgrade() -> None:
    op.add_column(
        "users",
        sa.Column("role", sa.Text(), server_default="user", nullable=False),
    )
    op.execute(sa.text("""
        UPDATE users u SET role = r.name
        FROM user_roles ur JOIN roles r ON r.id = ur.role_id
        WHERE ur.user_id = u.id AND ur.is_primary = TRUE
          AND r.name IN ('superadmin', 'admin', 'user')
    """))
    op.create_check_constraint(
        "ck_users_role", "users", "role IN ('superadmin', 'admin', 'user')"
    )
    op.drop_index("idx_auth_audit_created", table_name="auth_audit_log")
    op.drop_table("auth_audit_log")
    op.drop_index("idx_user_roles_role_id", table_name="user_roles")
    op.drop_table("user_roles")
    op.drop_table("role_permissions")
    op.drop_table("permissions")
    op.drop_table("roles")
    op.drop_column("users", "auth_version")
