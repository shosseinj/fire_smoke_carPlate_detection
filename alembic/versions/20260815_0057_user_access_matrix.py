"""Replace roles with direct per-user access grants.

Revision ID: 20260815_0057
Revises: 20260811_0056
"""

from alembic import op
import sqlalchemy as sa


revision = "20260815_0057"
down_revision = "20260811_0056"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_permission_grants",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("application", sa.String(100), nullable=False),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("scope_type", sa.String(20), nullable=False),
        sa.Column("scope_id", sa.Integer(), server_default="0", nullable=False),
        sa.Column("assigned_by", sa.Integer(), nullable=True),
        sa.Column("created_at_utc", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.CheckConstraint("(scope_type = 'global' AND scope_id = 0) OR (scope_type IN ('building', 'section', 'camera') AND scope_id > 0)", name="ck_user_permission_grants_target"),
        sa.ForeignKeyConstraint(["assigned_by"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "application", "action", "scope_type", "scope_id", name="uq_user_permission_grant"),
    )
    op.create_index("idx_user_permission_grants_lookup", "user_permission_grants", ["user_id", "application", "action"])
    bind = op.get_bind()
    supported = {
        "app.read", "app.manage", "app.system", "auth.users.manage", "auth.roles.manage",
        *(f"{resource}.{action}" for resource in ("buildings", "sections", "cameras") for action in ("create", "read", "edit", "delete")),
    }
    assigned_keys = {
        str(row[0])
        for row in bind.execute(sa.text(
            "SELECT DISTINCT p.key FROM role_permissions rp JOIN permissions p ON p.id = rp.permission_id"
        ))
    }
    unsupported = assigned_keys - supported
    if unsupported:
        raise RuntimeError(
            "مهاجرت دسترسی متوقف شد؛ مجوزهای تعریف‌نشده در ماتریس وجود دارد: "
            + ", ".join(sorted(unsupported))
        )
    bind.execute(sa.text("""
        INSERT INTO user_permission_grants (user_id, application, action, scope_type, scope_id, assigned_by)
        SELECT DISTINCT ur.user_id,
          CASE split_part(p.key, '.', 1)
            WHEN 'app' THEN 'application' WHEN 'auth' THEN 'auth'
            ELSE split_part(p.key, '.', 1) END,
          CASE WHEN split_part(p.key, '.', 1) = 'auth' THEN 'manage' ELSE split_part(p.key, '.', 2) END,
          'global', 0, ur.assigned_by
        FROM user_roles ur
        JOIN role_permissions rp ON rp.role_id = ur.role_id
        JOIN permissions p ON p.id = rp.permission_id
        ON CONFLICT DO NOTHING
    """))
    bind.execute(sa.text("""
        INSERT INTO user_permission_grants (user_id, application, action, scope_type, scope_id, assigned_by)
        SELECT DISTINCT ur.user_id, split_part(p.key, '.', 1), split_part(p.key, '.', 2),
          s.scope_type, s.scope_id, COALESCE(s.assigned_by, ur.assigned_by)
        FROM user_roles ur JOIN role_permission_scopes s ON s.role_id = ur.role_id
        JOIN permissions p ON p.id = s.permission_id
        ON CONFLICT DO NOTHING
    """))
    # Preserve broad legacy personnel behavior as explicit matrix cells.
    for source_action, target_actions in (("read", ("read",)), ("manage", ("create", "read", "edit")), ("system", ("delete",))):
        for target_action in target_actions:
            bind.execute(sa.text("""
                INSERT INTO user_permission_grants (user_id, application, action, scope_type, scope_id)
                SELECT DISTINCT user_id, 'personnel', :target_action, 'global', 0
                FROM user_permission_grants WHERE application = 'application' AND action = :source_action
                ON CONFLICT DO NOTHING
            """), {"source_action": source_action, "target_action": target_action})
    missing_global = bind.execute(sa.text("""
        SELECT COUNT(*) FROM (
          SELECT DISTINCT ur.user_id,
            CASE split_part(p.key, '.', 1) WHEN 'app' THEN 'application' WHEN 'auth' THEN 'auth' ELSE split_part(p.key, '.', 1) END AS application,
            CASE WHEN split_part(p.key, '.', 1) = 'auth' THEN 'manage' ELSE split_part(p.key, '.', 2) END AS action
          FROM user_roles ur JOIN role_permissions rp ON rp.role_id = ur.role_id JOIN permissions p ON p.id = rp.permission_id
        ) source
        LEFT JOIN user_permission_grants target ON target.user_id = source.user_id
          AND target.application = source.application AND target.action = source.action
          AND target.scope_type = 'global' AND target.scope_id = 0
        WHERE target.id IS NULL
    """)).scalar_one()
    if missing_global:
        raise RuntimeError("راستی‌آزمایی مهاجرت دسترسی‌های سراسری ناموفق بود")
    op.drop_table("role_permission_scopes")
    op.drop_table("user_roles")
    op.drop_table("role_permissions")
    op.drop_table("permissions")
    op.drop_table("roles")


def downgrade() -> None:
    raise RuntimeError("بازگشت خودکار از ماتریس دسترسی پشتیبانی نمی‌شود؛ از نسخه پشتیبان پایگاه داده استفاده کنید")
