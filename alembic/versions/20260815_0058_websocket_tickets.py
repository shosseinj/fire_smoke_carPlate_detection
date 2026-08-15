"""Add single-use WebSocket tickets and live-channel grants.

Revision ID: 20260815_0058
Revises: 20260815_0057
"""

from alembic import op
import sqlalchemy as sa


revision = "20260815_0058"
down_revision = "20260815_0057"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "websocket_tickets",
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("application", sa.String(100), nullable=False),
        sa.Column("scope_type", sa.String(20), nullable=False),
        sa.Column("scope_id", sa.Integer(), server_default="0", nullable=False),
        sa.Column("expires_at_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at_utc", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at_utc", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("token_hash"),
    )
    op.create_index("idx_websocket_tickets_expiry", "websocket_tickets", ["expires_at_utc"])
    bind = op.get_bind()
    for application in ("broadcast", "video_wall", "results"):
        bind.execute(sa.text("""
            INSERT INTO user_permission_grants (user_id, application, action, scope_type, scope_id)
            SELECT DISTINCT user_id, :application, 'read', 'global', 0
            FROM user_permission_grants
            WHERE application = 'application' AND action IN ('read', 'manage', 'system')
            ON CONFLICT DO NOTHING
        """), {"application": application})


def downgrade() -> None:
    op.execute("DELETE FROM user_permission_grants WHERE application IN ('broadcast', 'video_wall', 'results')")
    op.drop_index("idx_websocket_tickets_expiry", table_name="websocket_tickets")
    op.drop_table("websocket_tickets")
