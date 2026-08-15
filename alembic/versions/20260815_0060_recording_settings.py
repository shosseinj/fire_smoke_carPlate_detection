"""Add configurable recording policies and output metadata."""

from alembic import op
import sqlalchemy as sa

revision = "20260815_0060"
down_revision = "20260815_0059"
branch_labels = None
depends_on = None


def upgrade() -> None:
    checks = (
        sa.CheckConstraint("quality_preset IN ('low','medium','high','original')", name="ck_recording_settings_quality"),
        sa.CheckConstraint("segment_seconds IN (60,120,300,900)", name="ck_recording_settings_segment"),
        sa.CheckConstraint("retention_days IN (7,30,90,365)", name="ck_recording_settings_retention"),
    )
    op.create_table(
        "recording_settings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("continuous_enabled", sa.Boolean(), nullable=False, server_default=sa.text("FALSE")),
        sa.Column("quality_preset", sa.String(16), nullable=False, server_default="medium"),
        sa.Column("segment_seconds", sa.Integer(), nullable=False, server_default="120"),
        sa.Column("retention_days", sa.Integer(), nullable=False, server_default="30"),
        sa.Column("created_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("updated_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("created_at_utc", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at_utc", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint("id = 1", name="ck_recording_settings_singleton"), *checks,
    )
    op.create_table(
        "recording_camera_settings",
        sa.Column("source_uri", sa.Text(), sa.ForeignKey("sources.source_uri", ondelete="CASCADE"), primary_key=True),
        sa.Column("continuous_enabled", sa.Boolean(), nullable=False),
        sa.Column("quality_preset", sa.String(16), nullable=False),
        sa.Column("segment_seconds", sa.Integer(), nullable=False),
        sa.Column("retention_days", sa.Integer(), nullable=False),
        sa.Column("created_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("updated_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("created_at_utc", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at_utc", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint("quality_preset IN ('low','medium','high','original')", name="ck_recording_camera_settings_quality"),
        sa.CheckConstraint("segment_seconds IN (60,120,300,900)", name="ck_recording_camera_settings_segment"),
        sa.CheckConstraint("retention_days IN (7,30,90,365)", name="ck_recording_camera_settings_retention"),
    )
    for name, type_, default, nullable in (
        ("quality_preset", sa.String(16), "medium", False), ("retention_days", sa.Integer(), "30", False),
        ("output_width", sa.Integer(), None, True), ("output_height", sa.Integer(), None, True),
        ("output_fps", sa.Integer(), None, True), ("output_bitrate_bps", sa.Integer(), None, True),
    ):
        op.add_column("recording_jobs", sa.Column(name, type_, nullable=nullable, server_default=default))
    op.execute("INSERT INTO recording_settings (id) VALUES (1)")
    op.execute("""
        INSERT INTO user_permission_grants (user_id, application, action, scope_type, scope_id, assigned_by)
        SELECT DISTINCT user_id, 'recording_settings', action, 'global', 0, assigned_by
        FROM user_permission_grants WHERE application = 'general_settings' AND action IN ('read','edit','reset')
        ON CONFLICT DO NOTHING
    """)


def downgrade() -> None:
    op.execute("DELETE FROM user_permission_grants WHERE application = 'recording_settings'")
    for name in ("output_bitrate_bps", "output_fps", "output_height", "output_width", "retention_days", "quality_preset"):
        op.drop_column("recording_jobs", name)
    op.drop_table("recording_camera_settings")
    op.drop_table("recording_settings")
