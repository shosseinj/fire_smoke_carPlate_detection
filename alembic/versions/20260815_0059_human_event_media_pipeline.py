"""human event media persistence

Revision ID: 20260815_0059
Revises: 20260815_0058
"""
from alembic import op
import sqlalchemy as sa

revision = "20260815_0059"
down_revision = "20260815_0058"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("human_logs", sa.Column("event_id", sa.String(255), nullable=True))
    op.create_unique_constraint("uq_human_logs_event_id", "human_logs", ["event_id"])
    op.create_table(
        "recording_segments",
        sa.Column("segment_id", sa.String(64), primary_key=True),
        sa.Column("camera_id", sa.Text(), nullable=False),
        sa.Column("bucket", sa.Text(), nullable=False),
        sa.Column("object_key", sa.Text(), nullable=False, unique=True),
        sa.Column("started_at_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("frame_width", sa.Integer(), nullable=False),
        sa.Column("frame_height", sa.Integer(), nullable=False),
        sa.Column("fps", sa.Float(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("published_at_utc", sa.DateTime(timezone=True)),
        sa.Column("created_at_utc", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint("ended_at_utc > started_at_utc", name="ck_recording_segments_interval"),
        sa.CheckConstraint("frame_width > 0 AND frame_height > 0 AND fps > 0", name="ck_recording_segments_media"),
    )
    op.create_index("idx_recording_segments_camera_interval", "recording_segments", ["camera_id", "started_at_utc", "ended_at_utc"])
    op.create_table(
        "detection_event_outbox",
        sa.Column("event_id", sa.String(255), primary_key=True),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("stream", sa.Text(), nullable=False), sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("published_at_utc", sa.DateTime(timezone=True)),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text()),
        sa.Column("created_at_utc", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("idx_detection_event_outbox_pending", "detection_event_outbox", ["created_at_utc"],
                    postgresql_where=sa.text("published_at_utc IS NULL"))


def downgrade() -> None:
    op.drop_table("detection_event_outbox")
    op.drop_table("recording_segments")
    op.drop_constraint("uq_human_logs_event_id", "human_logs", type_="unique")
    op.drop_column("human_logs", "event_id")
