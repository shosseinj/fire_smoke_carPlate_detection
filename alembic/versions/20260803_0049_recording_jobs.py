"""Add authoritative scheduled recording jobs.

Revision ID: 20260803_0049
Revises: 20260802_0048
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260803_0049"
down_revision = "20260802_0048"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gist")
    op.create_table(
        "recording_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("source_uri", sa.Text(), sa.ForeignKey("sources.source_uri", ondelete="RESTRICT"), nullable=False),
        sa.Column("created_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("idempotency_key", sa.String(255)),
        sa.Column("status", sa.String(32), nullable=False, server_default="scheduled"),
        sa.Column("scheduled_start_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("scheduled_end_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at_utc", sa.DateTime(timezone=True)),
        sa.Column("finished_at_utc", sa.DateTime(timezone=True)),
        sa.Column("cancel_requested_at_utc", sa.DateTime(timezone=True)),
        sa.Column("object_key", sa.Text()),
        sa.Column("content_type", sa.String(128)),
        sa.Column("size_bytes", sa.BigInteger()),
        sa.Column("spool_path", sa.Text()),
        sa.Column("warning", sa.Text()),
        sa.Column("error", sa.Text()),
        sa.Column("is_partial", sa.Boolean(), nullable=False, server_default=sa.text("FALSE")),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("object_expires_at_utc", sa.DateTime(timezone=True)),
        sa.Column("spool_expires_at_utc", sa.DateTime(timezone=True)),
        sa.Column("created_at_utc", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at_utc", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint("scheduled_end_utc > scheduled_start_utc", name="ck_recording_jobs_positive_duration"),
        sa.CheckConstraint("scheduled_end_utc <= scheduled_start_utc + INTERVAL '2 hours'", name="ck_recording_jobs_max_duration"),
        sa.CheckConstraint(
            "status IN ('scheduled','queued','recording','finalizing','uploading','completed','partial','failed','cancelled')",
            name="ck_recording_jobs_status",
        ),
        sa.UniqueConstraint("created_by", "idempotency_key", name="uq_recording_jobs_actor_idempotency"),
    )
    op.create_index("idx_recording_jobs_source_start", "recording_jobs", ["source_uri", "scheduled_start_utc"])
    op.create_index("idx_recording_jobs_status_start", "recording_jobs", ["status", "scheduled_start_utc"])
    op.execute(
        "ALTER TABLE recording_jobs ADD CONSTRAINT ex_recording_jobs_source_overlap "
        "EXCLUDE USING gist (source_uri WITH =, "
        "tstzrange(scheduled_start_utc, scheduled_end_utc, '[)') WITH &&) "
        "WHERE (status IN ('scheduled','queued','recording','finalizing','uploading'))"
    )


def downgrade() -> None:
    op.drop_table("recording_jobs")
