"""Canonical detection media keys, thumbnails, and video readiness.

Revision ID: 20260728_0039
Revises: 20260728_0038
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260728_0039"
down_revision = "20260728_0038"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("detection_logs", sa.Column("face_thumbnail", sa.Text(), nullable=True))
    op.add_column(
        "detection_logs",
        sa.Column("video_status", sa.String(length=16), nullable=False, server_default="missing"),
    )
    op.add_column(
        "detection_logs",
        sa.Column("face_video_status", sa.String(length=16), nullable=False, server_default="missing"),
    )
    op.add_column(
        "detection_logs",
        sa.Column("media_finalized_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        "ck_detection_logs_video_status",
        "detection_logs",
        "video_status IN ('missing', 'writing', 'ready', 'failed')",
    )
    op.create_check_constraint(
        "ck_detection_logs_face_video_status",
        "detection_logs",
        "face_video_status IN ('missing', 'writing', 'ready', 'failed')",
    )

    # Convert historical public URLs into root-relative storage keys.
    for column in (
        "face_image",
        "body_image",
        "snapshot_image",
        "video",
        "face_video_or_unknown_faces",
    ):
        op.execute(
            sa.text(
                f"UPDATE detection_logs SET {column} = substring({column} from 8) "
                f"WHERE {column} LIKE '/media/%'"
            )
        )

    # human_logs is an internal source table, but canonicalize it too so newly
    # generated detection logs and old human records use the same convention.
    for column in ("snapshot_url", "video_url", "face_video_url"):
        op.execute(
            sa.text(
                f"UPDATE human_logs SET {column} = substring({column} from 8) "
                f"WHERE {column} LIKE '/media/%'"
            )
        )

    op.execute(
        "UPDATE detection_logs SET video_status = 'ready' "
        "WHERE video IS NOT NULL AND btrim(video) <> ''"
    )
    op.execute(
        "UPDATE detection_logs SET face_video_status = 'ready' "
        "WHERE face_video_or_unknown_faces IS NOT NULL "
        "AND btrim(face_video_or_unknown_faces) <> ''"
    )
    op.execute(
        "UPDATE detection_logs SET media_finalized_at = updated_at_utc "
        "WHERE video_status = 'ready' OR face_video_status = 'ready'"
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_detection_logs_face_video_status", "detection_logs", type_="check"
    )
    op.drop_constraint(
        "ck_detection_logs_video_status", "detection_logs", type_="check"
    )
    op.drop_column("detection_logs", "media_finalized_at")
    op.drop_column("detection_logs", "face_video_status")
    op.drop_column("detection_logs", "video_status")
    op.drop_column("detection_logs", "face_thumbnail")
