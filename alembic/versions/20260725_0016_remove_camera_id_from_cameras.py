"""Remove camera_id from cameras table, use source_uri as PK.

Migration steps for PostgreSQL:
1. Drop FK ``detection_logs_camera_id_fkey`` (references old ``cameras(camera_id)``).
2. Backfill NULL ``source_uri`` from ``camera_id``.
3. Deduplicate ``source_uri`` — append ``__<camera_id>`` suffix to resolve collisions.
4. Make ``source_uri`` NOT NULL.
5. Drop the PK constraint on ``camera_id``.
6. Drop the ``camera_id`` column.
7. Add PK constraint on ``source_uri``.

The FK on ``detection_logs`` is not recreated because ``detection_logs.camera_id``
stores short IDs (e.g. ``camera_01``) which do not match the ``source_uri`` values
(RTSP URLs). The column remains as a plain text reference.

Revision ID: 20260725_0016
Revises: 20260725_0015
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260725_0016"
down_revision = "20260725_0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Drop FK that references cameras(camera_id)
    op.drop_constraint(
        "detection_logs_camera_id_fkey",
        "detection_logs",
        type_="foreignkey",
    )
    # 2. Backfill any NULL source_uri from camera_id
    op.execute(
        "UPDATE cameras SET source_uri = camera_id WHERE source_uri IS NULL"
    )
    # 3. Deduplicate source_uri — append __<camera_id> to duplicates
    op.execute(
        """
        UPDATE cameras
        SET source_uri = source_uri || '__' || camera_id
        FROM (
            SELECT ctid
            FROM (
                SELECT ctid, camera_id,
                       row_number() OVER (PARTITION BY source_uri ORDER BY camera_id) AS rn
                FROM cameras
            ) dup
            WHERE dup.rn > 1
        ) dups
        WHERE cameras.ctid = dups.ctid
        """
    )
    # 4. Make source_uri NOT NULL
    op.alter_column("cameras", "source_uri", nullable=False)
    # 5. Drop old PK on camera_id
    op.drop_constraint("cameras_pkey", "cameras", type_="primary")
    # 6. Drop the camera_id column
    op.drop_column("cameras", "camera_id")
    # 7. Add PK on source_uri
    op.create_primary_key("cameras_pkey", "cameras", ["source_uri"])


def downgrade() -> None:
    # 1. Re-add camera_id column
    op.add_column(
        "cameras",
        sa.Column("camera_id", sa.Text(), nullable=True),
    )
    # 2. Populate camera_id from source_uri
    op.execute(
        "UPDATE cameras SET camera_id = source_uri"
    )
    # 3. Drop PK on source_uri
    op.drop_constraint("cameras_pkey", "cameras", type_="primary")
    # 4. Make camera_id NOT NULL and re-add PK
    op.alter_column("cameras", "camera_id", nullable=False)
    op.create_primary_key("cameras_pkey", "cameras", ["camera_id"])
    # 5. Make source_uri nullable again
    op.alter_column("cameras", "source_uri", nullable=True)
    # 6. NOTE: The FK ``detection_logs_camera_id_fkey`` is NOT recreated here
    # because ``detection_logs.camera_id`` contains short IDs (e.g. ``camera_01``)
    # while ``cameras.camera_id`` was populated from ``source_uri`` (RTSP URLs)
    # during the downgrade, so the values no longer match. The FK was a loose
    # reference on a historical/audit table and is safe to leave removed.
