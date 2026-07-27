"""Make sources.fps the only runtime FPS control.

Revision ID: 20260727_0029
Revises: 20260727_0028
"""
from __future__ import annotations

from alembic import op


revision = "20260727_0029"
down_revision = "20260727_0028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE general_settings
        SET operational_json = (
            (
                COALESCE(NULLIF(operational_json, ''), '{}')::jsonb
                - 'video_ingest_fps'
                - 'video_preview_fps'
            )
            || jsonb_build_object(
                'rtsp_source_count',
                GREATEST(
                    COALESCE(
                        (
                            COALESCE(NULLIF(operational_json, ''), '{}')::jsonb
                            ->> 'rtsp_source_count'
                        )::integer,
                        0
                    ),
                    256
                ),
                'static_video_source_count',
                GREATEST(
                    COALESCE(
                        (
                            COALESCE(NULLIF(operational_json, ''), '{}')::jsonb
                            ->> 'static_video_source_count'
                        )::integer,
                        0
                    ),
                    256
                )
            )
        )::text
        WHERE id = 1
        """
    )
    op.execute(
        """
        UPDATE sources
        SET metadata_json = jsonb_set(
            COALESCE(NULLIF(metadata_json, ''), '{}')::jsonb,
            '{_settings_overrides}',
            (
                COALESCE(NULLIF(metadata_json, ''), '{}')::jsonb
                -> '_settings_overrides'
            )
            - 'video_ingest_fps'
            - 'video_preview_fps',
            true
        )::text
        WHERE jsonb_typeof(
            COALESCE(NULLIF(metadata_json, ''), '{}')::jsonb
            -> '_settings_overrides'
        ) = 'object'
        """
    )


def downgrade() -> None:
    return None
