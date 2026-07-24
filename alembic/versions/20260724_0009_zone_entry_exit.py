"""Add transition_type and track_id to detection_room_matches for zone entry/exit tracking."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260724_0009"
down_revision = "20260723_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("""
        ALTER TABLE detection_room_matches
        ADD COLUMN track_id INTEGER
    """))
    op.execute(sa.text("""
        ALTER TABLE detection_room_matches
        ADD COLUMN transition_type TEXT
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_matches_track ON detection_room_matches (track_id)
    """))
    op.execute(sa.text("""
        CREATE INDEX idx_matches_transition ON detection_room_matches (transition_type)
    """))


def downgrade() -> None:
    op.execute(sa.text("DROP INDEX IF EXISTS idx_matches_transition"))
    op.execute(sa.text("DROP INDEX IF EXISTS idx_matches_track"))
    op.execute(sa.text("ALTER TABLE detection_room_matches DROP COLUMN IF EXISTS transition_type"))
    op.execute(sa.text("ALTER TABLE detection_room_matches DROP COLUMN IF EXISTS track_id"))
