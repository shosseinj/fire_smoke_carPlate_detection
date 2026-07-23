"""Add import_progress table, force column, and face threshold constraint."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260723_0008"
down_revision = "20260723_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("""
        CREATE TABLE import_progress (
            id SERIAL PRIMARY KEY,
            import_type TEXT NOT NULL,
            source_filename TEXT NOT NULL,
            total_rows INTEGER NOT NULL DEFAULT 0,
            imported_rows INTEGER NOT NULL DEFAULT 0,
            skipped_rows INTEGER NOT NULL DEFAULT 0,
            failed_rows INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'running',
            error_message TEXT,
            created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
            created_at_utc TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at_utc TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """))
    op.execute(sa.text("CREATE INDEX idx_import_progress_type ON import_progress (import_type)"))
    op.execute(sa.text("CREATE INDEX idx_import_progress_status ON import_progress (status)"))
    op.execute(sa.text("CREATE INDEX idx_import_progress_created_by ON import_progress (created_by)"))

    op.execute(sa.text("""
        ALTER TABLE general_settings
        ADD COLUMN force INTEGER NOT NULL DEFAULT 0
    """))
    op.execute(sa.text("""
        ALTER TABLE general_settings
        ADD CONSTRAINT ck_general_settings_force CHECK (force IN (0, 1))
    """))
    op.execute(sa.text("""
        ALTER TABLE general_settings
        ADD CONSTRAINT ck_confirmation_ge_face_rec
        CHECK (confirmation_threshold >= face_rec_score)
    """))


def downgrade() -> None:
    op.execute(sa.text("DROP TABLE IF EXISTS import_progress CASCADE"))
    op.execute(sa.text("ALTER TABLE general_settings DROP CONSTRAINT IF EXISTS ck_confirmation_ge_face_rec"))
    op.execute(sa.text("ALTER TABLE general_settings DROP CONSTRAINT IF EXISTS ck_general_settings_force"))
    op.execute(sa.text("ALTER TABLE general_settings DROP COLUMN IF EXISTS force"))
