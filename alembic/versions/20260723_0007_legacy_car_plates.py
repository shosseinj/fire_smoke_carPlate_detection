"""Add the legacy registered-car-plate compatibility resource."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260723_0007"
down_revision = "20260722_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("""
        CREATE TABLE car_plates (
            id SERIAL PRIMARY KEY,
            left_digits VARCHAR(2) NOT NULL,
            plate_alphabet VARCHAR(1) NOT NULL,
            right_digits VARCHAR(3) NOT NULL,
            iran_code VARCHAR(2) NOT NULL,
            plate_format VARCHAR(32) NOT NULL DEFAULT 'standard',
            usage_type VARCHAR(32) NOT NULL,
            vehicle_type VARCHAR(32) NOT NULL,
            owner_name VARCHAR(200) NOT NULL,
            owner_phone VARCHAR(32) NOT NULL,
            color VARCHAR(50), brand VARCHAR(80), model VARCHAR(80),
            manufacture_year INTEGER, description TEXT,
            is_active INTEGER NOT NULL DEFAULT 1,
            deleted_at_utc TIMESTAMP WITH TIME ZONE,
            created_at_utc TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at_utc TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_car_plates_components UNIQUE (left_digits, plate_alphabet, right_digits, iran_code)
        )
    """))
    op.execute(sa.text("CREATE INDEX idx_car_plates_owner_phone ON car_plates (owner_phone)"))
    op.execute(sa.text("CREATE INDEX idx_car_plates_active ON car_plates (is_active)"))


def downgrade() -> None:
    op.execute(sa.text("DROP TABLE IF EXISTS car_plates CASCADE"))
