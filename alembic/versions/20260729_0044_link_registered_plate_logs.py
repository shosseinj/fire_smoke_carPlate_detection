"""Link recognized plate logs to registered car plates.

Revision ID: 20260729_0044
Revises: 20260729_0043
"""
from __future__ import annotations

from alembic import op


revision = "20260729_0044"
down_revision = "20260729_0043"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Clear legacy orphan values before enforcing referential integrity.
    op.execute(
        "UPDATE plate_logs p SET plate_id = NULL "
        "WHERE plate_id IS NOT NULL "
        "AND NOT EXISTS (SELECT 1 FROM car_plates c WHERE c.id = p.plate_id)"
    )
    # Backfill only exact normalized matches to active, non-deleted registrations.
    op.execute(
        """
        UPDATE plate_logs p
        SET plate_id = (
            SELECT c.id
            FROM car_plates c
            WHERE c.deleted_at_utc IS NULL
              AND c.is_active = 1
              AND CONCAT(
                    c.left_digits,
                    c.plate_alphabet,
                    c.right_digits,
                    c.iran_code
                  ) = p.plate_number
            ORDER BY c.id
            LIMIT 1
        )
        WHERE p.plate_id IS NULL
          AND EXISTS (
            SELECT 1
            FROM car_plates c
            WHERE c.deleted_at_utc IS NULL
              AND c.is_active = 1
              AND CONCAT(
                    c.left_digits,
                    c.plate_alphabet,
                    c.right_digits,
                    c.iran_code
                  ) = p.plate_number
          )
        """
    )
    op.create_foreign_key(
        "fk_plate_logs_plate_id_car_plates",
        "plate_logs",
        "car_plates",
        ["plate_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_plate_logs_plate_id_car_plates", "plate_logs", type_="foreignkey"
    )
