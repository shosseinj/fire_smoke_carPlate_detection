"""Preserve raw OCR text for invalid plate formats.

Revision ID: 20260729_0045
Revises: 20260729_0044
"""
from __future__ import annotations

from alembic import op


revision = "20260729_0045"
down_revision = "20260729_0044"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("plate_logs", "plate_number", nullable=True)
    op.create_check_constraint(
        "ck_plate_logs_has_ocr_text",
        "plate_logs",
        "plate_number IS NOT NULL OR raw_plate_text IS NOT NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_plate_logs_has_ocr_text", "plate_logs", type_="check"
    )
    op.execute("DELETE FROM plate_logs WHERE plate_number IS NULL")
    op.alter_column("plate_logs", "plate_number", nullable=False)
