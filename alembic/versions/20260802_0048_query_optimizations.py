"""Add query-oriented indexes and normalized plate storage.

Revision ID: 20260802_0048
Revises: 20260802_0047
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260802_0048"
down_revision = "20260802_0047"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "car_plates",
        sa.Column(
            "normalized_plate",
            sa.Text(),
            sa.Computed(
                "left_digits || plate_alphabet || right_digits || iran_code",
                persisted=True,
            ),
            nullable=True,
        ),
    )
    op.create_index(
        "idx_car_plates_active_normalized",
        "car_plates",
        ["normalized_plate"],
        unique=False,
        postgresql_where=sa.text("deleted_at_utc IS NULL AND is_active = 1"),
    )

    op.create_index(
        "idx_detection_logs_source_human",
        "detection_logs",
        ["source_human_log_id"],
    )
    op.create_index(
        "idx_detection_logs_attendance_personnel_time",
        "detection_logs",
        ["personnel_id", "detection_time"],
        postgresql_where=sa.text(
            "counts_for_attendance = 1 AND personnel_id IS NOT NULL"
        ),
    )
    op.create_index(
        "idx_detection_logs_attendance_person_time",
        "detection_logs",
        ["person", "detection_time"],
        postgresql_where=sa.text("counts_for_attendance = 1"),
    )
    op.create_index(
        "idx_detection_logs_room_time",
        "detection_logs",
        ["room_id", "detection_time"],
    )
    op.create_index(
        "idx_detection_logs_camera_time",
        "detection_logs",
        ["camera_id", "detection_time"],
    )
    op.create_index(
        "idx_detection_logs_personnel_room_time",
        "detection_logs",
        ["personnel_id", "room_id", "detection_time"],
    )

    op.execute(sa.text(
        "CREATE INDEX idx_personnel_images_personnel_order "
        "ON personnel_images (personnel_id, is_primary DESC, uploaded_at_utc DESC)"
    ))
    op.execute(sa.text(
        "CREATE INDEX idx_matches_camera_track_room_time "
        "ON detection_room_matches "
        "(camera_id, track_id, room_id, matched_at_utc DESC)"
    ))
    op.execute(sa.text(
        "CREATE INDEX idx_matches_room_time "
        "ON detection_room_matches (room_id, matched_at_utc DESC)"
    ))
    op.create_index(
        "idx_human_logs_attendance_personnel_last_seen",
        "human_logs",
        ["personnel_id", "last_seen"],
        postgresql_where=sa.text(
            "counts_for_attendance = 1 AND personnel_id IS NOT NULL"
        ),
    )
    op.create_index(
        "idx_requests_person_status_dates",
        "personnel_requests",
        ["personnel_id", "status", "start_date", "end_date"],
    )


def downgrade() -> None:
    op.drop_index("idx_requests_person_status_dates", table_name="personnel_requests")
    op.drop_index(
        "idx_human_logs_attendance_personnel_last_seen", table_name="human_logs"
    )
    op.drop_index("idx_matches_room_time", table_name="detection_room_matches")
    op.drop_index(
        "idx_matches_camera_track_room_time", table_name="detection_room_matches"
    )
    op.drop_index(
        "idx_personnel_images_personnel_order", table_name="personnel_images"
    )
    op.drop_index("idx_detection_logs_camera_time", table_name="detection_logs")
    op.drop_index(
        "idx_detection_logs_personnel_room_time", table_name="detection_logs"
    )
    op.drop_index("idx_detection_logs_room_time", table_name="detection_logs")
    op.drop_index(
        "idx_detection_logs_attendance_person_time", table_name="detection_logs"
    )
    op.drop_index(
        "idx_detection_logs_attendance_personnel_time", table_name="detection_logs"
    )
    op.drop_index("idx_detection_logs_source_human", table_name="detection_logs")
    op.drop_index("idx_car_plates_active_normalized", table_name="car_plates")
    op.drop_column("car_plates", "normalized_plate")
