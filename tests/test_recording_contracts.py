from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.api.recordings import RecordingCreate
from app.core.recording_source_ref import recording_source_ref
from app.core.recording_job_store import MAX_DURATION
from app.database import ALEMBIC_HEAD_REVISION, recording_jobs


def test_recording_schema_is_timezone_aware_and_duration_bounded() -> None:
    start = datetime.now(timezone.utc) + timedelta(minutes=1)
    source_ref = recording_source_ref("rtsp://camera/1")
    payload = RecordingCreate(source_ref=source_ref, scheduled_start_utc=start, scheduled_end_utc=start + MAX_DURATION)
    assert payload.scheduled_start_utc.tzinfo is not None

    with pytest.raises(ValidationError):
        RecordingCreate(source_ref=source_ref, scheduled_start_utc=start.replace(tzinfo=None), scheduled_end_utc=start + timedelta(minutes=1))
    with pytest.raises(ValidationError):
        RecordingCreate(source_ref=source_ref, scheduled_start_utc=start, scheduled_end_utc=start + MAX_DURATION + timedelta(seconds=1))
    with pytest.raises(ValidationError):
        RecordingCreate(source_uri="rtsp://camera.invalid/…", scheduled_start_utc=start, scheduled_end_utc=start + timedelta(minutes=1))


def test_database_metadata_exposes_authoritative_recording_table() -> None:
    assert ALEMBIC_HEAD_REVISION == "20260803_0049"
    assert recording_jobs.c.scheduled_start_utc.type.timezone is True
    assert recording_jobs.c.scheduled_end_utc.type.timezone is True
    assert {constraint.name for constraint in recording_jobs.constraints} >= {
        "ck_recording_jobs_positive_duration",
        "ck_recording_jobs_max_duration",
        "ck_recording_jobs_status",
        "uq_recording_jobs_actor_idempotency",
    }
