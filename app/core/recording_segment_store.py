from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from app.database import Database, ensure_database


@dataclass(frozen=True, slots=True)
class RecordingSegment:
    segment_id: str
    camera_id: str
    bucket: str
    object_key: str
    started_at_utc: datetime
    ended_at_utc: datetime
    frame_width: int
    frame_height: int
    fps: float
    sha256: str


class IncompleteCoverageError(RuntimeError):
    """Requested interval is not yet continuously covered by closed segments."""


def _utc_datetime(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _segment_from_row(row: object) -> RecordingSegment:
    value = dict(row)
    value["started_at_utc"] = _utc_datetime(value["started_at_utc"])
    value["ended_at_utc"] = _utc_datetime(value["ended_at_utc"])
    return RecordingSegment(**value)


def match_covering_segments(
    segments: Iterable[RecordingSegment], start: datetime, end: datetime, *, max_segments: int
) -> tuple[RecordingSegment, ...]:
    """Pure half-open [start,end) overlap matcher with continuous coverage validation."""
    if end <= start or max_segments <= 0:
        raise ValueError("invalid requested interval or segment bound")
    overlapping = sorted(
        (s for s in segments if s.started_at_utc < end and s.ended_at_utc > start),
        key=lambda s: (s.started_at_utc, s.ended_at_utc, s.segment_id),
    )
    if not overlapping or len(overlapping) > max_segments:
        raise IncompleteCoverageError("recording coverage is absent or exceeds its bound")
    cursor = start
    gap_tolerance = timedelta(milliseconds=250)
    selected: list[RecordingSegment] = []
    for segment in overlapping:
        if segment.started_at_utc > cursor + gap_tolerance:
            raise IncompleteCoverageError("recording coverage contains a gap")
        if segment.ended_at_utc <= cursor:
            continue
        selected.append(segment)
        cursor = max(cursor, segment.ended_at_utc)
        if cursor >= end:
            return tuple(selected)
    raise IncompleteCoverageError("recording coverage is incomplete")


class RecordingSegmentStore:
    def __init__(self, database: Database | str) -> None:
        self.database = ensure_database(database)

    def upsert(self, segment: RecordingSegment) -> RecordingSegment:
        if (segment.frame_width <= 0 or segment.frame_height <= 0 or segment.fps <= 0
                or segment.ended_at_utc <= segment.started_at_utc):
            raise ValueError("segment media metadata and interval must be positive")
        with self.database.connection() as connection:
            connection.execute(
                """INSERT INTO recording_segments
                (segment_id,camera_id,bucket,object_key,started_at_utc,ended_at_utc,frame_width,frame_height,fps,sha256)
                VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING""",
                (segment.segment_id, segment.camera_id, segment.bucket, segment.object_key,
                 segment.started_at_utc, segment.ended_at_utc, segment.frame_width,
                 segment.frame_height, segment.fps, segment.sha256),
            )
            row = connection.execute(
                "SELECT segment_id,camera_id,bucket,object_key,started_at_utc,ended_at_utc,frame_width,frame_height,fps,sha256 FROM recording_segments WHERE segment_id=?",
                (segment.segment_id,),
            ).fetchone()
        if row is None:
            raise ValueError("segment object key conflicts with durable metadata")
        existing = _segment_from_row(row)
        if existing != segment:
            raise ValueError("segment id conflicts with durable metadata")
        return existing

    def mark_published(self, segment_id: str) -> None:
        with self.database.connection() as connection:
            connection.execute("UPDATE recording_segments SET published_at_utc=? WHERE segment_id=?", (datetime.now(timezone.utc), segment_id))

    def match(self, camera_id: str, start: datetime, end: datetime, *, max_segments: int) -> tuple[RecordingSegment, ...]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """SELECT segment_id,camera_id,bucket,object_key,started_at_utc,ended_at_utc,frame_width,frame_height,fps,sha256
                FROM recording_segments WHERE camera_id=? AND started_at_utc < ? AND ended_at_utc > ?
                ORDER BY started_at_utc,ended_at_utc,segment_id LIMIT ?""",
                (camera_id, end, start, max_segments + 1),
            ).fetchall()
        return match_covering_segments((_segment_from_row(row) for row in rows), start, end, max_segments=max_segments)

    def unpublished(self, limit: int = 100) -> tuple[RecordingSegment, ...]:
        with self.database.connection() as connection:
            rows = connection.execute(
                "SELECT segment_id,camera_id,bucket,object_key,started_at_utc,ended_at_utc,frame_width,frame_height,fps,sha256 FROM recording_segments WHERE published_at_utc IS NULL ORDER BY created_at_utc LIMIT ?",
                (max(1, min(limit, 1000)),),
            ).fetchall()
        return tuple(_segment_from_row(row) for row in rows)
