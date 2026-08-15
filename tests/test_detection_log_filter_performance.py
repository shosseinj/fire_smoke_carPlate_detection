from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import app.api.detection_logs as api
import pytest
from fastapi import HTTPException
from app.core.detection_log_store import DetectionLogRecord


def _record(log_id: int) -> DetectionLogRecord:
    return DetectionLogRecord(
        id=log_id,
        source_system="test",
        source_event_key=None,
        source_human_log_id=None,
        personnel_id=11,
        person="0011223344",
        confidence=0.91,
        detection_time="2026-07-30T04:30:00+00:00",
        ref_img_id="ref-1",
        room_id=22,
        camera_id="camera-1",
        access_granted=True,
        counts_for_attendance=True,
        log_type="fake",
        import_source_parts=None,
        face_image="faces/person.jpg",
        face_thumbnail="faces/person-thumb.jpg",
        body_image=None,
        snapshot_image=None,
        video=None,
        face_video_or_unknown_faces=None,
        video_status="missing",
        face_video_status="missing",
        media_finalized_at=None,
        created_by=31,
        updated_by=32,
        created_at_utc="2026-07-30T04:30:01+00:00",
        updated_at_utc="2026-07-30T04:30:02+00:00",
    )


class _RowsConnection:
    def __init__(self) -> None:
        self.execute_count = 0

    def execute(self, _sql, ids):
        self.execute_count += 1
        return SimpleNamespace(
            fetchall=lambda: [
                {
                    "id": log_id,
                    "personnel_record_id": 11,
                    "fname": "Ali",
                    "lname": "Ahmadi",
                    "room_name": "Room 1",
                    "section_name": "Section 1",
                    "building_name": "Building 1",
                    "created_by_username": "creator",
                    "updated_by_username": "editor",
                }
                for log_id in ids
            ]
        )


def test_filter_enrichment_uses_one_query_independent_of_page_size(monkeypatch):
    connection = _RowsConnection()
    runtime = SimpleNamespace(
        database=SimpleNamespace(connection=lambda: connection),
    )
    monkeypatch.setattr(api, "get_runtime", lambda: runtime)

    one = api._filter_response_enrichment([_record(1)])
    assert connection.execute_count == 1
    assert one[1]["full_name"] == "Ali Ahmadi"

    connection.execute_count = 0
    many = api._filter_response_enrichment([_record(i) for i in range(1, 201)])
    assert connection.execute_count == 1
    assert len(many) == 200


class _Media:
    def __init__(self) -> None:
        self.thumbnail_calls = 0
        self.exists_calls: list[str | None] = []

    def exists(self, value):
        self.exists_calls.append(value)
        return value is not None

    def thumbnail_data_uri(self, thumbnail, face):
        self.thumbnail_calls += 1
        assert (thumbnail, face) == ("faces/person-thumb.jpg", "faces/person.jpg")
        return "data:image/jpeg;base64,thumb"


def test_enriched_response_preserves_fields_order_values_and_thumbnail_toggle(monkeypatch):
    media = _Media()
    runtime = SimpleNamespace(
        registry=SimpleNamespace(
            get=lambda source_uri: SimpleNamespace(name="Camera 1")
            if source_uri == "camera-1"
            else None
        )
    )
    monkeypatch.setattr(api, "get_runtime", lambda: runtime)
    monkeypatch.setattr(api, "get_detection_media_storage", lambda: media)
    monkeypatch.setattr(api, "_resolve_usernames", lambda _record: ("creator", "editor"))
    monkeypatch.setattr(api, "_resolve_personnel_name", lambda _personnel_id: "Ali Ahmadi")
    monkeypatch.setattr(
        api,
        "_resolve_names",
        lambda _room_id, _camera_id: (
            "Room 1",
            "Camera 1",
            "Section 1",
            "Building 1",
        ),
    )
    enrichment = {
        "created_by_username": "creator",
        "updated_by_username": "editor",
        "full_name": "Ali Ahmadi",
        "room_name": "Room 1",
        "section_name": "Section 1",
        "building_name": "Building 1",
    }

    legacy = api._build_response(_record(1), include_detail=True)
    enriched = api._build_response(
        _record(1), include_detail=True, enrichment=enrichment
    )
    assert list(enriched) == list(legacy)
    assert enriched == legacy

    media.thumbnail_calls = 0
    without_thumbnail = api._build_response(
        _record(1),
        include_detail=True,
        include_face_thumbnail=False,
        enrichment=enrichment,
    )
    assert without_thumbnail["face_thumbnail"] is None
    assert media.thumbnail_calls == 0

    with_thumbnail = api._build_response(
        _record(1),
        include_detail=True,
        include_face_thumbnail=True,
        enrichment=enrichment,
    )
    assert with_thumbnail["face_thumbnail"] == "data:image/jpeg;base64,thumb"
    assert media.thumbnail_calls == 1


def test_filter_media_availability_is_deduplicated_and_reused(monkeypatch):
    media = _Media()
    monkeypatch.setattr(api, "get_detection_media_storage", lambda: media)
    monkeypatch.setattr(
        api,
        "get_runtime",
        lambda: SimpleNamespace(registry=SimpleNamespace(get=lambda _source: None)),
    )
    records = [_record(1), replace(_record(2), body_image="body/person.jpg")]

    availability = api._filter_media_availability(records)

    assert sorted(media.exists_calls) == ["body/person.jpg", "faces/person.jpg"]
    assert availability == {
        "faces/person.jpg": True,
        "body/person.jpg": True,
    }

    media.exists_calls.clear()
    response = api._build_response(
        records[0],
        include_detail=True,
        include_face_thumbnail=False,
        enrichment={},
        media_availability=availability,
    )
    assert response["face_image_url"] == "/api/v1/logs/1/face"
    assert media.exists_calls == []

    missing = api._build_response(
        records[0],
        include_detail=True,
        include_face_thumbnail=False,
        enrichment={},
        media_availability={"faces/person.jpg": False},
    )
    assert missing["face_image_url"] is None
    assert media.exists_calls == []


def test_filter_store_query_skips_unused_total(monkeypatch):
    calls = []

    class _Store:
        def list_filter(self, **kwargs):
            calls.append(kwargs)
            return [], 0

    monkeypatch.setattr(api, "get_detection_log_store", lambda: _Store())
    monkeypatch.setattr(api, "_filter_response_enrichment", lambda records: {})

    result = api.filter_logs(
        period="all",
        from_date_jalali=None,
        to_date_jalali=None,
        personnel_id=None,
        national_code=None,
        room_id=None,
        camera_id=None,
        section_id=None,
        building_id=None,
        access_granted=None,
        counts_for_attendance=None,
        log_type=None,
        min_confidence=None,
        max_confidence=None,
        include_thumbnails=False,
        skip=0,
        limit=200,
        _={},
    )

    assert result == []
    assert calls[0]["include_total"] is False
    assert calls[0]["include_thumbnails"] is False
    assert calls[0]["offset"] == 0
    assert calls[0]["limit"] == 200


def test_filter_jalali_dates_imply_custom_period_when_period_is_omitted() -> None:
    utc_start, utc_end = api._period_to_utc_range(
        "all",
        "1405-01-18",
        "1405-01-19",
    )

    assert datetime.fromisoformat(utc_start) == datetime(
        2026, 4, 6, 20, 30, tzinfo=timezone.utc
    )
    assert datetime.fromisoformat(utc_end) == datetime(
        2026, 4, 8, 20, 30, tzinfo=timezone.utc
    )


@pytest.mark.parametrize("period", ["today", "last_week", "last_month"])
def test_filter_explicit_jalali_dates_override_preset_period(period: str) -> None:
    utc_start, utc_end = api._period_to_utc_range(
        period,
        "1405-01-18",
        "1405-01-19",
    )

    assert datetime.fromisoformat(utc_start) == datetime(
        2026, 4, 6, 20, 30, tzinfo=timezone.utc
    )
    assert datetime.fromisoformat(utc_end) == datetime(
        2026, 4, 8, 20, 30, tzinfo=timezone.utc
    )


def test_filter_rejects_incomplete_implicit_custom_period() -> None:
    with pytest.raises(HTTPException) as exc_info:
        api._period_to_utc_range("all", "1405-01-18", None)

    assert exc_info.value.status_code == 400


def test_filter_rejects_invalid_explicit_jalali_date() -> None:
    with pytest.raises(HTTPException) as exc_info:
        api._period_to_utc_range("today", "not-a-date", "1405-01-18")

    assert exc_info.value.status_code == 400


def test_filter_rejects_reversed_implicit_custom_period() -> None:
    with pytest.raises(HTTPException) as exc_info:
        api._period_to_utc_range("all", "1405-01-19", "1405-01-18")

    assert exc_info.value.status_code == 400


def test_filter_jalali_datetime_range_converts_to_utc_hour_precision() -> None:
    utc_start, utc_end = api._period_to_utc_range(
        "all",
        None,
        None,
        "1405-01-18 08:30:00",
        "1405-01-18 10:15:00",
    )

    assert datetime.fromisoformat(utc_start) == datetime(
        2026, 4, 7, 5, 0, tzinfo=timezone.utc
    )
    assert datetime.fromisoformat(utc_end) == datetime(
        2026, 4, 7, 6, 45, tzinfo=timezone.utc
    )


def test_filter_datetime_jalali_overrides_date_and_period() -> None:
    utc_start, utc_end = api._period_to_utc_range(
        "today",
        "1405-01-18",
        "1405-01-19",
        "1405-01-18 08:30:00",
        "1405-01-18 10:15:00",
    )

    assert utc_start == datetime(2026, 4, 7, 5, 0, tzinfo=timezone.utc).isoformat()
    assert utc_end == datetime(2026, 4, 7, 6, 45, tzinfo=timezone.utc).isoformat()


@pytest.mark.parametrize(
    "from_dt,to_dt",
    [
        ("1405-01-18 08:30:00", None),
        (None, "1405-01-18 10:15:00"),
    ],
)
def test_filter_datetime_requires_both_boundaries(from_dt, to_dt) -> None:
    with pytest.raises(HTTPException) as exc_info:
        api._period_to_utc_range("all", None, None, from_dt, to_dt)

    assert exc_info.value.status_code == 400


def test_filter_rejects_invalid_jalali_datetime() -> None:
    with pytest.raises(HTTPException) as exc_info:
        api._period_to_utc_range("all", None, None, "not-a-datetime", "1405-01-18 10:15:00")

    assert exc_info.value.status_code == 400


def test_filter_rejects_reversed_datetime_range() -> None:
    with pytest.raises(HTTPException) as exc_info:
        api._period_to_utc_range(
            "all",
            None,
            None,
            "1405-01-18 10:15:00",
            "1405-01-18 08:30:00",
        )

    assert exc_info.value.status_code == 400


def test_filter_store_receives_datetime_range_boundaries(monkeypatch) -> None:
    calls = []

    class _Store:
        def list_filter(self, **kwargs):
            calls.append(kwargs)
            return [], 0

    monkeypatch.setattr(api, "get_detection_log_store", lambda: _Store())
    monkeypatch.setattr(api, "_filter_response_enrichment", lambda records: {})

    api.filter_logs(
        period="all",
        from_date_jalali=None,
        to_date_jalali=None,
        from_datetime_jalali="1405-01-18 08:30:00",
        to_datetime_jalali="1405-01-18 10:15:00",
        personnel_id=None,
        national_code=None,
        room_id=None,
        camera_id=None,
        section_id=None,
        building_id=None,
        access_granted=None,
        counts_for_attendance=None,
        log_type=None,
        min_confidence=None,
        max_confidence=None,
        include_thumbnails=False,
        skip=0,
        limit=200,
        _={},
    )

    assert calls[0]["from_date_utc"] == datetime(
        2026, 4, 7, 5, 0, tzinfo=timezone.utc
    ).isoformat()
    assert calls[0]["to_date_utc"] == datetime(
        2026, 4, 7, 6, 45, tzinfo=timezone.utc
    ).isoformat()


def test_filter_time_only_combines_with_date_range() -> None:
    utc_start, utc_end = api._period_to_utc_range(
        "all",
        "1405-01-18",
        "1405-01-18",
        None,
        None,
        "08:23:00",
        "08:25:00",
    )

    assert datetime.fromisoformat(utc_start) == datetime(
        2026, 4, 7, 4, 53, tzinfo=timezone.utc
    )
    assert datetime.fromisoformat(utc_end) == datetime(
        2026, 4, 7, 4, 55, tzinfo=timezone.utc
    )


def test_filter_time_only_accepts_persian_digits_and_no_seconds() -> None:
    utc_start, utc_end = api._period_to_utc_range(
        "all",
        "۱۴۰۵-۰۱-۱۸",
        "۱۴۰۵-۰۱-۱۸",
        None,
        None,
        "۰۸:۲۳",
        "۰۸:۲۵",
    )

    assert utc_start == datetime(2026, 4, 7, 4, 53, tzinfo=timezone.utc).isoformat()
    assert utc_end == datetime(2026, 4, 7, 4, 55, tzinfo=timezone.utc).isoformat()


def test_filter_time_only_requires_dates_and_both_times() -> None:
    with pytest.raises(HTTPException) as exc_info:
        api._period_to_utc_range("all", None, None, None, None, "08:23:00", "08:25:00")
    assert exc_info.value.status_code == 400

    with pytest.raises(HTTPException) as exc_info:
        api._period_to_utc_range(
            "all", "1405-01-18", "1405-01-18", None, None, "08:23:00", None
        )
    assert exc_info.value.status_code == 400


def test_filter_rejects_invalid_or_reversed_time_only_range() -> None:
    with pytest.raises(HTTPException) as exc_info:
        api._period_to_utc_range(
            "all", "1405-01-18", "1405-01-18", None, None, "not-a-time", "08:25:00"
        )
    assert exc_info.value.status_code == 400

    with pytest.raises(HTTPException) as exc_info:
        api._period_to_utc_range(
            "all", "1405-01-18", "1405-01-18", None, None, "08:25:00", "08:23:00"
        )
    assert exc_info.value.status_code == 400
