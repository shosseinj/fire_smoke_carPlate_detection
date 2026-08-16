from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi import Response

from app.api.recordings import RecordingCreate, create_recording, recording_source_options
from app.api.recordings import router
from app.core.recording_job_store import RecordingJob
from app.core.recording_source_ref import recording_source_ref


def test_recording_router_exposes_dashboard_download_and_authenticated_crud() -> None:
    routes = {(route.path, method) for route in router.routes for method in route.methods}
    assert ("/api/v1/recordings", "POST") in routes
    assert ("/api/v1/recordings", "GET") in routes
    assert ("/api/v1/recordings/{job_id}", "GET") in routes
    assert ("/api/v1/recordings/{job_id}/cancel", "POST") in routes
    assert ("/api/v1/recordings/{job_id}/download", "GET") in routes
    assert ("/api/v1/recordings/sources", "GET") in routes
    paths = [route.path for route in router.routes]
    assert paths.index("/api/v1/recordings/sources") < paths.index("/api/v1/recordings/{job_id}")
    assert all(any(dependency.call.__name__ == "get_current_user" for dependency in route.dependant.dependencies)
               for route in router.routes if route.path != "/api/v1/recordings")


def test_legacy_content_route_is_retained_but_hidden_from_openapi() -> None:
    content = next(route for route in router.routes if route.path.endswith("/{job_id}/content"))
    assert content.include_in_schema is False


def test_enqueue_failure_keeps_durable_job_accepted_and_requests_reconcile() -> None:
    start = datetime.now(timezone.utc) + timedelta(minutes=1)
    authoritative_uri = "rtsp://operator:highly-secret@camera.internal:8554/private-stream?token=hidden"
    job = RecordingJob("11111111-1111-1111-1111-111111111111", authoritative_uri, 7, None,
                       "scheduled", start, start + timedelta(minutes=1))

    class Store:
        def create(self, **_kwargs: object) -> tuple[RecordingJob, bool]:
            return job, True

    class Scheduler:
        def schedule_job(self, _job: RecordingJob) -> None:
            raise ConnectionError("redis unavailable")

    reconciles: list[bool] = []
    coordinator = SimpleNamespace(
        spool=SimpleNamespace(status=lambda: SimpleNamespace(high_water_exceeded=False)),
        request_reconcile=lambda: reconciles.append(True),
    )
    source = SimpleNamespace(source_uri=authoritative_uri, name="دوربین ورودی", enabled=True, source_type="rtsp")
    runtime = SimpleNamespace(
        settings=SimpleNamespace(recording_enabled=True),
        recording_coordinator=coordinator,
        registry=SimpleNamespace(list=lambda: [source], get=lambda uri: source if uri == authoritative_uri else None),
        live_branch=SimpleNamespace(enabled=True, has_source=lambda _uri: True),
    )
    response = Response()
    result = create_recording(
        RecordingCreate(source_ref=recording_source_ref(job.source_uri), scheduled_start_utc=job.scheduled_start_utc,
                        scheduled_end_utc=job.scheduled_end_utc),
        SimpleNamespace(id=7), Store(), Scheduler(), response, None, runtime,  # type: ignore[arg-type]
    )

    assert result.id == job.id
    assert response.status_code == 202
    assert reconciles == [True]
    payload = result.model_dump()
    assert "source_uri" not in payload
    assert payload["source_ref"] == recording_source_ref(authoritative_uri)
    assert "operator" not in payload["source_display"] and "highly-secret" not in payload["source_display"]


def test_authenticated_source_options_never_expose_rtsp_credentials() -> None:
    uri = "rtsp://admin:secret@camera.internal:8554/private?token=hidden"
    source = SimpleNamespace(source_uri=uri, name="دوربین امن", enabled=True, source_type="rtsp")
    runtime = SimpleNamespace(
        settings=SimpleNamespace(recording_enabled=True),
        registry=SimpleNamespace(list=lambda: [source]),
        live_branch=SimpleNamespace(enabled=True, has_source=lambda candidate: candidate == uri),
    )
    payload = [item.model_dump() for item in recording_source_options(SimpleNamespace(id=1), runtime)]  # type: ignore[arg-type]
    serialized = str(payload)
    assert payload[0]["source_ref"] == recording_source_ref(uri)
    assert payload[0]["source_name"] == "دوربین امن"
    assert not any(secret in serialized for secret in ("admin", "secret", "private", "token", "hidden"))

import pytest

pytestmark = pytest.mark.streaming
