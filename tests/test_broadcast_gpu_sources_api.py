from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.api.broadcast_gpu import list_broadcast_gpu_sources
from app.api.sources import _source_record_for_create
from app.core.source_registry import SourceRecord
from app.core.video_ingestor import VideoFileIngestor
from app.schemas import SourceCreate


def _runtime(records: list[SourceRecord], attached: set[str] | None = None) -> SimpleNamespace:
    registry = SimpleNamespace(list=lambda: list(records))
    attached = attached or set()
    manager = SimpleNamespace(enabled=True, has_source=lambda source_uri: source_uri in attached)
    return SimpleNamespace(registry=registry, live_branch=manager)


def test_enabled_static_source_is_returned_with_live_branch_fields() -> None:
    record = SourceRecord(
        id=7,
        source_uri="file:///workspace/data/1.mp4",
        name="Sample static video",
        source_type="static_video",
        enabled=True,
    )
    result = list_broadcast_gpu_sources(_runtime([record], {record.source_uri}))
    assert result["enabled"] is True
    assert result["sources"] == [{
        "source_id": "7",
        "source_uri": "file:///workspace/data/1.mp4",
        "name": "Sample static video",
        "enabled": True,
        "active": True,
        "source_type": "static_video",
        "tasks": [],
        "frame_width": 640,
        "frame_height": 640,
        "wall_profile": "320x320",
        "fullscreen_profile": "native",
    }]


def test_all_sources_are_returned_and_active_requires_live_attachment() -> None:
    enabled = SourceRecord(id=1, source_uri="rtsp://camera/1", name="Camera", enabled=True)
    disabled = SourceRecord(id=2, source_uri="rtsp://camera/2", name="Disabled", enabled=False)
    records = [enabled, disabled]
    runtime = _runtime(records, {enabled.source_uri, disabled.source_uri})
    first = list_broadcast_gpu_sources(runtime)["sources"]
    assert [item["source_id"] for item in first] == ["1", "2"]
    assert [item["active"] for item in first] == [True, False]
    added = SourceRecord(id=3, source_uri="file:///workspace/data/1.mp4", name="Static", source_type="static_video")
    records.append(added)
    refreshed = list_broadcast_gpu_sources(runtime)["sources"]
    assert {item["source_id"] for item in refreshed} == {"1", "2", "3"}
    assert next(item for item in refreshed if item["source_id"] == "3")["active"] is False


def test_production_source_registration_accepts_existing_local_static_file(tmp_path: Path) -> None:
    video = tmp_path / "1.mp4"
    video.write_bytes(b"not decoded here")
    payload = SourceCreate(
        source_uri=str(video),
        name="Uploaded-path static source",
        source_type="static_video",
    )
    static_store = SimpleNamespace(get=lambda _uri: None)
    runtime = SimpleNamespace(static_video_store=static_store)
    record, was_uploaded = _source_record_for_create(payload, runtime)
    assert record.source_uri == str(video)
    assert record.source_type == "static_video"
    assert was_uploaded is False


def test_file_uri_resolves_to_container_path() -> None:
    ingestor = VideoFileIngestor.__new__(VideoFileIngestor)
    ingestor.project_root = Path("/")
    resolved = ingestor._resolve_uri("file:///workspace/data/1.mp4")
    assert resolved.replace("\\", "/").endswith("/workspace/data/1.mp4")
