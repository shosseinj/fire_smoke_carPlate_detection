from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.api.cameras import _response as camera_response
from app.api.cameras import get_preview_config
from app.api.processor_tests import media_preview_smoke_test
from app.api.sources import _response as source_response
from app.core.media_preview import preview_stream_path
from app.core.source_registry import SourceRecord


def test_camera_and_source_aliases_expose_same_opaque_preview_path() -> None:
    record = SourceRecord(
        source_id="lobby-camera",
        name="Lobby",
        source_uri="rtsp://operator:secret@camera.internal/live",
    )

    camera = camera_response(record)
    source = source_response(record)

    assert camera.preview_path == source.preview_path
    assert camera.preview_path == preview_stream_path(record.source_id)
    assert record.source_id not in camera.preview_path
    assert "operator" not in camera.source_uri
    assert "secret" not in source.source_uri


def test_preview_config_contains_only_public_connection_parts() -> None:
    runtime = SimpleNamespace(
        settings=SimpleNamespace(
            media_preview_enabled=True,
            media_preview_whep_port=8889,
            media_preview_whep_path_suffix="/whep",
        )
    )
    config = get_preview_config(runtime).model_dump()

    assert config == {
        "enabled": True,
        "whep_port": 8889,
        "whep_path_suffix": "/whep",
    }
    assert "url" not in config
    assert "rtsp" not in str(config).lower()


def test_compose_media_preview_ports_and_internal_services() -> None:
    root = Path(__file__).resolve().parents[1]
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
    mediamtx = (root / "deploy" / "mediamtx.yml").read_text(encoding="utf-8")

    assert "bluenviron/mediamtx:1.18.2" in compose
    assert '"8554:8554"' not in compose
    assert '"8889:8889"' in compose
    assert '"8189:8189/udp"' in compose
    assert "9997:9997" not in compose
    assert "9998:9998" not in compose
    assert "MEDIA_PREVIEW_PUBLISH_BASE: rtsp://mediamtx:8554" in compose
    assert "MTX_WEBRTCADDITIONALHOSTS" in compose
    assert "apiAddress: :9997" in mediamtx
    assert "metricsAddress: :9998" in mediamtx


def test_media_preview_smoke_reports_publishers_without_source_uris() -> None:
    records = [
        SourceRecord(source_id="camera-a", name="A", source_uri="rtsp://user:secret@a/live"),
        SourceRecord(source_id="camera-b", name="B", source_uri="rtsp://user:secret@b/live"),
    ]
    source_status = {
        record.source_id: {"active": True}
        for record in records
    }
    runtime = SimpleNamespace(
        registry=SimpleNamespace(list=lambda: records),
        settings=SimpleNamespace(media_preview_enabled=True),
        media_preview_publisher=SimpleNamespace(status=lambda: {"sources": source_status}),
    )

    result = media_preview_smoke_test(runtime)

    assert result["_summary"]["status"] == "PASS"
    assert result["steps"][-1]["detail"] == "active=2,expected=2"
    assert "rtsp://" not in str(result)
    assert "secret" not in str(result)
