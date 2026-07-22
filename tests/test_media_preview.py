from __future__ import annotations

from pathlib import Path

import pytest

from app.core.media_preview import (
    preview_publish_uri,
    preview_stream_path,
    redact_preview_uri,
    redact_rtsp_credentials,
)
from app.core.media_preview_publisher import MediaPreviewPublisher


class EmptyRegistry:
    def list(self) -> list[object]:
        return []


def test_preview_path_is_stable_opaque_and_source_specific() -> None:
    first = preview_stream_path("front-door-camera")

    assert first == preview_stream_path("front-door-camera")
    assert first != preview_stream_path("loading-bay-camera")
    assert first.startswith("preview-")
    assert "front" not in first
    assert len(first) == len("preview-") + 24


def test_publish_uri_preserves_base_path_and_redacts_credentials() -> None:
    uri = preview_publish_uri(
        "rtsp://publisher:private@mediamtx:8554/internal",
        "camera-01",
    )

    assert uri.endswith(f"/internal/{preview_stream_path('camera-01')}")
    assert redact_preview_uri(uri).startswith("rtsp://mediamtx:8554/")
    assert "publisher" not in redact_rtsp_credentials(f"sink failed for {uri}")
    assert "private" not in redact_rtsp_credentials(f"sink failed for {uri}")


@pytest.mark.parametrize(
    "base_uri",
    ["http://mediamtx:8889", "rtsp:///missing-host", "rtsp://host/path?q=1"],
)
def test_publish_uri_rejects_invalid_internal_bases(base_uri: str) -> None:
    with pytest.raises(ValueError):
        preview_publish_uri(base_uri, "camera-01")


def test_preview_publisher_status_never_exposes_internal_uri(tmp_path: Path) -> None:
    publisher = MediaPreviewPublisher(
        registry=EmptyRegistry(),  # type: ignore[arg-type]
        project_root=tmp_path,
        publish_base="rtsp://user:secret@mediamtx:8554",
        enabled=False,
    )

    status = publisher.status()
    assert status["enabled"] is False
    assert status["backend"] == "independent_h264_remux"
    assert "secret" not in str(status)


def test_preview_failure_is_scheduled_only_in_independent_publisher(
    tmp_path: Path,
) -> None:
    publisher = MediaPreviewPublisher(
        registry=EmptyRegistry(),  # type: ignore[arg-type]
        project_root=tmp_path,
        reconnect_seconds=3,
    )

    publisher._fail(
        "camera-01",
        "publish failed rtsp://publisher:private@mediamtx:8554/path",
    )

    assert publisher._failed == {"camera-01"}
    assert publisher._retry_after["camera-01"] > 0
    assert "private" not in publisher.status()["last_error"]
