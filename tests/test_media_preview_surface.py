from __future__ import annotations

from app.core.media_preview import preview_stream_path


def test_preview_stream_path_is_stable_without_exposing_source_id() -> None:
    first = preview_stream_path("camera-a")
    second = preview_stream_path("camera-a")

    assert first == second
    assert "camera-a" not in first
