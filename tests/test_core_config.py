from __future__ import annotations


def test_config_has_expected_defaults() -> None:
    from app.config import settings

    assert settings.app_name is not None
    assert settings.processor_mode in ("mock", "real")
    assert settings.saved_media_path is not None
    assert settings.detection_events_enabled is False
    assert settings.detection_events_queue_capacity > 0
    assert all((settings.detection_events_human_stream, settings.detection_events_fire_smoke_stream,
                settings.detection_events_plate_stream, settings.detection_events_recording_segment_stream))
