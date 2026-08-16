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
    assert settings.human_event_media_enabled is False
    assert settings.human_event_media_write_enabled is False
    assert settings.human_event_clip_padding_seconds == 5.0
    assert settings.human_event_media_group == "detection-media-v1"
    assert settings.continuous_recording_temp_path.as_posix().endswith(
        "saved_media/temporary_minIO/continuous"
    )
    assert settings.human_event_media_temp_path.as_posix().endswith(
        "saved_media/temporary_minIO/human_track"
    )


def test_legacy_detection_persistence_defaults_true_and_accepts_false(monkeypatch) -> None:
    import importlib
    import app.config as config

    monkeypatch.delenv("LEGACY_DETECTION_PERSISTENCE_ENABLED", raising=False)
    assert importlib.reload(config).settings.legacy_detection_persistence_enabled is True
    monkeypatch.setenv("LEGACY_DETECTION_PERSISTENCE_ENABLED", "false")
    assert importlib.reload(config).settings.legacy_detection_persistence_enabled is False
    monkeypatch.delenv("LEGACY_DETECTION_PERSISTENCE_ENABLED")
    importlib.reload(config)


def test_detection_events_env_true_overrides_false_default(monkeypatch) -> None:
    import importlib
    import app.config as config

    monkeypatch.delenv("DETECTION_EVENTS_ENABLED", raising=False)
    assert importlib.reload(config).settings.detection_events_enabled is False
    monkeypatch.setenv("DETECTION_EVENTS_ENABLED", "true")
    assert importlib.reload(config).settings.detection_events_enabled is True
    monkeypatch.delenv("DETECTION_EVENTS_ENABLED")
    importlib.reload(config)
