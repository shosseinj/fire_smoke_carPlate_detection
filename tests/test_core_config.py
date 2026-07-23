from __future__ import annotations


def test_config_has_expected_defaults() -> None:
    from app.config import settings

    assert settings.app_name is not None
    assert settings.processor_mode in ("mock", "real")
    assert settings.saved_media_path is not None