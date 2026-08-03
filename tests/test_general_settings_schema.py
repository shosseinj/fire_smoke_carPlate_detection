from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.general_settings import GeneralSettingsPatch, OperationalSettingsPatch


def test_general_settings_patch_accepts_force() -> None:
    payload = GeneralSettingsPatch.model_validate({"force": True})

    assert payload.force is True


def test_general_settings_patch_rejects_unknown_top_level_fields() -> None:
    with pytest.raises(ValidationError):
        GeneralSettingsPatch.model_validate({"unknown_setting": True})


def test_general_settings_operational_patch_rejects_fire_thresholds() -> None:
    with pytest.raises(ValidationError):
        OperationalSettingsPatch.model_validate(
            {
                "fire_confidence": 0.12,
                "smoke_confidence": 0.34,
            }
        )


@pytest.mark.parametrize("field", ["video_ingest_fps", "video_preview_fps"])
def test_general_settings_operational_patch_rejects_legacy_fps_fields(
    field: str,
) -> None:
    with pytest.raises(ValidationError):
        OperationalSettingsPatch.model_validate({field: 25})
