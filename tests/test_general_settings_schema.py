from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.general_settings import OperationalSettingsPatch


def test_general_settings_operational_patch_rejects_fire_thresholds() -> None:
    with pytest.raises(ValidationError):
        OperationalSettingsPatch.model_validate(
            {
                "fire_confidence": 0.12,
                "smoke_confidence": 0.34,
            }
        )
