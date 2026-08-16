from __future__ import annotations

import pytest

from app.api.recording_settings import RecordingPolicyPatch, _changes
from app.core.recording_settings_store import QUALITY_PRESETS, RecordingPolicy


def test_recording_policy_defaults_are_storage_safe() -> None:
    policy = RecordingPolicy()
    assert policy.continuous_enabled is False
    assert policy.quality_preset == "medium"
    assert policy.segment_seconds == 120
    assert policy.retention_days == 30
    assert policy.to_dict()["output"] == QUALITY_PRESETS["medium"]


@pytest.mark.parametrize("field,value", [
    ("quality_preset", "ultra"), ("segment_seconds", 10), ("retention_days", 31),
])
def test_recording_policy_rejects_unsupported_choices(field: str, value: object) -> None:
    values = {"continuous_enabled": False, "quality_preset": "medium", "segment_seconds": 120, "retention_days": 30}
    values[field] = value
    with pytest.raises(ValueError):
        RecordingPolicy(**values)


def test_api_patch_accepts_practical_presets() -> None:
    changes = _changes(RecordingPolicyPatch(
        continuous_enabled=True, quality_preset="high", segment_seconds=300, retention_days=90,
    ))
    assert changes == {"continuous_enabled": True, "quality_preset": "high", "segment_seconds": 300, "retention_days": 90}

pytestmark = pytest.mark.unit
