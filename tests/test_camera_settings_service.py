from __future__ import annotations


def test_resolve_camera_setting_uses_global_only_for_none() -> None:
    def resolve_camera_setting(camera_val: object, global_val: object) -> object:
        return camera_val if camera_val is not None else global_val

    assert resolve_camera_setting(None, True) is True
    assert resolve_camera_setting(False, True) is False
    assert resolve_camera_setting(0.0, 0.4) == 0.0
    assert resolve_camera_setting(0.8, 0.4) == 0.8
import pytest

pytestmark = pytest.mark.unit
