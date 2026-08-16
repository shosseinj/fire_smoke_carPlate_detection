from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _dashboard() -> str:
    return (ROOT / "app" / "web" / "dashboard.html").read_text(encoding="utf-8")


def test_dashboard_exposes_persian_recording_settings_controls() -> None:
    html = _dashboard()
    assert '<html lang="fa" dir="rtl">' in html
    assert "تنظیمات ذخیره ویدئو" in html
    assert "ذخیره پیوسته فعال باشد" in html
    assert 'id="recordingQuality"' in html
    assert 'id="recordingSegment"' in html
    assert 'id="recordingRetention"' in html


def test_recording_settings_client_uses_opaque_source_references() -> None:
    html = _dashboard()
    assert "camera.source_ref" in html
    assert "camera.source_uri" not in html
    assert "encodeURIComponent(sourceRef)" in html
    assert "/api/v1/recording-settings" in html


def test_recording_settings_do_not_acquire_live_transport() -> None:
    html = _dashboard()
    start = html.index("function renderRecordingSettings")
    end = html.index('document.addEventListener("fullscreenchange"')
    code = html[start:end]
    assert "acquireLiveBranch" not in code
    assert "RTCPeerConnection" not in code
    assert "whep" not in code.lower()


def test_dashboard_hides_settings_without_read_permission() -> None:
    html = _dashboard()
    assert 'id="recordingSettingsPanel"' in html
    assert 'id="recordingContinuous"' in html
    assert "response.status === 401 || response.status === 403" in html
    assert "recordingSettingsPanel.hidden = true" in html

import pytest

pytestmark = pytest.mark.unit
