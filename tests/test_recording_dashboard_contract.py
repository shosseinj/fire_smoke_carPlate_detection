from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _dashboard() -> str:
    return (ROOT / "app" / "web" / "dashboard.html").read_text(encoding="utf-8")


def test_dashboard_exposes_persian_scheduled_recording_controls() -> None:
    html = _dashboard()
    assert '<html lang="fa" dir="rtl">' in html
    assert "ضبط زمان‌بندی‌شده" in html
    assert 'list="recordingCameraOptions"' in html
    assert html.count('type="datetime-local"') == 2
    assert "نام دوربین را جست‌وجو کنید" in html
    assert "زمان‌بندی ضبط" in html
    assert "مدت:" in html
    assert 'return detected || "Asia/Tehran"' in html


def test_recording_client_uses_opaque_source_ref_and_secure_routes() -> None:
    html = _dashboard()
    recording_start = html.index("const recordingStatusLabels")
    recording_end = html.index("document.addEventListener(\"fullscreenchange\"")
    recording_code = html[recording_start:recording_end]
    submit_start = recording_code.index('recordingForm.addEventListener("submit"')
    submit_code = recording_code[submit_start:]
    assert 'fetch("/api/v1/recordings/sources"' in recording_code
    assert "source_ref: source.source_ref" in submit_code
    assert "source_uri: source.source_uri" not in submit_code
    assert "source_display" in recording_code
    assert "localDateTimeToIso" in html
    assert "new Date(instant).toISOString()" in html
    assert 'fetch("/api/v1/recordings"' in html
    assert '/api/v1/recordings/${encodeURIComponent(jobId)}`' in html
    assert '/api/v1/recordings/${encodeURIComponent(jobId)}/cancel' in html
    assert '/api/v1/recordings/${encodeURIComponent(jobId)}/download' in html
    assert "window.setInterval(pollRecordingJobs, 3000)" in html


def test_recording_controls_do_not_acquire_live_transport() -> None:
    html = _dashboard()
    recording_start = html.index("const recordingStatusLabels")
    recording_end = html.index("function showLiveBranchError")
    recording_code = html[recording_start:recording_end]
    assert "acquireLiveBranch" not in recording_code
    assert "RTCPeerConnection" not in recording_code
    assert "whep" not in recording_code.lower()
    assert 'fetch("/api/v1/sources/preview-config"' in html
