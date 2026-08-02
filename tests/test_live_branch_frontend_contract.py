from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_exposes_real_live_branch_surface() -> None:
    html = (ROOT / "app" / "web" / "dashboard.html").read_text(encoding="utf-8")
    assert "See live branch" in html
    assert "/api/v1/live-branch/" in html
    assert "RTCPeerConnection" in html
    assert "video.autoplay = true" in html
    assert "video.muted = true" in html
    assert "video.playsInline = true" in html
    assert "width: 260px" in html and "height: 260px" in html
    assert "/api/v1/broadcast-gpu/sources" in html
    assert "liveBranchSources.filter((item) => item.enabled && item.active !== false)" in html


def test_live_branch_routes_are_distinct_and_contract_is_typed() -> None:
    from app.api.live_branch import LiveBranchAcquire, LiveBranchSession

    assert LiveBranchAcquire.model_json_schema()["required"] == ["source_uri"]
    assert "session_id" in LiveBranchSession.model_json_schema()["required"]
    assert "/api/v1/live-branch" not in "/api/v1/broadcast"
