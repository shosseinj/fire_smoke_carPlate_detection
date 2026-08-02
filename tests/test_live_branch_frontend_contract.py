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
    assert "attempt < 10" in html


def test_live_branch_routes_are_distinct_and_contract_is_typed() -> None:
    from app.api.live_branch import LiveBranchAcquire, LiveBranchSession

    assert LiveBranchAcquire.model_json_schema()["required"] == ["source_uri"]
    assert "session_id" in LiveBranchSession.model_json_schema()["required"]
    assert "/api/v1/live-branch" not in "/api/v1/broadcast"


def test_fullscreen_acquire_route_is_registered_for_frontend_path() -> None:
    from app.api.live_branch import router

    matching = [
        route for route in router.routes
        if route.path == "/api/v1/live-branch/{profile}/acquire"
        and "POST" in route.methods
    ]
    assert matching, "POST /api/v1/live-branch/fullscreen/acquire must resolve through the profile route"


def test_wall_acquire_url_is_get_discoverable_without_creating_a_session() -> None:
    from app.api.live_branch import router

    matching = [
        route for route in router.routes
        if route.path == "/api/v1/live-branch/{profile}/acquire"
        and "GET" in route.methods
    ]
    assert matching, "GET wall acquire compatibility/discovery route must be registered"
