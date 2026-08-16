from __future__ import annotations

from types import SimpleNamespace
import urllib.request

from starlette.requests import Request

from app.api import live_branch


class _Manager:
    enabled = True

    def acquire(self, source_uri: str, profile: str, session_id: str) -> dict[str, object]:
        return {
            "path": "live-branch/wall/camera",
            "url": "http://127.0.0.1:8789/live-branch/wall/camera",
            "width": 320,
            "height": 320,
        }


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "server": ("127.0.0.1", 9999),
            "path": "/api/v1/live-branch/wall/acquire",
            "headers": [(b"host", b"127.0.0.1:9999")],
        }
    )


def _remote_request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "server": ("192.168.110.10", 9999),
            "path": "/api/v1/live-branch/wall/acquire",
            "headers": [(b"host", b"192.168.110.10:9999")],
        }
    )


def test_acquire_does_not_probe_browser_whep_url(monkeypatch) -> None:
    def fail_if_probed(*_args, **_kwargs):
        raise AssertionError("the API container must not probe a browser-facing WHEP URL")

    monkeypatch.setattr(urllib.request, "urlopen", fail_if_probed)
    runtime = SimpleNamespace(
        live_branch=_Manager(),
        registry=SimpleNamespace(get=lambda source_uri: object()),
    )

    result = live_branch._acquire(
        "wall",
        live_branch.LiveBranchAcquire(source_uri="camera"),
        _request(),
        runtime,
    )

    assert result["enabled"] is True
    assert result["whep_url"] == "http://127.0.0.1:8789/live-branch/wall/camera/whep"
    assert result["session_id"]


def test_acquire_rewrites_loopback_whep_url_for_remote_dashboard() -> None:
    runtime = SimpleNamespace(
        live_branch=_Manager(),
        registry=SimpleNamespace(get=lambda source_uri: object()),
    )

    result = live_branch._acquire(
        "wall",
        live_branch.LiveBranchAcquire(source_uri="camera"),
        _remote_request(),
        runtime,
    )

    assert result["whep_url"] == (
        "http://192.168.110.10:8789/live-branch/wall/camera/whep"
    )

import pytest

pytestmark = pytest.mark.streaming
