from __future__ import annotations

import threading
import uuid
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.runtime import Runtime

router = APIRouter(prefix="/api/v1/live-branch", tags=["live-branch"])
Profile = Literal["wall", "fullscreen"]


class LiveBranchAcquire(BaseModel):
    source_uri: str = Field(min_length=1)
    client_id: str | None = Field(default=None, min_length=1, max_length=200)


class LiveBranchSession(BaseModel):
    source_uri: str = Field(min_length=1)
    session_id: str = Field(min_length=1, max_length=100)


_sessions: dict[str, tuple[str, Profile, str]] = {}
_sessions_lock = threading.RLock()


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


def _manager(runtime: Runtime) -> Any:
    manager = runtime.live_branch
    if manager is None:
        raise HTTPException(status_code=503, detail="live branch manager is unavailable")
    return manager


def _browser_whep_url(url: str, request: Request) -> str:
    if url.startswith(("http://", "https://")):
        parsed = urlsplit(url)
        hostname = parsed.hostname or ""
        if hostname not in {"mediamtx", "video-ai-router"} and "." in hostname:
            return url if url.endswith("/whep") else f"{url.rstrip('/')}/whep"
        path = parsed.path
    else:
        path = url
    host = request.headers.get("host", "127.0.0.1").split(":", 1)[0]
    scheme = "https" if request.url.scheme == "https" else "http"
    return f"{scheme}://{host}:8889/{path.strip('/')}/whep"


def _acquire(profile: Profile, payload: LiveBranchAcquire, request: Request, runtime: Runtime) -> dict[str, Any]:
    manager = _manager(runtime)
    if not manager.enabled:
        return {"enabled": False, "source_id": payload.source_uri, "source_uri": payload.source_uri, "profile": profile}
    if runtime.registry.get(payload.source_uri) is None:
        raise HTTPException(status_code=404, detail="source_uri was not found")
    client_id = payload.client_id or uuid.uuid4().hex
    with _sessions_lock:
        session_id = uuid.uuid4().hex
        _sessions[session_id] = (payload.source_uri, profile, client_id)
    try:
        result = manager.acquire(payload.source_uri, profile, session_id)
    except Exception as exc:
        with _sessions_lock:
            _sessions.pop(session_id, None)
        raise HTTPException(status_code=503, detail=f"live branch acquire failed: {exc}") from exc
    path = str(result.get("path", ""))
    url = str(result.get("url", ""))
    whep_url = _browser_whep_url(url, request)
    return {
        "enabled": True,
        "session_id": session_id,
        "source_id": payload.source_uri,
        "source_uri": payload.source_uri,
        "profile": profile,
        "stream_path": path,
        "whep_url": whep_url,
    }


@router.post("/{profile}/acquire")
def acquire(
    profile: Profile,
    payload: LiveBranchAcquire,
    request: Request,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    return _acquire(profile, payload, request, runtime)


@router.post("/{profile}/heartbeat")
def heartbeat(
    profile: Profile,
    payload: LiveBranchSession,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    manager = _manager(runtime)
    if not manager.enabled:
        return {"enabled": False, "source_id": payload.source_uri, "source_uri": payload.source_uri, "profile": profile}
    with _sessions_lock:
        owner = _sessions.get(payload.session_id)
    if owner is None or owner[0] != payload.source_uri or owner[1] != profile:
        raise HTTPException(status_code=409, detail="stale or non-owned live branch session")
    if not manager.heartbeat(payload.source_uri, profile, payload.session_id):
        raise HTTPException(status_code=409, detail="live branch session expired")
    return {"enabled": True, "session_id": payload.session_id, "source_id": payload.source_uri, "source_uri": payload.source_uri, "profile": profile, "alive": True}


@router.delete("/{profile}/release")
def release(
    profile: Profile,
    payload: LiveBranchSession,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    manager = _manager(runtime)
    if not manager.enabled:
        return {"enabled": False, "source_id": payload.source_uri, "source_uri": payload.source_uri, "profile": profile, "released": False}
    with _sessions_lock:
        owner = _sessions.get(payload.session_id)
        if owner is None or owner[0] != payload.source_uri or owner[1] != profile:
            raise HTTPException(status_code=409, detail="stale or non-owned live branch session")
        _sessions.pop(payload.session_id, None)
    released = manager.release(payload.source_uri, profile, payload.session_id)
    return {"enabled": True, "session_id": payload.session_id, "source_id": payload.source_uri, "source_uri": payload.source_uri, "profile": profile, "released": released}
