from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from app.core.auth import get_current_user, require_role
from app.core.auth_store import UserRecord
from app.core.video_ingestor import VideoFileIngestor
from app.runtime import Runtime

router = APIRouter(prefix="/api/v1", tags=["developer"])


def get_runtime() -> Runtime:
    from app.main import runtime
    return runtime


# ── Project information ─────────────────────────────────────────────────


PROJECT_INFO = {
    "project": {
        "name": "Unified Video AI Task Router",
        "name_en": "Unified Video AI Task Router",
        "description": (
            "Dynamic source/task routing for batched fire/smoke, face recognition, "
            "and Iranian plate recognition."
        ),
        "current_version": "2.0.0",
        "api_version": "2.0.0",
    }
}

RELEASES: list[dict[str, Any]] = [
    {
        "version": "2.0.0",
        "title": "Dynamic task routing with DeepStream",
        "release_date": "2026-07-01",
        "status": "stable",
        "summary": "Production release with real-time task routing, face recognition, plate OCR, and fire/smoke detection.",
        "features": [
            {"number": 1, "title": "Fire/smoke severity pipeline", "description": "Rolling-window severity with TensorRT acceleration."},
            {"number": 2, "title": "Iranian plate OCR pipeline", "description": "Vehicle yolo -> plate yolo -> OCR with seven-digit and Persian letter validation."},
            {"number": 3, "title": "Face recognition pipeline", "description": "Human YOLO-pose, face detection, quality gates, ArcFace embeddings, Qdrant search."},
            {"number": 4, "title": "DeepStream GPU ingestion", "description": "NVIDIA DeepStream SDK with RTSP and local file support."},
        ],
    },
    {
        "version": "1.0.0",
        "title": "Initial release",
        "release_date": "2026-01-15",
        "status": "stable",
        "summary": "Initial release with basic fire/smoke and plate detection.",
        "features": [
            {"number": 1, "title": "Fire/smoke detection", "description": "Basic fire and smoke detection with configurable confidence thresholds."},
            {"number": 2, "title": "Plate detection", "description": "Iranian plate detection and OCR."},
        ],
    },
]


@router.get(
    "/project-info",
    summary="Get project information",
    description="Public endpoint returning project metadata.",
)
def project_info() -> dict[str, Any]:
    return dict(PROJECT_INFO)


@router.get(
    "/project-info/current",
    summary="Get current project version",
    description="Public endpoint returning current version of the project.",
)
def project_info_current() -> dict[str, Any]:
    return dict(PROJECT_INFO)


@router.get(
    "/project-info/releases",
    summary="List all project releases",
    description="Public endpoint returning all release records.",
)
def project_info_releases() -> dict[str, Any]:
    return {"releases": RELEASES, "total": len(RELEASES)}


@router.get(
    "/project-info/releases/{version}",
    summary="Get a specific release",
    description="Public endpoint returning one release by version.",
)
def project_info_release(version: str) -> dict[str, Any]:
    for release in RELEASES:
        if release["version"] == version:
            return dict(release)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"message": "Release not found", "version": version},
    )


# ── Developer discovery ────────────────────────────────────────────────


def _collect_apps(request: Request) -> list[dict[str, Any]]:
    routes = request.app.routes
    prefix_map: dict[str, list[str]] = {}
    for route in routes:
        methods = getattr(route, "methods", set()) or set()
        path = getattr(route, "path_str", getattr(route, "path", ""))
        if not path:
            continue
        if any(skip in path for skip in ("/docs", "/openapi", "/redoc", "/swagger", "/health", "/media", "/")):
            continue
        if methods == {"WEBSOCKET",} or methods == {"websocket",}:
            continue
        parts = path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] == "api":
            app_key = parts[1]
            prefix_map.setdefault(app_key, []).append(path)
    items: list[dict[str, Any]] = []
    for app_key, urls in sorted(prefix_map.items()):
        items.append({"app_key": app_key, "total_endpoints": len(urls), "urls": sorted(urls)})
    return items


@router.get(
    "/developer/apps",
    summary="List developer apps",
    description="Public endpoint listing all registered API apps and their endpoint counts.",
)
def developer_apps(request: Request) -> dict[str, Any]:
    apps = _collect_apps(request)
    total_endpoints = sum(item["total_endpoints"] for item in apps)
    return {
        "success": True,
        "total_apps": len(apps),
        "total_endpoints": total_endpoints,
        "items": [{"app_key": item["app_key"], "total_endpoints": item["total_endpoints"]} for item in apps],
    }


@router.get(
    "/developer/apps/{app_key}/urls",
    summary="List URLs for a specific app",
    description="Public endpoint returning all URLs registered under a specific app key.",
)
def developer_app_urls(app_key: str, request: Request) -> dict[str, Any]:
    apps = _collect_apps(request)
    for item in apps:
        if item["app_key"] == app_key:
            return {
                "success": True,
                "app_key": app_key,
                "total_endpoints": item["total_endpoints"],
                "items": item["urls"],
            }
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"message": "App not found", "app_key": app_key},
    )


@router.get(
    "/developer/urls",
    summary="List all developer URLs",
    description="Public endpoint listing all registered API URLs across all apps.",
)
def developer_urls(request: Request) -> dict[str, Any]:
    apps = _collect_apps(request)
    all_urls: list[str] = []
    for item in apps:
        all_urls.extend(item["urls"])
    return {
        "success": True,
        "total_endpoints": len(all_urls),
        "items": all_urls,
    }


@router.get(
    "/developer/openapi-summary",
    summary="OpenAPI summary",
    description="Public endpoint returning a summary of OpenAPI endpoints.",
)
def developer_openapi_summary(request: Request) -> dict[str, Any]:
    apps = _collect_apps(request)
    total_endpoints = sum(item["total_endpoints"] for item in apps)
    return {
        "success": True,
        "total_apps": len(apps),
        "total_endpoints": total_endpoints,
        "items": [{"app_key": item["app_key"], "total_endpoints": item["total_endpoints"]} for item in apps],
    }


# ── Test request (admin only, safe allowlist) ──────────────────────────


_ALLOWED_TARGETS: dict[str, set[str]] = {
    "/api/v1/diagnostics/overview": {"GET"},
    "/api/v1/diagnostics/checks": {"GET"},
    "/api/v1/router/status": {"GET"},
    "/api/v1/faces/status": {"GET"},
    "/api/v1/faces/quality-settings": {"GET"},
    "/api/v1/models/settings": {"GET"},
    "/api/v1/models/artifacts": {"GET"},
    "/api/v1/settings/general": {"GET"},
    "/api/v1/general-settings/": {"GET"},
    "/api/v1/tests/all": {"GET"},
    "/api/v1/cameras/health-check": {"POST"},
}

_TEST_ALLOWED_METHODS = frozenset({"GET", "POST"})


class TestRequestPayload(BaseModel):
    method: str = Field(default="GET", description="HTTP method")
    url: str = Field(description="Target URL path (e.g. /api/v1/diagnostics/overview)")
    headers: dict[str, str] = Field(default_factory=dict, description="Optional additional headers")
    body: dict[str, Any] | None = Field(default=None, description="JSON body for POST requests")


@router.post(
    "/developer/test-request",
    summary="Forward a test request to an internal endpoint",
    description=(
        "Admin/superuser-only endpoint that forwards requests to a safe allowlist of "
        "read-only or non-destructive diagnostic endpoints."
    ),
)
def test_request(
    payload: TestRequestPayload,
    request: Request,
    current_user: UserRecord = Depends(require_role("admin")),
) -> dict[str, Any]:
    method = payload.method.upper().strip()
    url = payload.url.strip()

    if method not in _TEST_ALLOWED_METHODS:
        return {
            "success": False,
            "status_code": 400,
            "method": method,
            "url": url,
            "detail": "Method not allowed",
        }

    if url not in _ALLOWED_TARGETS:
        return {
            "success": False,
            "status_code": 400,
            "method": method,
            "url": url,
            "detail": "URL not in allowed targets",
        }

    allowed_methods = _ALLOWED_TARGETS[url]
    if method not in allowed_methods:
        return {
            "success": False,
            "status_code": 400,
            "method": method,
            "url": url,
            "detail": f"Method {method} not allowed for this URL",
        }

    safe_headers = {}
    caller_headers = dict(payload.headers or {})
    for key, value in caller_headers.items():
        lower = key.lower()
        if lower in {"authorization", "cookie", "set-cookie", "host", "x-forwarded-for",
                       "x-forwarded-host", "x-real-ip", "proxy-authorization"}:
            continue
        safe_headers[key] = value

    app = request.app
    with TestClient(app) as client:
        if method == "GET":
            resp = client.get(url, headers=safe_headers)
        elif method == "POST":
            resp = client.post(url, json=payload.body, headers=safe_headers)
        else:
            return {
                "success": False,
                "status_code": 405,
                "method": method,
                "url": url,
                "detail": "Method not implemented",
            }

    try:
        data = resp.json()
    except Exception:
        data = resp.text

    return {
        "success": resp.status_code < 500,
        "status_code": resp.status_code,
        "method": method,
        "url": url,
        "data": data,
    }


# ── Camera health check (public, registered cameras only) ──────────────


class CameraHealthCheckPayload(BaseModel):
    camera_url: str = Field(description="Camera source URL to check")


@router.post(
    "/cameras/health-check",
    summary="Check camera reachability",
    description="Public endpoint that checks if a registered camera source is reachable.",
)
def camera_health_check(
    payload: CameraHealthCheckPayload,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    submitted_url = payload.camera_url.strip()

    cameras = runtime.registry.list()
    matched = None
    for cam in cameras:
        stored_uri = (cam.source_uri or "").strip()
        if stored_uri and stored_uri == submitted_url:
            matched = cam
            break

    if matched is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Camera URL not found in registered cameras", "camera_url": submitted_url},
        )

    try:
        reachable = VideoFileIngestor.check_source_reachable(matched.source_uri)
    except Exception:
        reachable = False

    return {
        "success": reachable,
        "camera_id": matched.source_id,
        "camera_url": VideoFileIngestor.redact_uri(matched.source_uri) if matched.source_uri else None,
        "reachable": reachable,
    }
