from __future__ import annotations

import copy
import re
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from starlette.routing import WebSocketRoute

router = APIRouter(prefix="/api/v1/developer", tags=["Developer"])

_INTERNAL_DOC_PATHS = {
    "/openapi.json",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
}

_UNSAFE_TEST_PATHS = {
    "/api/v1/developer/test-request",
}

_SAMPLE_BY_FORMAT = {
    "date-time": "2026-01-01T08:30:00Z",
    "date": "2026-01-01",
    "time": "08:30:00",
    "email": "user@example.com",
    "uuid": "00000000-0000-4000-8000-000000000001",
    "uri": "https://example.com/resource",
}

_SAMPLE_BY_NAME = {
    "id": 1,
    "camera_id": 1,
    "room_id": 1,
    "section_id": 1,
    "building_id": 1,
    "personnel_id": 1,
    "log_id": 1,
    "national_code": "0012345678",
    "username": "admin",
    "password": "********",
    "token": "Bearer <access-token>",
    "access_token": "<access-token>",
    "camera_url": "rtsp://example-camera/stream",
    "camera_name": "Camera 1",
    "room_name": "\u0627\u062a\u0627\u0642 \u0646\u0645\u0648\u0646\u0647",
    "section_name": "\u0628\u062e\u0634 \u0646\u0645\u0648\u0646\u0647",
    "building_name": "\u0633\u0627\u062e\u062a\u0645\u0627\u0646 \u0646\u0645\u0648\u0646\u0647",
    "first_name": "\u0646\u0627\u0645",
    "last_name": "\u0646\u0627\u0645 \u062e\u0627\u0646\u0648\u0627\u062f\u06af\u06cc",
    "full_name": "\u0646\u0627\u0645 \u0646\u0627\u0645 \u062e\u0627\u0646\u0648\u0627\u062f\u06af\u06cc",
    "message": "\u067e\u06cc\u0627\u0645 \u0646\u0645\u0648\u0646\u0647",
    "detail": "\u062c\u0632\u0626\u06cc\u0627\u062a \u0646\u0645\u0648\u0646\u0647",
    "status": "active",
    "created_at": "2026-01-01T08:30:00Z",
    "updated_at": "2026-01-01T08:30:00Z",
    "detection_time": "2026-01-01T08:30:00Z",
    "jalali_date": "1404/10/11",
    "jalali_datetime": "1404/10/11 12:00",
    "log_type": "camera_rtsp",
    "detection_type": "known",
    "face_image_url": "/media/faces/sample.jpg",
    "body_image_url": "/media/bodies/sample.jpg",
    "snapshot_image_url": "/media/snapshots/sample.jpg",
    "video_url": "/media/videos/sample.mp4",
}


class DeveloperTestRequest(BaseModel):
    method: str = Field(examples=["GET"])
    path: str = Field(examples=["/api/v1/sources/active"])
    query: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    body: Any | None = None

    @field_validator("method")
    @classmethod
    def _normalize_method(cls, value: str) -> str:
        method = value.upper().strip()
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            raise ValueError("method must be one of GET, POST, PUT, PATCH, DELETE")
        return method

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        path = value.strip()
        if not path.startswith("/api/v1/"):
            raise ValueError("path must start with /api/v1/")
        if path in _UNSAFE_TEST_PATHS:
            raise ValueError("this developer endpoint cannot call itself")
        return path


def _normalize_key(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "default"


def _public_openapi(request: Request) -> dict[str, Any]:
    return request.app.openapi()


def _resolve_ref(schema: Any, openapi: dict[str, Any]) -> Any:
    if not isinstance(schema, dict):
        return schema
    ref = schema.get("$ref")
    if not ref or not isinstance(ref, str) or not ref.startswith("#/components/schemas/"):
        return schema
    name = ref.rsplit("/", 1)[-1]
    resolved = openapi.get("components", {}).get("schemas", {}).get(name)
    if resolved is None:
        return schema
    return resolved


def _merge_all_of(schema: dict[str, Any], openapi: dict[str, Any]) -> dict[str, Any]:
    if "allOf" not in schema:
        return schema
    merged: dict[str, Any] = {"type": "object", "properties": {}}
    for part in schema.get("allOf", []):
        resolved = _resolve_ref(part, openapi)
        if isinstance(resolved, dict):
            resolved = _merge_all_of(resolved, openapi)
            merged["properties"].update(resolved.get("properties", {}))
            if "required" in resolved:
                merged.setdefault("required", [])
                for item in resolved["required"]:
                    if item not in merged["required"]:
                        merged["required"].append(item)
    for key, value in schema.items():
        if key != "allOf":
            if key == "properties" and isinstance(value, dict):
                merged.setdefault("properties", {}).update(value)
            else:
                merged[key] = value
    return merged


def _sample_from_schema(schema: Any, openapi: dict[str, Any], field_name: str = "", depth: int = 0) -> Any:
    if depth > 8:
        return None
    if not isinstance(schema, dict):
        return None

    schema = copy.deepcopy(_resolve_ref(schema, openapi))
    if not isinstance(schema, dict):
        return None
    schema = _merge_all_of(schema, openapi)

    if "example" in schema:
        return schema["example"]
    if "examples" in schema:
        examples = schema["examples"]
        if isinstance(examples, list) and examples:
            return examples[0]
        if isinstance(examples, dict) and examples:
            first = next(iter(examples.values()))
            if isinstance(first, dict) and "value" in first:
                return first["value"]
            return first
    if "default" in schema:
        return schema["default"]
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]

    lower_name = field_name.lower()
    if lower_name in _SAMPLE_BY_NAME:
        return _SAMPLE_BY_NAME[lower_name]

    fmt = schema.get("format")
    if fmt in _SAMPLE_BY_FORMAT:
        return _SAMPLE_BY_FORMAT[fmt]

    if "anyOf" in schema:
        options = [item for item in schema["anyOf"] if item.get("type") != "null"]
        return _sample_from_schema(options[0], openapi, field_name, depth + 1) if options else None
    if "oneOf" in schema:
        return _sample_from_schema(schema["oneOf"][0], openapi, field_name, depth + 1) if schema["oneOf"] else None

    schema_type = schema.get("type")
    if not schema_type and "properties" in schema:
        schema_type = "object"

    if schema_type == "object":
        properties = schema.get("properties", {})
        if not properties:
            return {}
        return {
            name: _sample_from_schema(prop_schema, openapi, name, depth + 1)
            for name, prop_schema in properties.items()
        }
    if schema_type == "array":
        return [_sample_from_schema(schema.get("items", {}), openapi, field_name, depth + 1)]
    if schema_type == "integer":
        return 1
    if schema_type == "number":
        return 0.4
    if schema_type == "boolean":
        return True
    if schema_type == "string":
        return _SAMPLE_BY_NAME.get(lower_name, "string")
    return None


def _json_schema_for_content(content: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(content, dict):
        return None
    json_content = content.get("application/json") or content.get("application/*+json")
    if not isinstance(json_content, dict):
        return None
    schema = json_content.get("schema")
    return schema if isinstance(schema, dict) else None


def _extract_request_schema(operation: dict[str, Any]) -> dict[str, Any] | None:
    request_body = operation.get("requestBody")
    if not isinstance(request_body, dict):
        return None
    return _json_schema_for_content(request_body.get("content", {}))


def _extract_response_schema(operation: dict[str, Any]) -> dict[str, Any] | None:
    responses = operation.get("responses", {})
    for status_code in ("200", "201", "202", "204", "default"):
        response = responses.get(status_code)
        if isinstance(response, dict):
            schema = _json_schema_for_content(response.get("content", {}))
            if schema is not None:
                return schema
    for response in responses.values():
        if isinstance(response, dict):
            schema = _json_schema_for_content(response.get("content", {}))
            if schema is not None:
                return schema
    return None


def _parameters_by_location(operation: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {"path": [], "query": [], "header": [], "cookie": []}
    for parameter in operation.get("parameters", []) or []:
        if not isinstance(parameter, dict):
            continue
        location = parameter.get("in", "query")
        grouped.setdefault(location, []).append(
            {
                "name": parameter.get("name"),
                "required": bool(parameter.get("required", False)),
                "description": parameter.get("description"),
                "schema": parameter.get("schema", {}),
            }
        )
    return grouped


def _sample_query(operation: dict[str, Any], openapi: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for parameter in operation.get("parameters", []) or []:
        if not isinstance(parameter, dict) or parameter.get("in") != "query":
            continue
        name = parameter.get("name")
        if not name:
            continue
        if parameter.get("required"):
            result[name] = _sample_from_schema(parameter.get("schema", {}), openapi, name)
    return result


def _operation_to_endpoint(
    *,
    app_key: str,
    app_title: str,
    path: str,
    method: str,
    operation: dict[str, Any],
    openapi: dict[str, Any],
) -> dict[str, Any]:
    request_schema = _extract_request_schema(operation)
    response_schema = _extract_response_schema(operation)
    sample_body = _sample_from_schema(request_schema, openapi) if request_schema else None
    sample_response = _sample_from_schema(response_schema, openapi) if response_schema else None
    sample_query = _sample_query(operation, openapi)

    return {
        "app_key": app_key,
        "app_title": app_title,
        "method": method.upper(),
        "path": path,
        "operation_id": operation.get("operationId"),
        "summary": operation.get("summary") or operation.get("operationId") or f"{method.upper()} {path}",
        "description": operation.get("description"),
        "tags": operation.get("tags", []),
        "deprecated": bool(operation.get("deprecated", False)),
        "auth_required": bool(operation.get("security")),
        "parameters": _parameters_by_location(operation),
        "request_body_required": bool(operation.get("requestBody", {}).get("required", False))
        if isinstance(operation.get("requestBody"), dict)
        else False,
        "request_schema": request_schema,
        "sample_request_body": sample_body,
        "response_schema": response_schema,
        "sample_json_response": sample_response,
        "test_request": {
            "method": method.upper(),
            "path": path,
            "query": sample_query,
            "headers": {"Authorization": "Bearer <access-token>"} if bool(operation.get("security")) else {},
            "body": sample_body,
        },
    }


def _collect_http_endpoints(openapi: dict[str, Any]) -> list[dict[str, Any]]:
    endpoints: list[dict[str, Any]] = []
    paths = openapi.get("paths", {})
    for path, methods in paths.items():
        if path in _INTERNAL_DOC_PATHS or not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                continue
            if not isinstance(operation, dict):
                continue
            tags = operation.get("tags") or ["Default"]
            app_title = str(tags[0])
            app_key = _normalize_key(app_title)
            endpoints.append(
                _operation_to_endpoint(
                    app_key=app_key,
                    app_title=app_title,
                    path=path,
                    method=method,
                    operation=operation,
                    openapi=openapi,
                )
            )
    endpoints.sort(key=lambda item: (item["app_key"], item["path"], item["method"]))
    return endpoints


def _collect_websocket_endpoints(request: Request) -> list[dict[str, Any]]:
    endpoints: list[dict[str, Any]] = []
    for route_item in request.app.routes:
        if isinstance(route_item, WebSocketRoute):
            path = getattr(route_item, "path", "")
            if not path.startswith("/api/v1/"):
                continue
            endpoints.append(
                {
                    "app_key": "websocket",
                    "app_title": "WebSocket",
                    "method": "WEBSOCKET",
                    "path": path,
                    "operation_id": getattr(route_item, "name", None),
                    "summary": getattr(route_item, "name", None) or f"WEBSOCKET {path}",
                    "description": None,
                    "tags": ["WebSocket"],
                    "deprecated": False,
                    "auth_required": False,
                    "parameters": {"path": [], "query": [], "header": [], "cookie": []},
                    "request_body_required": False,
                    "request_schema": None,
                    "sample_request_body": None,
                    "response_schema": None,
                    "sample_json_response": None,
                    "test_request": {
                        "method": "WEBSOCKET",
                        "path": path,
                        "query": {},
                        "headers": {},
                        "body": None,
                    },
                }
            )
    endpoints.sort(key=lambda item: item["path"])
    return endpoints


def _collect_endpoints(request: Request) -> list[dict[str, Any]]:
    openapi = _public_openapi(request)
    return _collect_http_endpoints(openapi) + _collect_websocket_endpoints(request)


def _group_apps(endpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    apps: dict[str, dict[str, Any]] = {}
    for endpoint in endpoints:
        app_key = endpoint["app_key"]
        app = apps.setdefault(
            app_key,
            {
                "app_key": app_key,
                "title": endpoint["app_title"],
                "endpoint_count": 0,
                "methods": [],
                "paths": [],
            },
        )
        app["endpoint_count"] += 1
        if endpoint["method"] not in app["methods"]:
            app["methods"].append(endpoint["method"])
        if endpoint["path"] not in app["paths"]:
            app["paths"].append(endpoint["path"])

    result = list(apps.values())
    for item in result:
        item["methods"].sort()
        item["paths"].sort()
    result.sort(key=lambda item: item["title"].lower())
    return result


@router.get("/apps")
def list_developer_apps(request: Request) -> dict[str, Any]:
    endpoints = _collect_endpoints(request)
    apps = _group_apps(endpoints)
    return {
        "success": True,
        "total_apps": len(apps),
        "total_endpoints": len(endpoints),
        "items": apps,
    }


@router.get("/apps/{app_key}/urls")
def list_developer_app_urls(app_key: str, request: Request) -> dict[str, Any]:
    normalized_key = _normalize_key(app_key)
    endpoints = [item for item in _collect_endpoints(request) if item["app_key"] == normalized_key]
    if not endpoints:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="\u0627\u067e\u0644\u06cc\u06a9\u06cc\u0634\u0646 \u06cc\u0627 \u0645\u0627\u0698\u0648\u0644 \u0645\u0648\u0631\u062f \u0646\u0638\u0631 \u06cc\u0627\u0641\u062a \u0646\u0634\u062f",
        )
    return {
        "success": True,
        "app_key": normalized_key,
        "title": endpoints[0]["app_title"],
        "endpoint_count": len(endpoints),
        "items": endpoints,
    }


@router.get("/urls")
def list_all_developer_urls(request: Request) -> dict[str, Any]:
    endpoints = _collect_endpoints(request)
    return {
        "success": True,
        "total_endpoints": len(endpoints),
        "items": endpoints,
    }


@router.get("/openapi-summary")
def openapi_summary(request: Request) -> dict[str, Any]:
    openapi = _public_openapi(request)
    endpoints = _collect_endpoints(request)
    apps = _group_apps(endpoints)
    endpoints_by_app = {
        app["app_key"]: [endpoint for endpoint in endpoints if endpoint["app_key"] == app["app_key"]]
        for app in apps
    }
    return {
        "success": True,
        "title": openapi.get("info", {}).get("title"),
        "version": openapi.get("info", {}).get("version"),
        "total_apps": len(apps),
        "total_endpoints": len(endpoints),
        "apps": apps,
        "endpoints_by_app": endpoints_by_app,
    }


@router.post("/test-request")
async def test_developer_request(
    request: Request,
    payload: DeveloperTestRequest = Body(...),
) -> dict[str, Any]:
    try:
        import httpx
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="\u0628\u0631\u0627\u06cc \u0627\u062c\u0631\u0627\u06cc \u062a\u0633\u062a \u062f\u0627\u062e\u0644\u06cc endpoint\u060c \u067e\u06a9\u06cc\u062c httpx \u0628\u0627\u06cc\u062f \u0646\u0635\u0628 \u0628\u0627\u0634\u062f",
        ) from exc

    headers = dict(payload.headers or {})
    blocked_headers = {"host", "content-length", "connection"}
    headers = {key: value for key, value in headers.items() if key.lower() not in blocked_headers}

    transport = httpx.ASGITransport(app=request.app)
    async with httpx.AsyncClient(transport=transport, base_url=str(request.base_url).rstrip("/")) as client:
        response = await client.request(
            method=payload.method,
            url=payload.path,
            params=payload.query,
            headers=headers,
            json=payload.body if payload.body is not None else None,
            timeout=10.0,
        )

    try:
        response_body: Any = response.json()
    except Exception:
        response_body = response.text

    return {
        "success": response.status_code < 500,
        "request": {
            "method": payload.method,
            "path": payload.path,
            "query": payload.query,
            "headers": {key: ("***" if key.lower() == "authorization" else value) for key, value in headers.items()},
            "body": payload.body,
        },
        "response": {
            "status_code": response.status_code,
            "headers": {
                key: value
                for key, value in response.headers.items()
                if key.lower() in {"content-type", "content-length"}
            },
            "body": response_body,
        },
    }
