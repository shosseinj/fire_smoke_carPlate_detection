from __future__ import annotations

import base64
from urllib.parse import urlsplit

import cv2
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from app.core.source_registry import SourceRecord
from app.core.media_preview import preview_stream_path
from app.core.operational_settings import CAMERA_SETTINGS_METADATA_KEY
from app.core.video_ingestor import VideoFileIngestor
from app.runtime import Runtime
from app.schemas import (
    CameraCreate,
    CameraBulkUpdate,
    CameraReplace,
    CameraResponse,
    CameraTaskUpdate,
    CameraSettingsPatch,
    CameraUpdate,
)


router = APIRouter(prefix="/api/v1/cameras", tags=["cameras"])


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


def _response(record: SourceRecord, runtime: Runtime | None = None) -> CameraResponse:
    source_uri = record.source_uri

    if source_uri:
        source_uri = VideoFileIngestor.redact_uri(source_uri)

    return CameraResponse(
        camera_id=record.source_id,
        name=record.name,
        enabled=record.enabled,
        tasks=sorted(record.tasks, key=lambda task: task.value),
        source_uri=source_uri,
        frame_width=record.frame_width,
        frame_height=record.frame_height,
        metadata={k: v for k, v in record.metadata.items() if k != CAMERA_SETTINGS_METADATA_KEY},
        created_at_utc=record.created_at_utc,
        updated_at_utc=record.updated_at_utc,
        settings_overrides=dict(record.metadata.get(CAMERA_SETTINGS_METADATA_KEY) or {}),
        effective_settings=(runtime.resolve_camera_settings(record.source_id) if runtime is not None else {}),
    )


def _legacy_response(record: SourceRecord, runtime: Runtime) -> dict[str, object]:
    """Expose the old camera field names without changing the native contract."""
    return {
        "id": record.source_id,
        "camera_id": record.source_id,
        "camera_name": record.name,
        "camera_url": VideoFileIngestor.redact_uri(record.source_uri) if record.source_uri else None,
        "camera_type": "rtsp" if (record.source_uri or "").lower().startswith("rtsp") else "file",
        "resolution": f"{record.frame_width}x{record.frame_height}",
        "fps": runtime.resolve_camera_settings(record.source_id).get("video_ingest_fps"),
        "is_active": record.enabled,
        "created_at": record.created_at_utc,
        "updated_at": record.updated_at_utc,
    }


class LegacyCameraBatchActiveUpdate(BaseModel):
    ids: list[str] = Field(min_length=1)
    active_status: list[bool] = Field(min_length=1)


class LegacyCameraHealthCheckRequest(BaseModel):
    camera_url: str = Field(min_length=1)


@router.get("", response_model=list[CameraResponse])
@router.get("/", response_model=list[CameraResponse], include_in_schema=False)
def list_cameras(
    runtime: Runtime = Depends(get_runtime),
) -> list[CameraResponse]:
    return [_response(item, runtime) for item in runtime.registry.list()]


@router.get("/preview-config")
def get_preview_config(
    request: Request,
    runtime: Runtime = Depends(get_runtime),
) -> dict:
    configured_base = runtime.settings.media_preview_whep_base_url
    if configured_base:
        parsed = urlsplit(configured_base)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise HTTPException(status_code=500, detail="Invalid public preview URL")
        whep_base_url = configured_base
    else:
        hostname = request.url.hostname or "127.0.0.1"
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        whep_base_url = f"{request.url.scheme}://{hostname}:8789"
    return {
        "enabled": runtime.settings.media_preview_enabled,
        "whep_base_url": whep_base_url,
        "sources": [
            {
                "source_id": record.source_id,
                "name": record.name,
                "enabled": record.enabled,
                "tasks": sorted(task.value for task in record.tasks),
                "frame_width": record.frame_width,
                "frame_height": record.frame_height,
                "preview_path": preview_stream_path(record.source_id),
            }
            for record in runtime.registry.list()
        ],
    }


@router.get("/active")
def list_active_cameras_legacy(runtime: Runtime = Depends(get_runtime)) -> list[dict[str, object]]:
    return [_legacy_response(item, runtime) for item in runtime.registry.list() if item.enabled]


@router.get("/active/effective")
def list_active_effective_cameras_legacy(runtime: Runtime = Depends(get_runtime)) -> list[dict[str, object]]:
    return [
        {
            **_legacy_response(item, runtime),
            "effective_settings": runtime.resolve_camera_settings(item.source_id),
        }
        for item in runtime.registry.list()
        if item.enabled
    ]


@router.get("/{camera_id}/effective-settings")
def get_effective_camera_settings_legacy(
    camera_id: str,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, object]:
    record = runtime.registry.get(camera_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {**_legacy_response(record, runtime), "effective_settings": runtime.resolve_camera_settings(camera_id)}


@router.patch("/batch-active")
def update_camera_active_legacy(
    payload: LegacyCameraBatchActiveUpdate,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, object]:
    if len(payload.ids) != len(payload.active_status):
        raise HTTPException(status_code=400, detail="ids and active_status must have equal lengths")
    missing: list[str] = []
    updated = 0
    for camera_id, enabled in zip(payload.ids, payload.active_status):
        if runtime.registry.get(camera_id) is None:
            missing.append(camera_id)
            continue
        runtime.registry.update(camera_id, enabled=enabled)
        updated += 1
    return {"updated_count": updated, "failed_ids": missing}


@router.post("/health-check")
def check_camera_health_legacy(payload: LegacyCameraHealthCheckRequest) -> dict[str, object]:
    capture = cv2.VideoCapture(payload.camera_url)
    try:
        if not capture.isOpened():
            return {"status": "unhealthy", "message": "Camera could not be opened", "snapshot": None}
        success, frame = capture.read()
        if not success or frame is None:
            return {"status": "unhealthy", "message": "Camera frame could not be read", "snapshot": None}
        encoded, buffer = cv2.imencode(".jpg", frame)
        if not encoded:
            return {"status": "unhealthy", "message": "Camera frame could not be encoded", "snapshot": None}
        return {
            "status": "healthy",
            "message": "Camera is active",
            "snapshot": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('ascii')}",
        }
    finally:
        capture.release()


@router.post("/create-cameras")
def legacy_create_cameras(runtime: Runtime = Depends(get_runtime)) -> dict[str, object]:
    """Keep the legacy setup route reachable without copying its private seed URLs."""
    cameras = [_legacy_response(item, runtime) for item in runtime.registry.list()]
    return {
        "message": "Use the current camera create endpoint to add sources",
        "delete_previous": False,
        "cameras": cameras,
    }


@router.delete("/delete-all-cameras")
def legacy_delete_all_cameras(runtime: Runtime = Depends(get_runtime)) -> dict[str, int]:
    deleted_count = sum(1 for item in runtime.registry.list() if runtime.registry.delete(item.source_id))
    return {"deleted_count": deleted_count}


@router.post(
    "",
    response_model=CameraResponse,
    status_code=status.HTTP_201_CREATED,
)
@router.post(
    "/",
    response_model=CameraResponse,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
def create_camera(
    payload: CameraCreate,
    runtime: Runtime = Depends(get_runtime),
) -> CameraResponse:
    try:
        record = runtime.registry.create(
            SourceRecord(
                source_id=payload.camera_id,
                name=payload.name,
                enabled=payload.enabled,
                tasks=set(payload.tasks),
                source_uri=payload.source_uri,
                frame_width=payload.frame_width,
                frame_height=payload.frame_height,
                metadata=dict(payload.metadata),
            )
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return _response(record, runtime)


@router.get("/{camera_id}/settings")
def get_camera_settings(camera_id: str, runtime: Runtime = Depends(get_runtime)) -> dict:
    record = runtime.registry.get(camera_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {
        "camera_id": camera_id,
        "overrides": dict(record.metadata.get(CAMERA_SETTINGS_METADATA_KEY) or {}),
        "effective": runtime.resolve_camera_settings(camera_id),
    }


@router.patch("/{camera_id}/settings")
def update_camera_settings(camera_id: str, payload: CameraSettingsPatch, runtime: Runtime = Depends(get_runtime)) -> dict:
    if runtime.registry.get(camera_id) is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    changes = payload.model_dump(exclude_unset=True)
    try:
        effective = runtime.update_camera_overrides(camera_id, changes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    record = runtime.registry.require(camera_id)
    return {"camera_id": camera_id, "overrides": dict(record.metadata.get(CAMERA_SETTINGS_METADATA_KEY) or {}), "effective": effective}


@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(
    camera_id: str,
    runtime: Runtime = Depends(get_runtime),
) -> CameraResponse:
    record = runtime.registry.get(camera_id)

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found",
        )

    return _response(record, runtime)


@router.put(
    "/{camera_id}/tasks",
    response_model=CameraResponse,
    summary="Select AI tasks or play-only mode",
    description=(
        "Send an empty tasks array for video playback with zero inference. "
        "The camera remains enabled and continues through DeepStream."
    ),
)
def update_camera_tasks(
    camera_id: str,
    payload: CameraTaskUpdate,
    runtime: Runtime = Depends(get_runtime),
) -> CameraResponse:
    try:
        record = runtime.registry.update(
            camera_id,
            tasks=payload.tasks,
        )
        return _response(record, runtime)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found",
        ) from exc


@router.patch(
    "/bulk",
    response_model=list[CameraResponse],
    summary="Update multiple cameras",
    description=(
        "Partially update multiple cameras in one request. Every item must contain "
        "a camera_id and at least one field to update. The request is rejected before "
        "applying changes if any camera ID is missing or duplicated."
    ),
)
def update_cameras_bulk(
    payload: list[CameraBulkUpdate],
    runtime: Runtime = Depends(get_runtime),
) -> list[CameraResponse]:
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="At least one camera must be provided",
        )

    camera_ids = [item.camera_id for item in payload]

    if len(camera_ids) != len(set(camera_ids)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Duplicate camera IDs are not allowed",
        )

    missing_camera_ids = [
        camera_id
        for camera_id in camera_ids
        if runtime.registry.get(camera_id) is None
    ]

    if missing_camera_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "message": "Some cameras were not found",
                "camera_ids": missing_camera_ids,
            },
        )

    prepared_updates: list[tuple[str, dict]] = []

    for item in payload:
        values = item.model_dump(
            exclude_unset=True,
            exclude={"camera_id"},
        )

        if not values:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"No update fields provided for camera: {item.camera_id}",
            )

        prepared_updates.append((item.camera_id, values))

    updated_cameras: list[CameraResponse] = []

    for camera_id, values in prepared_updates:
        try:
            record = runtime.registry.update(
                camera_id,
                **values,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Camera not found: {camera_id}",
            ) from exc

        updated_cameras.append(_response(record, runtime))

    return updated_cameras


@router.patch("/{camera_id}", response_model=CameraResponse)
def update_camera(
    camera_id: str,
    payload: CameraUpdate,
    runtime: Runtime = Depends(get_runtime),
) -> CameraResponse:
    values = payload.model_dump(exclude_unset=True)

    try:
        record = runtime.registry.update(
            camera_id,
            **values,
        )
        return _response(record, runtime)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found",
        ) from exc


@router.put("/{camera_id}", response_model=CameraResponse)
def replace_camera(
    camera_id: str,
    payload: CameraReplace,
    runtime: Runtime = Depends(get_runtime),
) -> CameraResponse:
    try:
        record = runtime.registry.update(
            camera_id,
            name=payload.name,
            enabled=payload.enabled,
            tasks=payload.tasks,
            source_uri=payload.source_uri,
            frame_width=payload.frame_width,
            frame_height=payload.frame_height,
            metadata=payload.metadata,
        )
        return _response(record, runtime)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found",
        ) from exc


@router.post("/{camera_id}/enable", response_model=CameraResponse)
def enable_camera(
    camera_id: str,
    runtime: Runtime = Depends(get_runtime),
) -> CameraResponse:
    try:
        record = runtime.registry.update(
            camera_id,
            enabled=True,
        )
        return _response(record, runtime)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found",
        ) from exc


@router.post("/{camera_id}/disable", response_model=CameraResponse)
def disable_camera(
    camera_id: str,
    runtime: Runtime = Depends(get_runtime),
) -> CameraResponse:
    try:
        record = runtime.registry.update(
            camera_id,
            enabled=False,
        )
        return _response(record, runtime)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found",
        ) from exc


@router.delete(
    "/{camera_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_camera(
    camera_id: str,
    runtime: Runtime = Depends(get_runtime),
) -> Response:
    if not runtime.registry.delete(camera_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found",
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)
