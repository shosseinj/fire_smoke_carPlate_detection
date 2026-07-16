from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.core.source_registry import SourceRecord
from app.core.video_ingestor import VideoFileIngestor
from app.runtime import Runtime
from app.schemas import CameraCreate, CameraReplace, CameraResponse, CameraUpdate


router = APIRouter(prefix="/api/v1/cameras", tags=["cameras"])


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


def _response(record: SourceRecord) -> CameraResponse:
    source_uri = record.source_uri
    if source_uri:
        source_uri = VideoFileIngestor.redact_uri(source_uri)
    return CameraResponse(
        camera_id=record.source_id,
        name=record.name,
        enabled=record.enabled,
        tasks=sorted(record.tasks, key=lambda task: task.value),
        source_uri=source_uri,
        metadata=dict(record.metadata),
        created_at_utc=record.created_at_utc,
        updated_at_utc=record.updated_at_utc,
    )


@router.get("", response_model=list[CameraResponse])
def list_cameras(runtime: Runtime = Depends(get_runtime)) -> list[CameraResponse]:
    return [_response(item) for item in runtime.registry.list()]


@router.post("", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
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
                metadata=dict(payload.metadata),
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _response(record)


@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(
    camera_id: str,
    runtime: Runtime = Depends(get_runtime),
) -> CameraResponse:
    record = runtime.registry.get(camera_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return _response(record)


@router.patch("/{camera_id}", response_model=CameraResponse)
def update_camera(
    camera_id: str,
    payload: CameraUpdate,
    runtime: Runtime = Depends(get_runtime),
) -> CameraResponse:
    values = payload.model_dump(exclude_unset=True)
    try:
        return _response(runtime.registry.update(camera_id, **values))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Camera not found") from exc


@router.put("/{camera_id}", response_model=CameraResponse)
def replace_camera(
    camera_id: str,
    payload: CameraReplace,
    runtime: Runtime = Depends(get_runtime),
) -> CameraResponse:
    try:
        return _response(
            runtime.registry.update(
                camera_id,
                name=payload.name,
                enabled=payload.enabled,
                tasks=payload.tasks,
                source_uri=payload.source_uri,
                metadata=payload.metadata,
            )
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Camera not found") from exc


@router.post("/{camera_id}/enable", response_model=CameraResponse)
def enable_camera(
    camera_id: str,
    runtime: Runtime = Depends(get_runtime),
) -> CameraResponse:
    try:
        return _response(runtime.registry.update(camera_id, enabled=True))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Camera not found") from exc


@router.post("/{camera_id}/disable", response_model=CameraResponse)
def disable_camera(
    camera_id: str,
    runtime: Runtime = Depends(get_runtime),
) -> CameraResponse:
    try:
        return _response(runtime.registry.update(camera_id, enabled=False))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Camera not found") from exc


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(
    camera_id: str,
    runtime: Runtime = Depends(get_runtime),
) -> Response:
    if not runtime.registry.delete(camera_id):
        raise HTTPException(status_code=404, detail="Camera not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
