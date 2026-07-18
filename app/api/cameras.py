from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.core.source_registry import SourceRecord
from app.core.video_ingestor import VideoFileIngestor
from app.runtime import Runtime
from app.schemas import (
    CameraCreate,
    CameraBulkUpdate,
    CameraReplace,
    CameraResponse,
    CameraTaskUpdate,
    CameraUpdate,
)


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
        frame_width=record.frame_width,
        frame_height=record.frame_height,
        metadata=dict(record.metadata),
        created_at_utc=record.created_at_utc,
        updated_at_utc=record.updated_at_utc,
    )


@router.get("", response_model=list[CameraResponse])
def list_cameras(
    runtime: Runtime = Depends(get_runtime),
) -> list[CameraResponse]:
    return [_response(item) for item in runtime.registry.list()]


@router.post(
    "",
    response_model=CameraResponse,
    status_code=status.HTTP_201_CREATED,
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

    return _response(record)


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

    return _response(record)


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
        return _response(record)
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

        updated_cameras.append(_response(record))

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
        return _response(record)
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
        return _response(record)
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
        return _response(record)
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
        return _response(record)
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
