from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.core.source_registry import SourceRecord
from app.core.video_ingestor import VideoFileIngestor
from app.runtime import Runtime
from app.schemas import SourceCreate, SourceResponse, SourceUpdate, TaskAssignment

router = APIRouter(prefix="/api/v1/sources", tags=["sources"])


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


def _response(record: SourceRecord) -> SourceResponse:
    value = record.to_dict()
    if value.get("source_uri"):
        value["source_uri"] = VideoFileIngestor.redact_uri(value["source_uri"])
    return SourceResponse(**value)


@router.get("", response_model=list[SourceResponse])
def list_sources(runtime: Runtime = Depends(get_runtime)) -> list[SourceResponse]:
    return [_response(item) for item in runtime.registry.list()]


@router.post("", response_model=SourceResponse, status_code=status.HTTP_201_CREATED)
def create_source(payload: SourceCreate, runtime: Runtime = Depends(get_runtime)) -> SourceResponse:
    try:
        record = runtime.registry.create(
            SourceRecord(
                source_id=payload.source_id,
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


@router.get("/enabled/ids")
def enabled_source_ids(runtime: Runtime = Depends(get_runtime)) -> dict[str, list[str]]:
    return {"source_ids": runtime.registry.enabled_source_ids()}


@router.get("/{source_id}", response_model=SourceResponse)
def get_source(source_id: str, runtime: Runtime = Depends(get_runtime)) -> SourceResponse:
    record = runtime.registry.get(source_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return _response(record)


@router.patch("/{source_id}", response_model=SourceResponse)
def update_source(
    source_id: str,
    payload: SourceUpdate,
    runtime: Runtime = Depends(get_runtime),
) -> SourceResponse:
    values = payload.model_dump(exclude_unset=True)
    try:
        record = runtime.registry.update(source_id, **values)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Source not found") from exc
    return _response(record)


@router.post("/{source_id}/enable", response_model=SourceResponse)
def enable_source(source_id: str, runtime: Runtime = Depends(get_runtime)) -> SourceResponse:
    try:
        return _response(runtime.registry.update(source_id, enabled=True))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Source not found") from exc


@router.post("/{source_id}/disable", response_model=SourceResponse)
def disable_source(source_id: str, runtime: Runtime = Depends(get_runtime)) -> SourceResponse:
    try:
        return _response(runtime.registry.update(source_id, enabled=False))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Source not found") from exc


@router.put("/bulk/task-assignment", response_model=list[SourceResponse])
def bulk_task_assignment(
    payload: TaskAssignment,
    runtime: Runtime = Depends(get_runtime),
) -> list[SourceResponse]:
    missing = [source_id for source_id in payload.source_ids if runtime.registry.get(source_id) is None]
    if missing:
        raise HTTPException(status_code=404, detail={"missing_source_ids": missing})
    updated = [
        runtime.registry.update(
            source_id,
            tasks=payload.tasks,
            enabled=payload.enabled,
        )
        for source_id in payload.source_ids
    ]
    return [_response(item) for item in updated]


@router.delete("/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_source(source_id: str, runtime: Runtime = Depends(get_runtime)) -> Response:
    if not runtime.registry.delete(source_id):
        raise HTTPException(status_code=404, detail="Source not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
