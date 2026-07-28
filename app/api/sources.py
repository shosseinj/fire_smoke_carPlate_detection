from __future__ import annotations

from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.core.media_preview import preview_stream_path
from app.core.source_registry import SourceRecord
from app.core.video_ingestor import VideoFileIngestor
from app.runtime import Runtime
from app.schemas import BulkSourceCreate, BulkSourceUpdateItem, SourceCreate, SourceResponse, SourceUpdate, TaskAssignment

router = APIRouter(prefix="/api/v1/sources", tags=["sources"])


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


def _resolve_source_record(runtime: Runtime, source_id: str) -> SourceRecord | None:
    record = runtime.registry.get(source_id)
    if record is not None:
        return record
    if source_id.isdigit():
        return runtime.registry.get_by_id(int(source_id))
    return None


_SOURCE_OVERRIDE_FIELDS = frozenset({
    "fire_confidence", "smoke_confidence", "plate_confidence",
    "plate_iou", "vehicle_confidence", "vehicle_iou",
    "face_human_confidence", "face_detection_confidence",
    "face_recognition_threshold",
})


def _save_source_overrides(
    payload: SourceCreate | SourceUpdate | BulkSourceUpdateItem,
    record: SourceRecord,
    runtime: Runtime,
) -> None:
    """Persist per-source confidence overrides to the ``sources`` table."""
    if not record.source_uri:
        return
    overrides: dict[str, float] = {}
    raw = payload.model_dump(exclude_unset=True)
    for field in _SOURCE_OVERRIDE_FIELDS:
        val = raw.get(field)
        if val is not None:
            overrides[field] = float(val)
    if overrides:
        runtime.source_settings.set(record.source_uri, overrides)


def _response(record: SourceRecord, runtime: Runtime | None = None) -> SourceResponse:
    value = record.to_dict()
    # Ensure id is always an int
    value["id"] = value.get("id") or 0
    if value.get("source_uri"):
        value["source_uri"] = VideoFileIngestor.redact_uri(value["source_uri"])
    # Resolve per-source confidence thresholds from the `sources` table
    if runtime is not None and record.source_uri:
        resolved = runtime.source_settings.resolve(record.source_uri)
        for field in (
            "fire_confidence", "smoke_confidence", "plate_confidence",
            "plate_iou", "vehicle_confidence", "vehicle_iou",
            "face_human_confidence", "face_detection_confidence",
            "face_recognition_threshold",
        ):
            value[field] = resolved.get(field)
    return SourceResponse(**value)


@router.get("", response_model=list[SourceResponse])
def list_sources(runtime: Runtime = Depends(get_runtime)) -> list[SourceResponse]:
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
            raise HTTPException(status_code=500, detail="آدرس پیش‌نمایش عمومی نامعتبر است")
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
                "source_uri": record.source_uri,
                "name": record.name,
                "enabled": record.enabled,
                "tasks": sorted(task.value for task in record.tasks),
                "frame_width": record.frame_width,
                "frame_height": record.frame_height,
                "fps": record.fps,
                "loop": record.loop,
                "draw_human": record.draw_human,
                "draw_zone": record.draw_zone,
                "draw_fire": record.draw_fire,
                "draw_smoke": record.draw_smoke,
                "draw_vehicle": record.draw_vehicle,
                "draw_plate": record.draw_plate,
                "preview_path": preview_stream_path(record.source_uri),
            }
            for record in runtime.registry.list()
        ],
    }


@router.post("", response_model=SourceResponse, status_code=status.HTTP_201_CREATED)
def create_source(payload: SourceCreate, runtime: Runtime = Depends(get_runtime)) -> SourceResponse:
    if payload.room_id is not None and runtime.location_store.get_room(payload.room_id) is None:
        raise HTTPException(status_code=404, detail="اتاق یافت نشد")
    try:
        record = runtime.registry.create(
            SourceRecord(
                source_uri=payload.source_uri,
                name=payload.name,
                enabled=payload.enabled,
                tasks=set(payload.tasks),
                frame_width=payload.frame_width,
                frame_height=payload.frame_height,
                source_type=payload.source_type,
                room_id=payload.room_id,
                metadata=dict(payload.metadata),
                fps=payload.fps,
                loop=payload.loop,
                draw_human=payload.draw_human,
                draw_zone=payload.draw_zone,
                draw_fire=payload.draw_fire,
                draw_smoke=payload.draw_smoke,
                draw_vehicle=payload.draw_vehicle,
                draw_plate=payload.draw_plate,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    # Save per-source confidence overrides if provided
    _save_source_overrides(payload, record, runtime)
    return _response(record, runtime)


@router.post("/bulk", response_model=list[SourceResponse], status_code=status.HTTP_201_CREATED)
def bulk_create_sources(
    payload: BulkSourceCreate,
    runtime: Runtime = Depends(get_runtime),
) -> list[SourceResponse]:
    """Create multiple sources in one request."""
    # Validate room_ids before any writes
    for item in payload.sources:
        if item.room_id is not None and runtime.location_store.get_room(item.room_id) is None:
            raise HTTPException(status_code=404, detail=f"اتاق با شناسه {item.room_id} یافت نشد")
    results: list[SourceResponse] = []
    for item in payload.sources:
        try:
            record = runtime.registry.create(
                SourceRecord(
                    source_uri=item.source_uri,
                    name=item.name,
                    enabled=item.enabled,
                    tasks=set(item.tasks),
                    frame_width=item.frame_width,
                    frame_height=item.frame_height,
                    source_type=item.source_type,
                    room_id=item.room_id,
                    metadata=dict(item.metadata),
                    fps=item.fps,
                    loop=item.loop,
                    draw_human=item.draw_human,
                    draw_zone=item.draw_zone,
                    draw_fire=item.draw_fire,
                    draw_smoke=item.draw_smoke,
                    draw_vehicle=item.draw_vehicle,
                    draw_plate=item.draw_plate,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        _save_source_overrides(item, record, runtime)
        results.append(_response(record, runtime))
    return results


@router.get("/enabled/ids")
def enabled_source_ids(runtime: Runtime = Depends(get_runtime)) -> dict[str, list[str]]:
    return {"source_ids": runtime.registry.enabled_source_ids()}


@router.put("/bulk", response_model=list[SourceResponse])
def bulk_update_sources(
    payload: list[BulkSourceUpdateItem],
    runtime: Runtime = Depends(get_runtime),
) -> list[SourceResponse]:
    """Update multiple sources in one request, each identified by database id."""
    if not payload:
        raise HTTPException(status_code=400, detail="لیست به‌روزرسانی خالی است")
    # Validate room_ids before any writes
    for item in payload:
        if item.room_id is not None and runtime.location_store.get_room(item.room_id) is None:
            raise HTTPException(status_code=404, detail=f"اتاق با شناسه {item.room_id} یافت نشد")
    results: list[SourceResponse] = []
    for item in payload:
        record = runtime.registry.get_by_id(item.id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"منبع با شناسه {item.id} یافت نشد")
        values = item.model_dump(exclude_unset=True)
        values.pop("id", None)
        new_source_uri = values.pop("source_uri", None)
        registry_values = {
            key: value
            for key, value in values.items()
            if key not in _SOURCE_OVERRIDE_FIELDS
        }
        try:
            if new_source_uri is not None and new_source_uri != record.source_uri:
                record = runtime.registry.rename(record.source_uri, new_source_uri)
            record = runtime.registry.update(record.source_uri, **registry_values)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"منبع با شناسه {item.id} یافت نشد") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        _save_source_overrides(item, record, runtime)
        results.append(_response(record, runtime))
    return results


@router.get("/{id:path}", response_model=SourceResponse)
def get_source(id: str, runtime: Runtime = Depends(get_runtime)) -> SourceResponse:
    record = _resolve_source_record(runtime, id)
    if record is None:
        raise HTTPException(status_code=404, detail="منبع یافت نشد")
    return _response(record, runtime)


@router.patch("/{id:path}", response_model=SourceResponse)
def update_source(
    id: str,
    payload: SourceUpdate,
    runtime: Runtime = Depends(get_runtime),
) -> SourceResponse:
    values = payload.model_dump(exclude_unset=True)
    if values.get("room_id") is not None and runtime.location_store.get_room(values["room_id"]) is None:
        raise HTTPException(status_code=404, detail="اتاق یافت نشد")
    record = _resolve_source_record(runtime, id)
    if record is None:
        raise HTTPException(status_code=404, detail="منبع یافت نشد")
    new_source_uri = values.pop("source_uri", None)
    registry_values = {
        key: value
        for key, value in values.items()
        if key not in _SOURCE_OVERRIDE_FIELDS
    }
    try:
        if new_source_uri is not None and new_source_uri != record.source_uri:
            record = runtime.registry.rename(record.source_uri, new_source_uri)
        record = runtime.registry.update(record.source_uri, **registry_values)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="منبع یافت نشد") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    # Save per-source confidence overrides if any were provided
    _save_source_overrides(payload, record, runtime)
    return _response(record, runtime)


@router.post("/{id:path}/enable", response_model=SourceResponse)
def enable_source(id: str, runtime: Runtime = Depends(get_runtime)) -> SourceResponse:
    try:
        record = _resolve_source_record(runtime, id)
        if record is None:
            raise KeyError(id)
        return _response(runtime.registry.update(record.source_uri, enabled=True), runtime)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="منبع یافت نشد") from exc


@router.post("/{id:path}/disable", response_model=SourceResponse)
def disable_source(id: str, runtime: Runtime = Depends(get_runtime)) -> SourceResponse:
    try:
        record = _resolve_source_record(runtime, id)
        if record is None:
            raise KeyError(id)
        return _response(runtime.registry.update(record.source_uri, enabled=False), runtime)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="منبع یافت نشد") from exc


@router.put("/bulk/task-assignment", response_model=list[SourceResponse])
def bulk_task_assignment(
    payload: TaskAssignment,
    runtime: Runtime = Depends(get_runtime),
) -> list[SourceResponse]:
    missing = [
        source_id
        for source_id in payload.source_ids
        if _resolve_source_record(runtime, source_id) is None
    ]
    if missing:
        raise HTTPException(status_code=404, detail={"missing_source_ids": missing})
    updated = [
        runtime.registry.update(
            _resolve_source_record(runtime, source_id).source_uri,
            tasks=payload.tasks,
            enabled=payload.enabled,
        )
        for source_id in payload.source_ids
    ]
    return [_response(item, runtime) for item in updated]


@router.delete("/{id:path}", status_code=status.HTTP_204_NO_CONTENT)
def delete_source(id: str, runtime: Runtime = Depends(get_runtime)) -> Response:
    record = _resolve_source_record(runtime, id)
    if record is None or not runtime.registry.delete(record.source_uri):
        raise HTTPException(status_code=404, detail="منبع یافت نشد")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
