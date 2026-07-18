from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.processors.face_recognition import FaceRecognitionProcessor
from app.runtime import Runtime


router = APIRouter(prefix="/api/v1/humans", tags=["human-tracking"])


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


@router.get(
    "/logs",
    summary="List tracks with best snapshot, full-frame video, and face video links",
)
def logs(
    camera_id: str | None = Query(default=None),
    name: str | None = Query(default=None),
    track_id: int | None = Query(default=None, ge=1),
    limit: int = Query(default=100, ge=1, le=1000),
    runtime: Runtime = Depends(get_runtime),
) -> dict:
    items = runtime.human_logs.list(
        camera=camera_id,
        name=name,
        track_id=track_id,
        limit=limit,
    )
    return {"items": items, "count": len(items)}


@router.get("/active", summary="Current ByteTrack humans and remembered identities")
def active(runtime: Runtime = Depends(get_runtime)) -> dict:
    processor = runtime.face_processor
    if not isinstance(processor, FaceRecognitionProcessor):
        raise HTTPException(
            status_code=503,
            detail="Live human tracks are unavailable while PROCESSOR_MODE=mock",
        )
    items = processor.active_tracks()
    return {"items": items, "count": len(items)}


@router.get("/status", summary="Human log queue, database, snapshot, and video status")
def status(runtime: Runtime = Depends(get_runtime)) -> dict:
    return runtime.human_logs.status()
