from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.runtime import Runtime

router = APIRouter(prefix="/api/v1/broadcast-gpu", tags=["broadcast-gpu"])


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


def _source_payload(record: Any, *, active: bool) -> dict[str, Any]:
    return {
        "source_id": str(record.id) if record.id is not None else record.source_uri,
        "source_uri": record.source_uri,
        "name": record.name,
        "enabled": bool(record.enabled),
        "active": active,
        "source_type": record.source_type,
        "tasks": sorted(task.value for task in record.tasks),
        "frame_width": record.frame_width,
        "frame_height": record.frame_height,
        "wall_profile": "320x320",
        "fullscreen_profile": "native",
    }


@router.get("/sources")
def list_broadcast_gpu_sources(runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    """Return every registry source and its actual live-branch readiness."""
    manager = runtime.live_branch
    sources = [
        _source_payload(
            record,
            active=bool(
                record.enabled
                and manager is not None
                and manager.enabled
                and manager.has_source(record.source_uri)
            ),
        )
        for record in runtime.registry.list()
    ]
    return {
        "enabled": bool(runtime.live_branch and runtime.live_branch.enabled),
        "sources": sources,
    }
