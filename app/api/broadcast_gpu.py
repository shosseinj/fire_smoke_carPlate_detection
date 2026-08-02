from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.runtime import Runtime

router = APIRouter(prefix="/api/v1/broadcast-gpu", tags=["broadcast-gpu"])


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


def _source_payload(record: Any) -> dict[str, Any]:
    return {
        "source_id": str(record.id) if record.id is not None else record.source_uri,
        "source_uri": record.source_uri,
        "name": record.name,
        "enabled": bool(record.enabled),
        "active": bool(record.enabled),
        "source_type": record.source_type,
        "tasks": sorted(task.value for task in record.tasks),
        "frame_width": record.frame_width,
        "frame_height": record.frame_height,
        "wall_profile": "320x320",
        "fullscreen_profile": "native",
    }


@router.get("/sources")
def list_broadcast_gpu_sources(runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    """Return the current enabled source registry for on-demand GPU branches."""
    sources = [
        _source_payload(record)
        for record in runtime.registry.list()
        if record.enabled
    ]
    return {
        "enabled": bool(runtime.live_branch and runtime.live_branch.enabled),
        "sources": sources,
    }
