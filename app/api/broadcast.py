from __future__ import annotations

import asyncio
import json
import queue
import struct
from collections.abc import Iterator
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from app.core.broadcast import AnnotatedBroadcastHub
from app.runtime import Runtime

router = APIRouter(tags=["annotated-broadcast"])
DASHBOARD_PATH = Path(__file__).resolve().parents[1] / "web" / "dashboard.html"
MJPEG_BOUNDARY = "annotated-frame"


class BroadcastStateUpdate(BaseModel):
    enabled: bool


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


@router.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="Open annotated camera dashboard",
)
async def dashboard() -> HTMLResponse:
    return HTMLResponse(
        DASHBOARD_PATH.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@router.get("/api/v1/broadcast/state")
def broadcast_state(runtime: Runtime = Depends(get_runtime)) -> dict:
    return runtime.broadcast.status()


@router.put("/api/v1/broadcast/state")
def update_broadcast_state(
    payload: BroadcastStateUpdate,
    runtime: Runtime = Depends(get_runtime),
) -> dict:
    runtime.broadcast.set_enabled(payload.enabled)
    return runtime.broadcast.status()


@router.websocket("/api/v1/broadcast/ws")
async def annotated_broadcast_websocket(websocket: WebSocket) -> None:
    from app.main import runtime

    await websocket.accept()
    if not runtime.broadcast.enabled:
        await websocket.close(code=1013, reason="Frontend broadcasting is disabled")
        return
    subscriber_id, target = runtime.broadcast.subscribe()
    try:
        while runtime.broadcast.enabled:
            try:
                frame = await asyncio.to_thread(target.get, True, 20.0)
            except queue.Empty:
                await websocket.send_json({"type": "keepalive"})
                continue
            try:
                if frame is None:
                    break
                header = json.dumps(
                    {
                        "source_id": frame.source_id,
                        "frame_index": frame.frame_index,
                        "tasks": list(frame.tasks),
                    },
                    separators=(",", ":"),
                ).encode("utf-8")
                await websocket.send_bytes(
                    struct.pack("!I", len(header)) + header + frame.jpeg
                )
            finally:
                target.task_done()
    except WebSocketDisconnect:
        pass
    finally:
        runtime.broadcast.unsubscribe(subscriber_id)


def _mjpeg_stream(hub: AnnotatedBroadcastHub, source_id: str) -> Iterator[bytes]:
    version = 0
    while hub.enabled:
        frame = hub.wait_next(source_id, version, timeout=10.0)
        if frame is None:
            continue
        version = frame.version
        yield (
            f"--{MJPEG_BOUNDARY}\r\n"
            "Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(frame.jpeg)}\r\n\r\n"
        ).encode("ascii") + frame.jpeg + b"\r\n"


@router.get("/api/v1/broadcast/streams/{source_id}.mjpg")
def annotated_stream(
    source_id: str,
    runtime: Runtime = Depends(get_runtime),
) -> StreamingResponse:
    if runtime.registry.get(source_id) is None:
        raise HTTPException(status_code=404, detail="Source not found")
    if not runtime.broadcast.enabled:
        raise HTTPException(status_code=503, detail="Frontend broadcasting is disabled")
    return StreamingResponse(
        _mjpeg_stream(runtime.broadcast, source_id),
        media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/api/v1/broadcast/snapshots/{source_id}.jpg")
def annotated_snapshot(
    source_id: str,
    runtime: Runtime = Depends(get_runtime),
) -> Response:
    if not runtime.broadcast.enabled:
        raise HTTPException(status_code=503, detail="Frontend broadcasting is disabled")
    frame = runtime.broadcast.latest(source_id)
    if frame is None:
        if runtime.registry.get(source_id) is None:
            raise HTTPException(status_code=404, detail="Source not found")
        raise HTTPException(status_code=404, detail="No processed frame is available yet")
    return Response(
        content=frame.jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )
