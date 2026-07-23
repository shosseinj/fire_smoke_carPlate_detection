from __future__ import annotations

import asyncio
import queue
import time
from pathlib import Path

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from app.core.broadcast import BroadcastControlEvent, EncodedBroadcastFrame
from app.runtime import Runtime


router = APIRouter(prefix="/api/v1/ws", tags=["legacy-websocket"])
VIDEO_PAGE_PATH = Path(__file__).resolve().parents[1] / "web" / "dashboard.html"
_connection_count = 0


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


def _legacy_metadata(frame: EncodedBroadcastFrame) -> dict[str, object]:
    return {
        "type": "video_metadata",
        "cam_id": frame.source_id,
        "persons": [],
        "scores": [],
        "timestamp": time.time(),
        "frame_index": frame.frame_index,
    }


@router.websocket("/live/{camera_id}")
async def legacy_live_feed(websocket: WebSocket, camera_id: str, runtime: Runtime = Depends(get_runtime)) -> None:
    global _connection_count
    if runtime.registry.get(camera_id) is None:
        await websocket.close(code=1008, reason="Camera not found")
        return
    await websocket.accept()
    _connection_count += 1
    subscriber_id, target = runtime.broadcast.subscribe()
    current_camera = camera_id
    sender = asyncio.create_task(_send_frames(websocket, target, lambda: current_camera))
    try:
        while True:
            message = await websocket.receive_text()
            if message == "ping":
                await websocket.send_json({"type": "pong", "timestamp": time.time()})
                continue
            if not message.startswith("subscribe:"):
                continue
            requested_camera = message.partition(":")[2].strip()
            if runtime.registry.get(requested_camera) is None:
                await websocket.send_json({"type": "error", "message": "Camera not found", "camera_id": requested_camera})
                continue
            previous_camera = current_camera
            current_camera = requested_camera
            await websocket.send_json({
                "type": "subscription",
                "status": "subscribed",
                "camera_id": current_camera,
                "message": f"Switched from {previous_camera} to {current_camera}",
            })
    except WebSocketDisconnect:
        pass
    finally:
        sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)
        runtime.broadcast.unsubscribe(subscriber_id)
        _connection_count = max(0, _connection_count - 1)


async def _send_frames(websocket: WebSocket, target: queue.Queue[object], camera_id) -> None:
    while True:
        try:
            item = await asyncio.to_thread(target.get, True, 20.0)
        except queue.Empty:
            await websocket.send_json({"type": "keepalive"})
            continue
        try:
            if item is None:
                return
            if isinstance(item, BroadcastControlEvent):
                continue
            if not isinstance(item, EncodedBroadcastFrame) or item.source_id != camera_id():
                continue
            await websocket.send_bytes(item.jpeg)
            await websocket.send_json(_legacy_metadata(item))
        finally:
            target.task_done()


@router.get("/connections/status")
def legacy_connection_status() -> dict[str, object]:
    return {"active_connections": _connection_count, "status": "healthy"}


@router.get("/video-page", response_class=HTMLResponse)
async def legacy_video_page() -> HTMLResponse:
    return HTMLResponse(VIDEO_PAGE_PATH.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})
