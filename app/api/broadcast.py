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

from app.core.broadcast import (
    AnnotatedBroadcastHub,
    BroadcastControlEvent,
    EncodedBroadcastFrame,
)
from app.runtime import Runtime
from app.core.recent_detection_service import build_recent_detections_message

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
async def annotated_broadcast_websocket(
    websocket: WebSocket,
    wall: bool = False,
    metadata_only: bool = False,
    batch: bool = False,
    fullscreen_source: str | None = None,
    runtime: Runtime = Depends(get_runtime),
) -> None:
    await websocket.accept()
    if not runtime.broadcast.enabled:
        await websocket.close(code=1013, reason="Frontend broadcasting is disabled")
        return
    if fullscreen_source is not None and runtime.registry.get(fullscreen_source) is None:
        await websocket.close(code=1008, reason="Fullscreen source not found")
        return

    demand_lease = (
        runtime.stream_demand.acquire_ai()
        if getattr(runtime, "stream_demand", None) is not None
        else None
    )
    subscriber_id: str | None = None
    recent_task: asyncio.Task | None = None
    try:
        subscriber_id, target = runtime.broadcast.subscribe(
            wall=wall,
            fullscreen_source=fullscreen_source,
        )
        recent_task = asyncio.create_task(
            asyncio.to_thread(build_recent_detections_message, runtime)
        )
        recent_sent = False
        while runtime.broadcast.enabled:
            if not recent_sent and recent_task.done():
                recent_sent = True
                try:
                    recent_message = recent_task.result()
                    await websocket.send_json(
                        recent_message
                        or {"type": "recent_detections", "detections": [], "count": 0}
                    )
                except Exception:
                    import logging
                    logging.getLogger("uvicorn.error").exception(
                        "Failed to send recent detections"
                    )
            try:
                frame = await asyncio.to_thread(target.get, True, 20.0)
            except queue.Empty:
                await websocket.send_json({"type": "keepalive"})
                continue
            try:
                if frame is None:
                    break
                if isinstance(frame, BroadcastControlEvent):
                    await websocket.send_json(frame.payload)
                    continue
                if metadata_only:
                    await websocket.send_json(
                        {
                            "type": "frame_metadata",
                            "source_id": frame.source_id,
                            "frame_index": frame.frame_index,
                            "tasks": list(frame.tasks),
                            "frame_width": frame.frame_width,
                            "frame_height": frame.frame_height,
                        }
                    )
                    continue
                if batch:
                    frames = [frame]
                    stop_after_send = False
                    await asyncio.sleep(0.005)
                    while len(frames) < 64:
                        try:
                            candidate = target.get_nowait()
                        except queue.Empty:
                            break
                        try:
                            if candidate is None:
                                stop_after_send = True
                                break
                            if isinstance(candidate, BroadcastControlEvent):
                                await websocket.send_json(candidate.payload)
                            else:
                                frames.append(candidate)
                        finally:
                            target.task_done()
                    latest_by_source: dict[str, EncodedBroadcastFrame] = {}
                    for item in frames:
                        previous = latest_by_source.get(item.source_id)
                        if previous is None or (
                            item.frame_index,
                            len(item.tasks),
                            item.version,
                        ) >= (
                            previous.frame_index,
                            len(previous.tasks),
                            previous.version,
                        ):
                            latest_by_source[item.source_id] = item
                    records = [
                        _annotated_frame_payload(
                            item,
                            wall=wall,
                            fullscreen_source=fullscreen_source,
                        )
                        for item in latest_by_source.values()
                    ]
                    await websocket.send_bytes(_source_frame_batch_payload(records))
                    if stop_after_send:
                        break
                    continue
                await websocket.send_bytes(
                    _annotated_frame_payload(
                        frame,
                        wall=wall,
                        fullscreen_source=fullscreen_source,
                    )
                )
            finally:
                target.task_done()
    except WebSocketDisconnect:
        pass
    finally:
        if recent_task is not None and not recent_task.done():
            recent_task.cancel()
        if subscriber_id is not None:
            runtime.broadcast.unsubscribe(subscriber_id)
        if demand_lease is not None:
            demand_lease.release()


@router.websocket("/api/v1/video-wall/ws")
async def source_video_wall_websocket(
    websocket: WebSocket,
    wall: bool = True,
    fullscreen_source: str | None = None,
    batch: bool = False,
    runtime: Runtime = Depends(get_runtime),
) -> None:
    """Stream source frames without AI overlays or result dependencies."""
    await websocket.accept()
    if not runtime.broadcast.enabled:
        await websocket.close(code=1013, reason="Video wall is disabled")
        return
    if fullscreen_source is not None and runtime.registry.get(fullscreen_source) is None:
        await websocket.close(code=1008, reason="Fullscreen source not found")
        return

    demand_lease = (
        runtime.stream_demand.acquire_video(fullscreen_source)
        if getattr(runtime, "stream_demand", None) is not None
        else None
    )
    subscriber_id: str | None = None
    try:
        subscriber_id, target = runtime.broadcast.subscribe_source_only(
            wall=wall,
            fullscreen_source=fullscreen_source,
        )
        while runtime.broadcast.enabled:
            try:
                frame = await asyncio.to_thread(target.get, True, 20.0)
            except queue.Empty:
                await websocket.send_json({"type": "keepalive"})
                continue
            try:
                if frame is None:
                    break
                if fullscreen_source is not None and frame.source_id != fullscreen_source:
                    continue
                frames = [frame]
                if batch:
                    # Briefly coalesce independently rendered cameras so one
                    # ASGI/WebSocket send carries a useful wall sweep. This is
                    # bounded well below one 15 FPS frame interval.
                    await asyncio.sleep(0.005)
                    while len(frames) < 64:
                        try:
                            candidate = target.get_nowait()
                        except queue.Empty:
                            break
                        if candidate is None:
                            target.task_done()
                            break
                        if (
                            fullscreen_source is None
                            or candidate.source_id == fullscreen_source
                        ):
                            frames.append(candidate)
                        else:
                            target.task_done()
                    dequeued_count = len(frames)
                    latest_by_source = {
                        item.source_id: item
                        for item in frames
                    }
                    frames = list(latest_by_source.values())
                    records = [
                        _source_frame_payload(
                            item,
                            wall=wall,
                            fullscreen_source=fullscreen_source,
                        )
                        for item in frames
                    ]
                    await websocket.send_bytes(_source_frame_batch_payload(records))
                    for _ in range(dequeued_count - 1):
                        target.task_done()
                else:
                    await websocket.send_bytes(
                        _source_frame_payload(
                            frame,
                            wall=wall,
                            fullscreen_source=fullscreen_source,
                        )
                    )
            finally:
                target.task_done()
    except WebSocketDisconnect:
        pass
    finally:
        if subscriber_id is not None:
            runtime.broadcast.unsubscribe_source_only(subscriber_id)
        if demand_lease is not None:
            demand_lease.release()


def _source_frame_payload(
    frame: EncodedBroadcastFrame,
    *,
    wall: bool,
    fullscreen_source: str | None,
) -> bytes:
    jpeg, width, height, profile = frame.rendition(
        full_resolution=not wall or frame.source_id == fullscreen_source
    )
    header = {
        "type": "source_frame",
        "source_id": frame.source_id,
        "frame_index": frame.frame_index,
        "frame_width": width,
        "frame_height": height,
        "render_profile": profile,
        "ai_processed": False,
    }
    encoded_header = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("!I", len(encoded_header)) + encoded_header + jpeg


def _annotated_frame_payload(
    frame: EncodedBroadcastFrame,
    *,
    wall: bool,
    fullscreen_source: str | None,
) -> bytes:
    jpeg, width, height, profile = frame.rendition(
        full_resolution=not wall or frame.source_id == fullscreen_source
    )
    header = {
        "source_id": frame.source_id,
        "frame_index": frame.frame_index,
        "tasks": list(frame.tasks),
        "render_profile": profile,
        "frame_width": width,
        "frame_height": height,
    }
    encoded_header = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("!I", len(encoded_header)) + encoded_header + jpeg


def _source_frame_batch_payload(records: list[bytes]) -> bytes:
    header = json.dumps(
        {"type": "source_frame_batch", "count": len(records)},
        separators=(",", ":"),
    ).encode()
    payload = bytearray(struct.pack("!I", len(header)) + header)
    for record in records:
        payload.extend(struct.pack("!I", len(record)))
        payload.extend(record)
    return bytes(payload)


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
