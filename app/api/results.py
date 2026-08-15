from __future__ import annotations

import asyncio
import queue
import time

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect

from app.core.types import TaskName
from app.runtime import Runtime
from app.core.websocket_auth import authenticate_websocket, websocket_access_still_valid

router = APIRouter(prefix="/api/v1", tags=["results"])


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


@router.get("/results/recent")
def recent_results(
    source_id: str | None = None,
    task: TaskName | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
    runtime: Runtime = Depends(get_runtime),
) -> list[dict]:
    return runtime.results.recent(source_id=source_id, task=task, limit=limit)


@router.get("/router/status")
def router_status(runtime: Runtime = Depends(get_runtime)) -> dict:
    return runtime.status()


@router.websocket("/results/ws")
async def result_websocket(websocket: WebSocket, ticket: str | None = None) -> None:
    from app.main import runtime

    access = await authenticate_websocket(websocket, "results", ticket)
    if access is None:
        return
    await websocket.accept()
    subscriber_id, target = runtime.results.subscribe()
    next_access_check = time.monotonic() + 30.0
    try:
        while True:
            if time.monotonic() >= next_access_check:
                if not websocket_access_still_valid(access):
                    await websocket.close(code=1008, reason="دسترسی شما به این جریان لغو شده است")
                    return
                next_access_check = time.monotonic() + 30.0
            try:
                payload = await asyncio.to_thread(target.get, True, 20.0)
            except queue.Empty:
                await websocket.send_json({"type": "keepalive"})
                continue
            try:
                await websocket.send_json(payload)
            finally:
                target.task_done()
    except WebSocketDisconnect:
        pass
    finally:
        runtime.results.unsubscribe(subscriber_id)
