from __future__ import annotations

import asyncio
import queue

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect

from app.core.types import TaskName
from app.runtime import Runtime

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
async def result_websocket(websocket: WebSocket) -> None:
    from app.main import runtime

    await websocket.accept()
    subscriber_id, target = runtime.results.subscribe()
    try:
        while True:
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
