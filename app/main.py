from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.frames import router as frames_router
from app.api.broadcast import router as broadcast_router
from app.api.cameras import router as cameras_router
from app.api.results import router as results_router
from app.api.plate_logs import router as plate_logs_router
from app.api.plate_settings import router as plate_settings_router
from app.api.models import router as models_router
from app.api.general_settings import router as general_settings_router
from app.api.fire_smoke_logs import router as fire_smoke_logs_router
from app.api.diagnostics import router as diagnostics_router
from app.api.sources import router as sources_router
from app.api.faces import router as faces_router
from app.api.humans import router as humans_router
from app.config import settings
from app.runtime import build_runtime

runtime = build_runtime(settings)

OPENAPI_TAGS = [
    {
        "name": "system-diagnostics",
        "description": "Maintenance checks, configuration visibility, and controlled pipeline restart.",
    },
    {
        "name": "general-settings",
        "description": "One place for model selection, dynamic policies, startup config, and camera modes.",
    },
    {
        "name": "model-management",
        "description": "Model catalog, TensorRT and ONNX conversion jobs, selection, and fallback order.",
    },
    {
        "name": "cameras",
        "description": "Authoritative camera-table CRUD, sizing, enable/disable, and task assignment.",
    },
    {
        "name": "frame-routing",
        "description": "Upload JPEG/PNG frames to test fire/smoke and plate model routing from Swagger.",
    },
    {
        "name": "fire-smoke",
        "description": "Stable incident logs and online rolling-window severity configuration.",
    },
    {
        "name": "face-recognition",
        "description": "Enrollment, identity management, and persistent landmark/pose quality gates.",
    },
    {
        "name": "human-tracking",
        "description": "ByteTrack identity history, best snapshots, human videos, and accepted-face videos.",
    },
    {
        "name": "plate-settings",
        "description": "General defaults and per-camera overrides with automatic inheritance.",
    },
    {
        "name": "plate-logs",
        "description": "Persistent recognized-plate records and snapshots.",
    },
    {
        "name": "results",
        "description": "Recent model results, worker status, and result WebSocket.",
    },
    {
        "name": "annotated-broadcast",
        "description": "Dashboard, annotated streams, snapshots, and multiplexed WebSocket.",
    },
    {
        "name": "sources",
        "description": "Backward-compatible alias for the camera registry.",
    },
]


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.start()
    try:
        yield
    finally:
        runtime.close()


app = FastAPI(
    title=settings.app_name,
    version="2.0.0",
    description=(
        "Dynamic source/task routing for batched fire/smoke, face recognition, and Iranian plate "
        "recognition. Open [/dashboard](/dashboard) for the synchronized annotated camera wall."
    ),
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
)
app.include_router(diagnostics_router)
app.include_router(general_settings_router)
app.include_router(models_router)
app.include_router(sources_router)
app.include_router(cameras_router)
app.include_router(frames_router)
app.include_router(results_router)
app.include_router(broadcast_router)
app.include_router(plate_logs_router)
app.include_router(fire_smoke_logs_router)
app.include_router(faces_router)
app.include_router(humans_router)
app.include_router(plate_settings_router)
app.mount("/media", StaticFiles(directory=settings.saved_media_path), name="media")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/dashboard")


@app.get("/health", tags=["system-diagnostics"], summary="Health and runtime counters")
def health() -> dict:
    status = runtime.status()
    return {
        "status": "ok",
        "processor_mode": settings.processor_mode,
        "registered_sources": len(runtime.registry.list()),
        "router_started": status["started"],
        "video_ingestor": status["video_ingestor"],
        "broadcast": status["broadcast"],
        "plate_log_count": status["plate_log_count"],
        "fire_smoke_logs": status["fire_smoke_logs"],
        "human_logs": status["human_logs"],
        "workers": status["workers"],
    }
