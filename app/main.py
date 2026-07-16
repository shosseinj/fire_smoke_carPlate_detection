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
from app.api.fire_smoke_logs import router as fire_smoke_logs_router
from app.api.sources import router as sources_router
from app.config import settings
from app.runtime import build_runtime

runtime = build_runtime(settings)


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
        "Dynamic source/task routing for batched fire/smoke and Iranian plate "
        "recognition. Open [/dashboard](/dashboard) for the synchronized annotated camera wall."
    ),
    lifespan=lifespan,
)
app.include_router(sources_router)
app.include_router(cameras_router)
app.include_router(frames_router)
app.include_router(results_router)
app.include_router(broadcast_router)
app.include_router(plate_logs_router)
app.include_router(fire_smoke_logs_router)
app.mount("/media", StaticFiles(directory=settings.saved_media_path), name="media")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/dashboard")


@app.get("/health")
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
        "workers": status["workers"],
    }
