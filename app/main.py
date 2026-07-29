from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.frames import router as frames_router
from app.api.broadcast import router as broadcast_router

from app.api.results import router as results_router
from app.api.plate_logs import router as plate_logs_router
from app.api.car_plates import router as car_plates_router
from app.api.plate_settings import router as plate_settings_router
from app.api.models import router as models_router

from app.api.general_settings import router as general_settings_router
from app.api.fire_smoke_logs import router as fire_smoke_logs_router
from app.api.fire_logs import router as fire_logs_router
from app.api.diagnostics import router as diagnostics_router
from app.api.processor_tests import router as processor_tests_router
from app.api.sources import router as sources_router
from app.api.cams import router as cams_router
from app.api.auth import router as auth_router
from app.api.faces import router as faces_router
from app.api.humans import router as humans_router
from app.api.personnel import router as personnel_router
from app.api.locations import buildings_router, sections_router, rooms_router
from app.api.shifts import router as shifts_router
from app.api.holidays import router as holidays_router
from app.api.requests import router as requests_router
from app.api.personnel_requests import router as personnel_requests_router
from app.api.personnel_images import router as personnel_images_router
from app.api.detection_logs import router as detection_logs_router
from app.api.extract_frames import router as extract_frames_router
from app.api.developer import router as developer_router
from app.api.project_info import router as project_info_router
from app.api.import_progress import router as import_progress_router
from app.api.static_videos import router as static_videos_router

from app.config import settings
from app.core.detection_media import RestrictedMediaStaticFiles
from app.core.frontend_messages import install_frontend_exception_handlers, localize_frontend_payload
from app.runtime import build_runtime

runtime = build_runtime(settings)

SWAGGER_UI_DIRECTORY = Path(__file__).resolve().parent / "web" / "swagger-ui"

OPENAPI_TAGS = [
    {"name": "authentication"},
    {"name": "Buildings"},
    {"name": "Sections"},
    {"name": "Rooms"},
    {"name": "system-diagnostics"},
    {"name": "processor-tests"},
    {"name": "general-settings"},
    {"name": "model-management"},
    {"name": "sources"},
    {"name": "Cameras"},
    {"name": "frame-routing"},
    {"name": "results"},
    {"name": "annotated-broadcast"},
    {"name": "plate-logs"},
    {"name": "car-plates"},
    {"name": "fire-smoke"},
    {"name": "fire-logs"},
    {"name": "face-recognition"},
    {"name": "human-tracking"},
    {"name": "personnel"},
    {"name": "plate-settings"},
    {"name": "Shifts"},
    {"name": "Holidays"},
    {"name": "Personnel Requests"},
    {"name": "personnel-images"},
    {"name": "Detection Logs"},
    {"name": "Developer"},
    {"name": "Project Information"},
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
        "recognition. Open [/dashboard](/dashboard) for the synchronized annotated source wall."
    ),
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
    docs_url=None,
)
install_frontend_exception_handlers(app)
app.mount(
    "/docs-assets",
    StaticFiles(directory=SWAGGER_UI_DIRECTORY),
    name="docs-assets",
)


@app.get("/docs", include_in_schema=False)
def swagger_ui() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/docs-assets/swagger-ui-bundle.js",
        swagger_css_url="/docs-assets/swagger-ui.css",
        swagger_favicon_url="/docs-assets/favicon.svg",
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
def swagger_ui_redirect() -> HTMLResponse:
    return get_swagger_ui_oauth2_redirect_html()


app.include_router(auth_router)
app.include_router(buildings_router)
app.include_router(sections_router)
app.include_router(rooms_router)
app.include_router(diagnostics_router)
app.include_router(processor_tests_router)
app.include_router(general_settings_router)
app.include_router(models_router)
app.include_router(sources_router)
app.include_router(cams_router)
app.include_router(frames_router)
app.include_router(results_router)
app.include_router(broadcast_router)

app.include_router(plate_logs_router)
app.include_router(car_plates_router)
app.include_router(fire_smoke_logs_router)
app.include_router(fire_logs_router)
app.include_router(faces_router)
app.include_router(humans_router)
app.include_router(personnel_router)
app.include_router(shifts_router)
app.include_router(holidays_router)
app.include_router(requests_router)
app.include_router(personnel_requests_router)
app.include_router(personnel_images_router)
app.include_router(detection_logs_router)
app.include_router(extract_frames_router)
app.include_router(plate_settings_router)
app.include_router(developer_router)
app.include_router(project_info_router)
app.include_router(import_progress_router)
app.include_router(static_videos_router)
settings.saved_media_path.mkdir(parents=True, exist_ok=True)
for media_directory in (
    "fire_smoke_snapshots",
    "fire_smoke_videos",
    "plate_snapshots",
    "plate_videos",
    "detected_faces",
    "face_thumbnails",
    "human_face_videos",
    "human_snapshots",
    "whole_snapshots",
    "human_videos",
    "personnel_cropped_faces",
    "personnel_snapshots",
    "personnel_zip_errors",
):
    (settings.saved_media_path / media_directory).mkdir(parents=True, exist_ok=True)
settings.static_video_upload_path.mkdir(parents=True, exist_ok=True)
# Keep compatibility for non-person media while preventing direct unauthenticated
# access to personnel images and person-detection evidence.
app.mount(
    "/media",
    RestrictedMediaStaticFiles(directory=settings.saved_media_path),
    name="media",
)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,allow_origins=["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/dashboard")


@app.get("/health", tags=["system-diagnostics"], summary="Health and runtime counters")
def health() -> dict:
    status = runtime.status()
    return localize_frontend_payload({
        "status": "ok",
        "processor_mode": settings.processor_mode,
        "registered_sources": len(runtime.registry.list()),
        "router_started": status["started"],
        "video_ingestor": status["video_ingestor"],
        "media_preview": status["media_preview"],
        "broadcast": status["broadcast"],
        "plate_log_count": status["plate_log_count"],
        "fire_smoke_logs": status["fire_smoke_logs"],
        "human_logs": status["human_logs"],
        "shift_count": status["shift_count"],
        "holiday_count": status["holiday_count"],
        "request_count": status["request_count"],
        "detection_log_count": status["detection_log_count"],
        "workers": status["workers"],
    })
