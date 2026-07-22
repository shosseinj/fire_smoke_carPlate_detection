from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import require_role
from app.core.auth_store import UserRecord
from app.core.deepstream_ingestor import DeepStreamIngestor
from app.core.fps_diagnostics import build_fps_report
from app.core.video_ingestor import VideoFileIngestor
from app.core.types import TaskName
from app.runtime import Runtime

router = APIRouter(
    prefix="/api/v1/diagnostics",
    tags=["system-diagnostics"],
)


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


def _fps_snapshot(runtime: Runtime) -> dict[str, Any]:
    router_status = runtime.router.status()
    return {
        "video_ingestor": (
            runtime.video_ingestor.status()
            if runtime.video_ingestor is not None
            else {"enabled": False, "running": False, "sources": {}}
        ),
        "workers": router_status["workers"],
        "broadcast": runtime.broadcast.status(),
    }


@router.get(
    "/fps",
    summary="Measure FPS and identify the limiting stage",
    description=(
        "Samples cumulative ingestion, worker, and broadcast counters over a bounded "
        "window. It reports configured caps, source starvation, task backpressure, "
        "processor failures, and delivered resolution bandwidth without exposing URIs."
    ),
)
async def fps_diagnostics(
    sample_seconds: Annotated[float, Query(ge=0.5, le=10.0)] = 2.0,
    expected_fps: Annotated[float | None, Query(gt=0.0, le=240.0)] = None,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    desired_fps = float(expected_fps or runtime.settings.video_preview_fps)
    camera_tasks = {
        camera.source_id: tuple(sorted(task.value for task in camera.tasks))
        for camera in runtime.registry.list()
        if camera.enabled
    }
    before = _fps_snapshot(runtime)
    started = time.monotonic()
    await asyncio.sleep(sample_seconds)
    observed_seconds = time.monotonic() - started
    after = _fps_snapshot(runtime)
    report = build_fps_report(
        before,
        after,
        sample_seconds=observed_seconds,
        expected_fps=desired_fps,
        camera_tasks=camera_tasks,
    )
    report["sampled_at_utc"] = datetime.now(timezone.utc).isoformat()
    return report


@router.get(
    "/overview",
    summary="Inspect every runtime section",
    description=(
        "Read-only overview for camera ingestion, model workers, configured "
        "confidence thresholds, broadcast, and persistent logs."
    ),
)
def diagnostics_overview(runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    status = runtime.status()
    cameras = runtime.registry.list()
    return {
        "camera_registry": {
            "total": len(cameras),
            "enabled": sum(1 for camera in cameras if camera.enabled),
            "revision": runtime.registry.revision,
        },
        "deepstream": status["video_ingestor"],
        "model_configuration": {
            "plate_pipeline": "vehicle -> plate -> OCR",
            "fire_minimum_score": runtime.settings.fire_confidence,
            "smoke_minimum_score": runtime.settings.smoke_confidence,
            "plate_minimum_score": runtime.settings.plate_confidence,
            "plate_class_ids": list(runtime.settings.plate_class_ids),
            "vehicle_minimum_score": runtime.settings.vehicle_confidence,
            "fire_model": {
                "path": str(runtime.settings.fire_model_path),
                "exists": runtime.settings.fire_model_path.is_file(),
            },
            "plate_detector": {
                "path": str(runtime.settings.plate_detector_weights),
                "exists": runtime.settings.plate_detector_weights.is_file(),
            },
            "vehicle_detector": {
                "path": str(runtime.settings.vehicle_detector_weights),
                "exists": runtime.settings.vehicle_detector_weights.is_file(),
                "class_ids": list(runtime.settings.vehicle_class_ids),
            },
            "face_recognition": runtime.face_processor.status(),
        },
        "fire_smoke_policy": runtime.fire_smoke_logs.settings(),
        "plate_detection_policy": runtime.plate_settings.general(),
        "model_management": runtime.models.snapshot(),
        "fire_smoke_logs": status["fire_smoke_logs"],
        "human_logs": status["human_logs"],
        "plate_log_count": status["plate_log_count"],
        "broadcast": status["broadcast"],
        "workers": status["workers"],
    }


@router.get(
    "/checks",
    summary="Run organized maintenance checks",
    description=(
        "Runs safe, read-only checks. WARN indicates configuration or source "
        "attention; FAIL indicates a required runtime section is unavailable."
    ),
)
def maintenance_checks(runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    status = runtime.status()
    video = status["video_ingestor"]
    enabled = [camera for camera in runtime.registry.list() if camera.enabled]
    active = set(video.get("sources", {}))
    face_assigned = any(
        TaskName.FACE_RECOGNITION in camera.tasks for camera in enabled
    )
    face_models_ready = all(
        path.is_file()
        for path in (
            runtime.settings.face_human_model_path,
            runtime.settings.face_detector_model_path,
            runtime.settings.face_embedding_model_path,
        )
    )
    local_missing = []
    project_root = runtime.settings.data_path.parent
    for camera in enabled:
        uri = camera.source_uri or ""
        if not uri or VideoFileIngestor.is_rtsp_uri(uri) or "://" in uri:
            continue
        path = runtime.video_ingestor.project_root / uri if runtime.video_ingestor else project_root / uri
        if not path.is_file():
            local_missing.append(camera.source_id)

    checks = [
        {
            "section": "camera_registry",
            "status": "pass" if enabled else "warn",
            "detail": f"{len(enabled)} enabled camera(s)",
        },
        {
            "section": "deepstream",
            "status": "pass" if video.get("running") else "fail",
            "detail": f"backend={video.get('backend')}, active_sources={len(active)}",
        },
        {
            "section": "local_media",
            "status": "warn" if local_missing else "pass",
            "detail": {"missing_camera_ids": local_missing},
        },
        {
            "section": "fire_smoke_model",
            "status": (
                "pass"
                if runtime.settings.processor_mode == "mock"
                or runtime.settings.fire_model_path.is_file()
                else "fail"
            ),
            "detail": {"minimum_score": runtime.settings.fire_confidence},
        },
        {
            "section": "vehicle_model",
            "status": (
                "pass"
                if runtime.settings.processor_mode == "mock"
                or runtime.settings.vehicle_detector_weights.is_file()
                else "fail"
            ),
            "detail": {"minimum_score": runtime.settings.vehicle_confidence},
        },
        {
            "section": "plate_model",
            "status": (
                "pass"
                if runtime.settings.processor_mode == "mock"
                or runtime.settings.plate_detector_weights.is_file()
                else "fail"
            ),
            "detail": {"minimum_score": runtime.settings.plate_confidence},
        },
        {
            "section": "face_recognition_models",
            "status": (
                "pass"
                if runtime.settings.processor_mode == "mock" or face_models_ready
                else "fail" if face_assigned else "warn"
            ),
            "detail": {
                "assigned_to_enabled_camera": face_assigned,
                "human_model": runtime.settings.face_human_model_path.is_file(),
                "face_model": runtime.settings.face_detector_model_path.is_file(),
                "embedding_model": runtime.settings.face_embedding_model_path.is_file(),
            },
        },
        {
            "section": "persistent_logs",
            "status": "pass",
            "detail": {
                "fire_smoke": status["fire_smoke_logs"]["count"],
                "plates": status["plate_log_count"],
                "humans": status["human_logs"]["count"],
            },
        },
    ]
    return {
        "overall": (
            "fail"
            if any(item["status"] == "fail" for item in checks)
            else "warn"
            if any(item["status"] == "warn" for item in checks)
            else "pass"
        ),
        "checks": checks,
    }


@router.post(
    "/cameras/{camera_id}/restart",
    summary="Restart one DeepStream camera pipeline",
    description=(
        "Schedules a synchronized NULL-state teardown and immediate reopen on "
        "the DeepStream ingestion thread. The application is not restarted."
    ),
)
def restart_camera_pipeline(
    camera_id: str,
    current_user: UserRecord = Depends(require_role("admin")),
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    ingestor = runtime.video_ingestor
    if not isinstance(ingestor, DeepStreamIngestor):
        raise HTTPException(status_code=409, detail="DeepStream ingestion is not active")
    if not ingestor.restart_source(camera_id):
        raise HTTPException(status_code=404, detail="Enabled video camera not found")
    return {"camera_id": camera_id, "restart_scheduled": True}
