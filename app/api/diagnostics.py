from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.core.deepstream_ingestor import DeepStreamIngestor
from app.core.video_ingestor import VideoFileIngestor
from app.runtime import Runtime

router = APIRouter(
    prefix="/api/v1/diagnostics",
    tags=["system-diagnostics"],
)


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


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
        },
        "fire_smoke_policy": runtime.fire_smoke_logs.settings(),
        "plate_detection_policy": runtime.plate_settings.general(),
        "model_management": runtime.models.snapshot(),
        "fire_smoke_logs": status["fire_smoke_logs"],
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
    local_missing = []
    project_root = runtime.settings.camera_db_path.parents[1]
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
            "section": "persistent_logs",
            "status": "pass",
            "detail": {
                "fire_smoke": status["fire_smoke_logs"]["count"],
                "plates": status["plate_log_count"],
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
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    ingestor = runtime.video_ingestor
    if not isinstance(ingestor, DeepStreamIngestor):
        raise HTTPException(status_code=409, detail="DeepStream ingestion is not active")
    if not ingestor.restart_source(camera_id):
        raise HTTPException(status_code=404, detail="Enabled video camera not found")
    return {"camera_id": camera_id, "restart_scheduled": True}
