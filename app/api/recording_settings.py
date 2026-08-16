from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from app.core.auth import require_permission
from app.core.auth_store import UserRecord
from app.core.recording_settings_store import QUALITY_PRESETS, RETENTION_DAYS, SEGMENT_SECONDS
from app.core.recording_source_ref import recording_source_ref, resolve_recording_source, safe_recording_source_name

router = APIRouter(prefix="/api/v1/recording-settings", tags=["recording-settings"])


def get_runtime():
    from app.main import runtime
    return runtime


class RecordingPolicyPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    continuous_enabled: bool | None = None
    quality_preset: str | None = None
    segment_seconds: int | None = None
    retention_days: int | None = None


def _changes(payload: RecordingPolicyPatch) -> dict[str, Any]:
    changes = payload.model_dump(exclude_unset=True, exclude_none=True)
    if "quality_preset" in changes and changes["quality_preset"] not in QUALITY_PRESETS:
        raise HTTPException(status_code=422, detail="invalid recording quality preset")
    if "segment_seconds" in changes and changes["segment_seconds"] not in SEGMENT_SECONDS:
        raise HTTPException(status_code=422, detail="invalid recording segment duration")
    if "retention_days" in changes and changes["retention_days"] not in RETENTION_DAYS:
        raise HTTPException(status_code=422, detail="invalid recording retention period")
    return changes


def _snapshot(runtime, source_uri: str | None = None) -> dict[str, Any]:
    global_policy = runtime.recording_settings.global_policy()
    override = runtime.recording_settings.camera_override(source_uri) if source_uri else None
    effective = override or global_policy
    return {
        "global": global_policy.to_dict(),
        "override": override.to_dict() if override else None,
        "effective": effective.to_dict(),
        "choices": {
            "quality_presets": QUALITY_PRESETS,
            "segment_seconds": sorted(SEGMENT_SECONDS),
            "retention_days": sorted(RETENTION_DAYS),
        },
        "cameras": [
            {"source_ref": recording_source_ref(source.source_uri),
             "source_name": safe_recording_source_name(source.name, source.source_uri)}
            for source in runtime.registry.list()
            if source.enabled and source.source_type == "rtsp"
        ],
    }


@router.get("")
def read_global_settings(
    _: UserRecord = Depends(require_permission("recording_settings.read")), runtime=Depends(get_runtime)
) -> dict[str, Any]:
    return _snapshot(runtime)


@router.patch("")
def update_global_settings(
    payload: RecordingPolicyPatch,
    user: UserRecord = Depends(require_permission("recording_settings.edit")),
    runtime=Depends(get_runtime),
) -> dict[str, Any]:
    runtime.recording_settings.update_global(_changes(payload), user.id)
    runtime.apply_recording_settings()
    return _snapshot(runtime)


@router.post("/reset")
def reset_global_settings(
    user: UserRecord = Depends(require_permission("recording_settings.reset")), runtime=Depends(get_runtime)
) -> dict[str, Any]:
    runtime.recording_settings.reset_global(user.id)
    runtime.apply_recording_settings()
    return _snapshot(runtime)


def _source(runtime, source_ref: str):
    source = resolve_recording_source(runtime.registry, source_ref)
    if source is None:
        raise HTTPException(status_code=404, detail="recording camera not found")
    return source


@router.get("/cameras/{source_ref}")
def read_camera_settings(
    source_ref: str, _: UserRecord = Depends(require_permission("recording_settings.read")), runtime=Depends(get_runtime)
) -> dict[str, Any]:
    source = _source(runtime, source_ref)
    return _snapshot(runtime, source.source_uri)


@router.patch("/cameras/{source_ref}")
def update_camera_settings(
    source_ref: str, payload: RecordingPolicyPatch,
    user: UserRecord = Depends(require_permission("recording_settings.edit")), runtime=Depends(get_runtime),
) -> dict[str, Any]:
    source = _source(runtime, source_ref)
    runtime.recording_settings.update_camera(source.source_uri, _changes(payload), user.id)
    runtime.apply_recording_settings(source.source_uri)
    return _snapshot(runtime, source.source_uri)


@router.delete("/cameras/{source_ref}")
def reset_camera_settings(
    source_ref: str, _: UserRecord = Depends(require_permission("recording_settings.reset")), runtime=Depends(get_runtime)
) -> dict[str, Any]:
    source = _source(runtime, source_ref)
    removed = runtime.recording_settings.reset_camera(source.source_uri)
    runtime.apply_recording_settings(source.source_uri)
    return {"removed": removed, **_snapshot(runtime, source.source_uri)}
