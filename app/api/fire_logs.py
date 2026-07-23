from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.core.auth import get_current_user, require_role
from app.core.auth_store import UserRecord
from app.runtime import Runtime


router = APIRouter(prefix="/api/v1/fire-logs", tags=["fire-logs"])


class FireLogCreate(BaseModel):
    detection_time: datetime
    camera_id: str = Field(min_length=1)
    hazard_type: str
    severity: str = Field(pattern="^(low|medium|high)$")
    confidence: float | None = Field(default=None, ge=0, le=1)
    snapshot_url: str | None = None
    video_url: str | None = None
    thumbnail_url: str | None = None


class FireLogUpdate(BaseModel):
    detection_time: datetime | None = None
    camera_id: str | None = None
    hazard_type: str | None = None
    severity: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    snapshot_url: str | None = None
    video_url: str | None = None
    thumbnail_url: str | None = None


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


@router.get("")
@router.get("/")
def list_fire_logs(
    camera_id: str | None = None,
    severity: str | None = Query(default=None, pattern="^(low|medium|high)$"),
    hazard_type: str | None = Query(default=None, max_length=64),
    detected_from: str | None = Query(default=None),
    detected_to: str | None = Query(default=None),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    _: UserRecord = Depends(get_current_user),
    runtime: Runtime = Depends(get_runtime),
) -> list[dict[str, Any]]:
    return runtime.fire_smoke_logs.list(
        camera=camera_id,
        severity=severity,
        hazard_type=hazard_type,
        detected_from=detected_from,
        detected_to=detected_to,
        limit=skip + limit,
    )[skip:]


@router.get("/{log_id}")
def get_fire_log(log_id: int, _: UserRecord = Depends(get_current_user), runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    value = runtime.fire_smoke_logs.get(log_id)
    if value is None:
        raise HTTPException(status_code=404, detail="Fire or smoke log not found")
    return value


@router.post("", status_code=status.HTTP_201_CREATED)
@router.post("/", status_code=status.HTTP_201_CREATED)
def create_fire_log(payload: FireLogCreate, _: UserRecord = Depends(require_role("admin")), runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    return runtime.fire_smoke_logs.create_manual(payload.model_dump())


@router.patch("/{log_id}")
def update_fire_log(log_id: int, payload: FireLogUpdate, _: UserRecord = Depends(require_role("superuser")), runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    value = runtime.fire_smoke_logs.update_manual(log_id, payload.model_dump(exclude_unset=True))
    if value is None:
        raise HTTPException(status_code=404, detail="Fire or smoke log not found")
    return value


@router.delete("/{log_id}")
def delete_fire_log(log_id: int, _: UserRecord = Depends(require_role("superuser")), runtime: Runtime = Depends(get_runtime)) -> dict[str, str]:
    if not runtime.fire_smoke_logs.delete(log_id):
        raise HTTPException(status_code=404, detail="Fire or smoke log not found")
    return {"message": "Fire or smoke log deleted successfully"}
