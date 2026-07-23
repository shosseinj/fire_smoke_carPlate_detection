from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from app.core.auth import require_role
from app.core.auth_store import UserRecord
from app.fire_core.policy import FireSmokePolicyConfig
from app.runtime import Runtime

router = APIRouter(prefix="/api/v1", tags=["fire-smoke"])


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


class FireSmokeSettingsUpdate(BaseModel):
    window_seconds: float = Field(ge=0.25, le=300.0, description="مدت پنجره زمانی (ثانیه)")
    low_count: int = Field(ge=1, le=100000, description="تعداد آستانه پایین")
    medium_count: int = Field(ge=2, le=100000, description="تعداد آستانه متوسط")
    high_count: int = Field(ge=3, le=100000, description="تعداد آستانه بالا")

    @model_validator(mode="after")
    def ordered_counts(self) -> "FireSmokeSettingsUpdate":
        if not self.low_count < self.medium_count < self.high_count:
            raise ValueError("تعدادها باید به صورت low_count < medium_count < high_count باشند")
        return self


@router.get("/fire-smoke-logs")
def list_fire_smoke_logs(
    camera_id: str | None = None,
    severity: str | None = Query(default=None, pattern="^(low|medium|high)$"),
    limit: int = Query(default=100, ge=1, le=1000),
    runtime: Runtime = Depends(get_runtime),
) -> list[dict[str, Any]]:
    return runtime.fire_smoke_logs.list(
        camera=camera_id,
        severity=severity,
        limit=limit,
    )


@router.get("/fire-smoke/settings", summary="خواندن تنظیمات تشخیص حریق و دود")
def get_fire_smoke_settings(
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    return runtime.fire_smoke_logs.settings()


@router.put("/fire-smoke/settings", summary="بروزرسانی تنظیمات تشخیص حریق و دود")
def update_fire_smoke_settings(
    payload: FireSmokeSettingsUpdate,
    _: UserRecord = Depends(require_role("admin")),
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    try:
        return runtime.fire_smoke_logs.update_policy(
            FireSmokePolicyConfig(**payload.model_dump())
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
