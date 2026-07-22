from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from app.core.auth import get_current_user, require_role
from app.core.auth_store import UserRecord
from app.core.general_settings_store import GeneralSettingsStore
from app.runtime import Runtime

router = APIRouter(prefix="/api/v1/general-settings", tags=["general-settings"])


def get_runtime() -> Runtime:
    from app.main import runtime
    return runtime


def get_store(runtime: Runtime = Depends(get_runtime)) -> GeneralSettingsStore:
    return runtime.general_settings


def _normalize_role(current_user: UserRecord) -> str:
    from app.core.auth import normalize_role
    return normalize_role(current_user.role)


# ── Legacy fields schema ───────────────────────────────────────────────

_LEGACY_BOOL_FIELDS = {
    "enable_processing", "process_fire", "process_plate",
    "counts_for_attendance", "draw_box", "draw_face",
    "draw_skeleton", "draw_zones",
}

_LEGACY_FLOAT_FIELDS = {
    "margin_level", "face_rec_score", "face_det_score",
    "human_det_score", "confirmation_threshold",
}


class LegacyGeneralSettingsPatch(BaseModel):
    enable_processing: bool | None = None
    process_fire: bool | None = None
    process_plate: bool | None = None
    counts_for_attendance: bool | None = None
    margin_level: float | None = Field(default=None, ge=1.0, le=5.0)
    draw_box: bool | None = None
    draw_face: bool | None = None
    draw_skeleton: bool | None = None
    draw_zones: bool | None = None
    face_rec_score: float | None = Field(default=None, ge=0.0, le=1.0)
    face_det_score: float | None = Field(default=None, ge=0.0, le=1.0)
    human_det_score: float | None = Field(default=None, ge=0.0, le=1.0)
    confirmation_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def reject_null_and_unknown(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        for key, value in list(values.items()):
            if value is None and key in _LEGACY_BOOL_FIELDS | _LEGACY_FLOAT_FIELDS:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={"field": key, "message": "Explicit null is not allowed"},
                )
        return values

    @model_validator(mode="after")
    def validate_threshold(self) -> LegacyGeneralSettingsPatch:
        face_rec = self.face_rec_score
        threshold = self.confirmation_threshold
        if face_rec is not None and threshold is not None:
            if threshold < face_rec:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "field": "confirmation_threshold",
                        "message": "confirmation_threshold must be >= face_rec_score",
                    },
                )
        return self


def _store_to_response(record: Any) -> dict[str, Any]:
    return record.to_dict()


# ── GET (authenticated) ────────────────────────────────────────────────


@router.get(
    "",
    summary="Get legacy general settings",
    description="Returns the singleton legacy general settings object. Authenticated users only.",
)
def get_legacy_settings(
    current_user: UserRecord = Depends(get_current_user),
    store: GeneralSettingsStore = Depends(get_store),
) -> dict[str, Any]:
    record = store.get()
    return _store_to_response(record)


# ── PATCH (admin only) ─────────────────────────────────────────────────


def _apply_runtime_settings(runtime: Runtime, settings: dict[str, Any]) -> None:
    pass


@router.patch(
    "",
    summary="Update legacy general settings",
    description="Partially updates legacy general settings. Admin/superuser only.",
)
def patch_legacy_settings(
    payload: LegacyGeneralSettingsPatch,
    current_user: UserRecord = Depends(require_role("admin")),
    runtime: Runtime = Depends(get_runtime),
    store: GeneralSettingsStore = Depends(get_store),
) -> dict[str, Any]:
    changes = payload.model_dump(exclude_unset=True, exclude_none=True)

    current_record = store.get()

    face_rec = changes.get("face_rec_score")
    threshold = changes.get("confirmation_threshold")

    if face_rec is not None and threshold is None:
        current_threshold = current_record.confirmation_threshold
        if current_threshold < face_rec:
            changes["confirmation_threshold"] = face_rec

    if face_rec is not None and threshold is not None:
        if threshold < face_rec:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "field": "confirmation_threshold",
                    "message": "confirmation_threshold must be >= face_rec_score",
                },
            )

    updated = store.update(changes, updated_by=current_user.id)

    return _store_to_response(updated)


# ── RESET (admin only) ─────────────────────────────────────────────────


@router.post(
    "/reset",
    summary="Reset legacy general settings to defaults",
    description="Resets all legacy general settings to defaults. Admin/superuser only.",
)
def reset_legacy_settings(
    current_user: UserRecord = Depends(require_role("admin")),
    store: GeneralSettingsStore = Depends(get_store),
) -> dict[str, Any]:
    updated = store.reset(updated_by=current_user.id)
    return _store_to_response(updated)
