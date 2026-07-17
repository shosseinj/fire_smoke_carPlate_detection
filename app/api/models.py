from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.runtime import Runtime


router = APIRouter(prefix="/api/v1/models", tags=["model-management"])


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


class ModelSettingsPatch(BaseModel):
    fire_smoke_model: str | None = Field(
        default=None,
        description="Catalog path for the selected fire/smoke model family.",
    )
    vehicle_detector_model: str | None = Field(
        default=None,
        description="Catalog path for the selected vehicle model family.",
    )
    plate_detector_model: str | None = Field(
        default=None,
        description="Catalog path for the selected plate-detector model family.",
    )
    preferred_format: Literal["engine", "onnx", "pt"] | None = Field(
        default=None,
        description="Runtime preference. Engine resolves to ONNX and then PT when enabled.",
    )
    allow_onnx_fallback: bool | None = None
    allow_pt_fallback: bool | None = None
    export_imgsz: int | None = Field(default=None, ge=32, le=4096)
    export_batch_size: int | None = Field(default=None, ge=1, le=128)
    export_workspace_gb: float | None = Field(default=None, ge=0.25, le=128.0)
    export_half: bool | None = None
    export_dynamic: bool | None = None
    export_timeout_seconds: int | None = Field(default=None, ge=30, le=7200)


class ModelConversionRequest(BaseModel):
    source_model: str = Field(
        description="Relative .pt path returned by GET /api/v1/models/artifacts."
    )
    output_directory: str | None = Field(
        default=None,
        description="Relative directory below MODEL_ROOT_PATH; defaults beside the PT file.",
    )
    imgsz: int | None = Field(default=None, ge=32, le=4096)
    batch: int | None = Field(default=None, ge=1, le=128)
    workspace_gb: float | None = Field(default=None, ge=0.25, le=128.0)
    half: bool | None = None
    dynamic: bool | None = None
    device: str = Field(default="0", min_length=1, max_length=50)
    create_onnx_fallback: bool = Field(
        default=True,
        description="Also export ONNX so it is available when TensorRT cannot run.",
    )
    overwrite: bool = False
    timeout_seconds: int | None = Field(
        default=None,
        ge=30,
        le=7200,
        description="Maximum time for each format before engine falls back to ONNX.",
    )


@router.get(
    "/artifacts",
    summary="List available PT, TensorRT, and ONNX models",
    description=(
        "Discovers installed versions such as nano, tiny, small, medium, and large. "
        "Only versions actually present below MODEL_ROOT_PATH are returned."
    ),
)
def list_model_artifacts(
    role: Literal["fire_smoke", "vehicle_detector", "plate_detector"] | None = None,
    format: Literal["pt", "engine", "onnx"] | None = Query(default=None),
    runtime: Runtime = Depends(get_runtime),
) -> list[dict[str, Any]]:
    return runtime.models.catalog(role=role, model_format=format)


@router.get("/settings", summary="Read selected models and fallback order")
def get_model_settings(runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    return runtime.models.snapshot()


@router.patch(
    "/settings",
    summary="Select models and runtime format",
    description=(
        "Selections are persisted and picked up at the next inference batch. "
        "DeepStream camera pipelines remain running."
    ),
)
def update_model_settings(
    payload: ModelSettingsPatch,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    try:
        return runtime.models.update(payload.model_dump(exclude_unset=True, exclude_none=True))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/conversions",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue PT to TensorRT conversion",
    description=(
        "Runs outside the request and inference threads. By default an ONNX fallback "
        "is exported as well. Poll the returned job ID for completion."
    ),
)
def create_model_conversion(
    payload: ModelConversionRequest,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    defaults = runtime.models.snapshot()
    try:
        return runtime.model_conversions.submit(
            source_model=payload.source_model,
            output_directory=payload.output_directory,
            imgsz=payload.imgsz or defaults["export_imgsz"],
            batch=payload.batch or defaults["export_batch_size"],
            workspace=payload.workspace_gb or defaults["export_workspace_gb"],
            half=(defaults["export_half"] if payload.half is None else payload.half),
            dynamic=(defaults["export_dynamic"] if payload.dynamic is None else payload.dynamic),
            device=payload.device,
            create_onnx_fallback=payload.create_onnx_fallback,
            overwrite=payload.overwrite,
            timeout_seconds=(
                payload.timeout_seconds or defaults["export_timeout_seconds"]
            ),
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/conversions", summary="List model conversion jobs")
def list_model_conversions(
    runtime: Runtime = Depends(get_runtime),
) -> list[dict[str, Any]]:
    return runtime.model_conversions.list()


@router.get("/conversions/{job_id}", summary="Read one conversion job")
def get_model_conversion(
    job_id: str,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    try:
        return runtime.model_conversions.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversion job not found") from exc
