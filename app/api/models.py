from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.runtime import Runtime


router = APIRouter(prefix="/api/v1/models", tags=["model-management"])


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


class ModelSettingsPatch(BaseModel):
    fire_smoke_model: str | None = Field(
        default=None,
        description=(
            "Exact catalog file for fire/smoke. Selecting .engine or .onnx keeps "
            "that format first for this model."
        ),
    )
    vehicle_detector_model: str | None = Field(
        default=None,
        description=(
            "Exact catalog file for vehicle detection. Selecting .engine or .onnx "
            "keeps that format first for this model."
        ),
    )
    plate_detector_model: str | None = Field(
        default=None,
        description=(
            "Exact catalog file for plate detection. Selecting .engine or .onnx "
            "keeps that format first for this model."
        ),
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
    select_when_ready: bool = Field(
        default=True,
        description=(
            "Persist the created engine, or successful ONNX fallback, as this "
            "model role's exact selection in general settings."
        ),
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


@router.get(
    "/artifacts/content",
    response_class=FileResponse,
    summary="Download a model artifact",
    description="Use a path or URL returned by the catalog, export job, or general settings.",
)
def download_model_artifact(
    path: str = Query(description="Relative artifact path below MODEL_ROOT_PATH."),
    runtime: Runtime = Depends(get_runtime),
) -> FileResponse:
    try:
        artifact = runtime.models.resolve_path(path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if (
        not artifact.is_file()
        or artifact.suffix.lower().lstrip(".") not in {"pt", "onnx", "engine"}
    ):
        raise HTTPException(status_code=404, detail="Model artifact not found")
    return FileResponse(
        artifact,
        filename=artifact.name,
        media_type="application/octet-stream",
    )


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
            select_when_ready=payload.select_when_ready,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/engine-exports",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a PT file and create a TensorRT engine",
    description=(
        "Upload a .pt model directly from Swagger. The export runs in the background. "
        "When select_when_ready is true, the generated engine (or ONNX fallback) is "
        "saved as that model role's exact selection in general settings. Poll status_url."
    ),
)
async def create_engine_from_upload(
    file: UploadFile = File(
        description="PyTorch/YOLO weights. Only a non-empty .pt file is accepted."
    ),
    role: Literal["fire_smoke", "vehicle_detector", "plate_detector"] = Form(
        description="Which runtime model should use the generated artifact."
    ),
    output_name: str | None = Form(
        default=None,
        description="Output filename without extension; defaults to the uploaded PT name.",
    ),
    output_directory: str | None = Form(
        default=None,
        description=(
            "Optional directory below MODEL_ROOT_PATH and inside the selected role; "
            "for example vehicle_detector/exports."
        ),
    ),
    imgsz: int | None = Form(default=None, ge=32, le=4096),
    batch: int | None = Form(default=None, ge=1, le=128),
    workspace_gb: float | None = Form(default=None, ge=0.25, le=128.0),
    half: bool | None = Form(default=None),
    dynamic: bool | None = Form(default=None),
    device: str = Form(default="0", min_length=1, max_length=50),
    create_onnx_fallback: bool = Form(
        default=True,
        description="Create ONNX too, so it can be selected if TensorRT export fails.",
    ),
    select_when_ready: bool = Form(
        default=True,
        description="Persist the created engine/ONNX URL as this role's selected model.",
    ),
    overwrite: bool = Form(default=False),
    timeout_seconds: int | None = Form(default=None, ge=30, le=7200),
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    defaults = runtime.models.snapshot()
    uploaded_path: str | None = None
    try:
        uploaded_path = await run_in_threadpool(
            runtime.model_conversions.stage_uploaded_pt,
            file.file,
            file.filename,
            role=role,
            output_directory=output_directory,
            output_name=output_name,
            overwrite=overwrite,
        )
        return runtime.model_conversions.submit(
            source_model=uploaded_path,
            output_directory=(
                str(Path(uploaded_path).parent).replace("\\", "/")
                if output_directory is None
                else output_directory
            ),
            imgsz=imgsz or defaults["export_imgsz"],
            batch=batch or defaults["export_batch_size"],
            workspace=workspace_gb or defaults["export_workspace_gb"],
            half=(defaults["export_half"] if half is None else half),
            dynamic=(defaults["export_dynamic"] if dynamic is None else dynamic),
            device=device,
            create_onnx_fallback=create_onnx_fallback,
            overwrite=overwrite,
            timeout_seconds=timeout_seconds or defaults["export_timeout_seconds"],
            select_when_ready=select_when_ready,
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()


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
