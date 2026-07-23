from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.api.models import ModelConversionRequest
from app.core.auth import require_role
from app.core.auth_store import UserRecord
from app.runtime import Runtime


router = APIRouter(prefix="/api/v1/model-exports", tags=["legacy-model-exports"])


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


@router.get("")
@router.get("/")
def list_legacy_model_exports(
    _: UserRecord = Depends(require_role("admin")),
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    items = runtime.model_conversions.list()
    return {"total": len(items), "items": items}


@router.post("", status_code=status.HTTP_202_ACCEPTED)
@router.post("/", status_code=status.HTTP_202_ACCEPTED)
def create_legacy_model_export(
    payload: ModelConversionRequest,
    _: UserRecord = Depends(require_role("admin")),
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    defaults = runtime.models.snapshot()
    try:
        result = runtime.model_conversions.submit(
            source_model=payload.source_model,
            output_directory=payload.output_directory,
            imgsz=payload.imgsz or defaults["export_imgsz"],
            batch=payload.batch or defaults["export_batch_size"],
            workspace=payload.workspace_gb or defaults["export_workspace_gb"],
            half=defaults["export_half"] if payload.half is None else payload.half,
            dynamic=defaults["export_dynamic"] if payload.dynamic is None else payload.dynamic,
            device=payload.device,
            create_onnx_fallback=payload.create_onnx_fallback,
            overwrite=payload.overwrite,
            timeout_seconds=payload.timeout_seconds or defaults["export_timeout_seconds"],
            select_when_ready=payload.select_when_ready,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["status_url"] = f"/api/v1/model-exports/{result['job_id']}"
    return result


@router.get("/{job_id}")
def get_legacy_model_export(
    job_id: str,
    _: UserRecord = Depends(require_role("admin")),
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    try:
        result = runtime.model_conversions.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Export job not found") from exc
    result["status_url"] = f"/api/v1/model-exports/{job_id}"
    return result


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_legacy_model_export(
    job_id: str,
    _: UserRecord = Depends(require_role("admin")),
    runtime: Runtime = Depends(get_runtime),
) -> Response:
    try:
        deleted = runtime.model_conversions.delete(job_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Export job not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
