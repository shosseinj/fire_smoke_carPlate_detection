from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict

from app.core.auth import effective_permissions, get_current_user
from app.core.auth_store import UserRecord
from app.core.frontend_messages import LocalizedJSONRoute
from app.runtime import Runtime


router = APIRouter(
    prefix="/api/v1/import-progress",
    tags=["import-progress"],
    route_class=LocalizedJSONRoute,
)


class ImportJobAccepted(BaseModel):
    job_id: int
    progress_id: int
    status: str
    status_url: str
    message: str


class ImportResultSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    total_rows: int | None = None
    imported: int | None = None
    skipped: int | None = None
    failed: int | None = None
    successful: int | None = None
    created: int | None = None
    updated: int | None = None


class ImportResultReport(BaseModel):
    model_config = ConfigDict(extra="allow")

    success: bool
    filename: str | None = None
    message: str | None = None
    database_changed: bool | None = None
    summary: ImportResultSummary | None = None
    successful_rows: list[dict[str, Any]] | None = None
    successful_row_details: list[dict[str, Any]] | None = None
    skipped_rows: list[dict[str, Any]] | int | None = None
    skipped_row_details: list[dict[str, Any]] | None = None
    failed_rows: list[dict[str, Any]] | int | None = None
    failed_row_details: list[dict[str, Any]] | None = None
    file_errors: list[dict[str, Any]] | None = None
    errors: list[dict[str, Any]] | None = None


class ImportProgressResponse(BaseModel):
    id: int
    import_type: str
    source_filename: str
    total_rows: int
    imported_rows: int
    skipped_rows: int
    failed_rows: int
    status: str
    error_message: str | None
    result: ImportResultReport | None
    processed_rows: int
    progress_percent: float
    created_by: int | None
    created_at: str
    updated_at: str


def get_runtime() -> Runtime:
    from app.main import runtime
    return runtime


@router.post("")
@router.post("/")
def create_import_progress(
    import_type: str = Query(..., description="نوع ورود اطلاعات"),
    source_filename: str = Query(..., max_length=255, description="نام فایل مبدأ"),
    current_user: UserRecord = Depends(get_current_user),
    runtime: Runtime = Depends(get_runtime),
) -> dict:
    record = runtime.import_progress.create(
        import_type=import_type,
        source_filename=source_filename,
        created_by=current_user.id,
    )
    return record.to_dict()


@router.get("/{progress_id}", response_model=ImportProgressResponse)
def get_import_progress(
    progress_id: int,
    current_user: UserRecord = Depends(get_current_user),
    runtime: Runtime = Depends(get_runtime),
) -> dict:
    record = runtime.import_progress.get(progress_id)
    if record is None:
        raise HTTPException(status_code=404, detail="رکورد پیشرفت ورود اطلاعات یافت نشد")
    permissions = effective_permissions(current_user)
    if record.created_by != current_user.id and not {"import_progress.read", "*"} & permissions:
        raise HTTPException(status_code=403, detail="شما فقط می‌توانید رکوردهای خود را مشاهده کنید")
    return record.to_dict()


@router.delete("/{progress_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_import_progress(
    progress_id: int,
    current_user: UserRecord = Depends(get_current_user),
    runtime: Runtime = Depends(get_runtime),
) -> None:
    record = runtime.import_progress.get(progress_id)
    if record is None:
        raise HTTPException(status_code=404, detail="رکورد پیشرفت ورود اطلاعات یافت نشد")
    permissions = effective_permissions(current_user)
    if record.created_by != current_user.id and not {"import_progress.read", "*"} & permissions:
        raise HTTPException(status_code=403, detail="شما فقط می‌توانید رکوردهای خود را حذف کنید")
    runtime.import_progress.delete(progress_id)
