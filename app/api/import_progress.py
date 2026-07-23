from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.auth import get_current_user, require_role
from app.core.auth_store import UserRecord
from app.runtime import Runtime


router = APIRouter(prefix="/api/v1/import-progress", tags=["import-progress"])


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


@router.get("/{progress_id}")
def get_import_progress(
    progress_id: int,
    current_user: UserRecord = Depends(get_current_user),
    runtime: Runtime = Depends(get_runtime),
) -> dict:
    record = runtime.import_progress.get(progress_id)
    if record is None:
        raise HTTPException(status_code=404, detail="رکورد پیشرفت ورود اطلاعات یافت نشد")
    if record.created_by != current_user.id and current_user.role not in ("admin", "superuser"):
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
    if record.created_by != current_user.id and current_user.role not in ("admin", "superuser"):
        raise HTTPException(status_code=403, detail="شما فقط می‌توانید رکوردهای خود را حذف کنید")
    runtime.import_progress.delete(progress_id)
