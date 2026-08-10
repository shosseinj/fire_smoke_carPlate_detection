from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from app.core.auth import require_role
from app.core.auth_store import UserRecord
from app.core.common_schemas import UserBrief, resolve_user_brief
from app.core.employee_type_store import EmployeeTypeRecord, EmployeeTypeStore
from app.core.frontend_messages import LocalizedJSONRoute
from app.core.jalali_utils import utc_iso_to_jalali_datetime
from app.runtime import Runtime


router = APIRouter(
    prefix="/api/v1/employee-types",
    tags=["Employee Types"],
    route_class=LocalizedJSONRoute,
)


def get_runtime() -> Runtime:
    from app.main import runtime
    return runtime


def _store(runtime: Runtime) -> EmployeeTypeStore:
    return runtime.employee_type_store


class EmployeeTypeCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None
    is_active: bool = True
    include_in_attendance_reports: bool = False


class EmployeeTypeUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None
    is_active: bool | None = None
    include_in_attendance_reports: bool | None = None


class EmployeeTypeResponse(BaseModel):
    id: int
    name: str
    description: str | None = None
    is_active: bool
    include_in_attendance_reports: bool
    personnel_count: int = 0
    created_at: str
    updated_at: str
    created_at_jalali: str = ""
    updated_at_jalali: str | None = None
    created_by: UserBrief | None = None
    updated_by: UserBrief | None = None


def _response(record: EmployeeTypeRecord, store: EmployeeTypeStore) -> EmployeeTypeResponse:
    created_by = updated_by = None
    if record.created_by is not None or record.updated_by is not None:
        with store._connection() as conn:
            created_by = resolve_user_brief(record.created_by, conn)
            updated_by = resolve_user_brief(record.updated_by, conn)
    return EmployeeTypeResponse(
        id=record.id,
        name=record.name,
        description=record.description,
        is_active=record.is_active,
        include_in_attendance_reports=record.include_in_attendance_reports,
        personnel_count=store.personnel_count(record.id),
        created_at=record.created_at_utc,
        updated_at=record.updated_at_utc,
        created_at_jalali=utc_iso_to_jalali_datetime(record.created_at_utc) or "",
        updated_at_jalali=utc_iso_to_jalali_datetime(record.updated_at_utc),
        created_by=created_by,
        updated_by=updated_by,
    )


@router.get("/", response_model=list[EmployeeTypeResponse])
def list_employee_types(
    is_active: bool | None = Query(default=None),
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> list[EmployeeTypeResponse]:
    store = _store(runtime)
    return [_response(record, store) for record in store.list(is_active=is_active)]


@router.get("/{employee_type_id}", response_model=EmployeeTypeResponse)
def get_employee_type(
    employee_type_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> EmployeeTypeResponse:
    store = _store(runtime)
    record = store.get(employee_type_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "نوع استخدام یافت نشد")
    return _response(record, store)


@router.post("/", response_model=EmployeeTypeResponse, status_code=status.HTTP_201_CREATED)
def create_employee_type(
    payload: EmployeeTypeCreate,
    runtime: Runtime = Depends(get_runtime),
    current_user: UserRecord = Depends(require_role("admin")),
) -> EmployeeTypeResponse:
    store = _store(runtime)
    try:
        record = store.create(
            name=payload.name,
            description=payload.description,
            is_active=payload.is_active,
            include_in_attendance_reports=payload.include_in_attendance_reports,
            created_by=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    return _response(record, store)


@router.put("/{employee_type_id}", response_model=EmployeeTypeResponse)
def update_employee_type(
    employee_type_id: int,
    payload: EmployeeTypeUpdate,
    runtime: Runtime = Depends(get_runtime),
    current_user: UserRecord = Depends(require_role("admin")),
) -> EmployeeTypeResponse:
    store = _store(runtime)
    try:
        record = store.update(
            employee_type_id,
            name=payload.name,
            description=payload.description,
            is_active=payload.is_active,
            include_in_attendance_reports=payload.include_in_attendance_reports,
            updated_by=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "نوع استخدام یافت نشد")
    return _response(record, store)


@router.delete("/{employee_type_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_employee_type(
    employee_type_id: int,
    runtime: Runtime = Depends(get_runtime),
    _: UserRecord = Depends(require_role("admin")),
) -> Response:
    store = _store(runtime)
    try:
        deleted = store.delete(employee_type_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "نوع استخدام یافت نشد")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
