from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated, Protocol

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.auth import get_current_user
from app.core.auth_store import UserRecord
from app.core.frontend_messages import LocalizedJSONRoute
from app.core.recording_job_store import (
    ALL_STATUSES,
    InvalidRecordingTransition,
    RecordingJob,
    RecordingJobStore,
    RecordingOverlapError,
)
from app.core.recording_scheduler import RecordingScheduler
from app.core.recording_source_ref import (
    redacted_recording_source,
    recording_source_ref,
    resolve_recording_source,
    safe_recording_source_name,
)


router = APIRouter(prefix="/api/v1/recordings", tags=["recordings"], route_class=LocalizedJSONRoute)


class RecordingContentService(Protocol):
    def presigned_download(self, job_uuid: str, expires: timedelta = timedelta(minutes=15)) -> str: ...
    def delete(self, job_uuid: str) -> None: ...


class RecordingCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_ref: str = Field(min_length=1, max_length=128)
    scheduled_start_utc: datetime
    scheduled_end_utc: datetime

    @field_validator("scheduled_start_utc", "scheduled_end_utc")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("زمان باید شامل منطقه زمانی باشد")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def valid_interval(self) -> "RecordingCreate":
        if self.scheduled_end_utc <= self.scheduled_start_utc:
            raise ValueError("زمان پایان باید بعد از زمان شروع باشد")
        if self.scheduled_end_utc - self.scheduled_start_utc > timedelta(hours=2):
            raise ValueError("حداکثر مدت ضبط دو ساعت است")
        return self


class RecordingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    source_ref: str
    source_name: str
    source_display: str
    created_by: int | None
    status: str
    scheduled_start_utc: datetime
    scheduled_end_utc: datetime
    started_at_utc: datetime | None
    finished_at_utc: datetime | None
    warning: str | None
    error: str | None
    is_partial: bool
    size_bytes: int | None
    attempt_count: int
    object_expires_at_utc: datetime | None
    created_at_utc: datetime | None
    updated_at_utc: datetime | None
    has_content: bool


def get_runtime():
    from app.main import runtime

    return runtime


def get_recording_store(runtime=Depends(get_runtime)) -> RecordingJobStore:
    return RecordingJobStore(runtime.database)


def get_recording_scheduler(
    store: Annotated[RecordingJobStore, Depends(get_recording_store)],
    runtime=Depends(get_runtime),
) -> RecordingScheduler:
    coordinator = getattr(runtime, "recording_coordinator", None)
    if coordinator is not None:
        return coordinator.scheduler
    return RecordingScheduler(store, getattr(runtime, "recording_redis", None))


def get_recording_content_service(runtime=Depends(get_runtime)) -> RecordingContentService:
    service = getattr(runtime, "recording_storage", None)
    if service is None:
        raise HTTPException(status_code=503, detail="سرویس ذخیره‌سازی ضبط در دسترس نیست")
    return service


def get_optional_recording_content_service(runtime=Depends(get_runtime)) -> RecordingContentService | None:
    return getattr(runtime, "recording_storage", None)


class RecordingSourceOption(BaseModel):
    source_ref: str
    source_name: str
    source_display: str


def _response(job: RecordingJob, runtime) -> RecordingResponse:
    source = runtime.registry.get(job.source_uri)
    values = {name: getattr(job, name) for name in RecordingJob.__dataclass_fields__ if name != "source_uri"}
    values.update({
        "source_ref": recording_source_ref(job.source_uri),
        "source_name": safe_recording_source_name(source.name if source is not None else "", job.source_uri),
        "source_display": redacted_recording_source(job.source_uri),
        "has_content": job.has_content,
    })
    return RecordingResponse.model_validate(values)


def _require(store: RecordingJobStore, job_id: str) -> RecordingJob:
    try:
        return store.require(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="درخواست ضبط یافت نشد") from exc


@router.post("", response_model=RecordingResponse, status_code=status.HTTP_201_CREATED)
def create_recording(
    payload: RecordingCreate,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    store: Annotated[RecordingJobStore, Depends(get_recording_store)],
    scheduler: Annotated[RecordingScheduler, Depends(get_recording_scheduler)],
    response: Response,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key", max_length=255)] = None,
    runtime=Depends(get_runtime),
) -> RecordingResponse:
    coordinator = getattr(runtime, "recording_coordinator", None)
    if not runtime.settings.recording_enabled or coordinator is None:
        raise HTTPException(status_code=503, detail="سرویس ضبط زمان‌بندی‌شده در دسترس نیست")
    try:
        source = resolve_recording_source(runtime.registry, payload.source_ref)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail="شناسه منبع ضبط یکتا نیست") from exc
    if source is None:
        raise HTTPException(status_code=422, detail="منبع ضبط یافت نشد")
    if not source.enabled or source.source_type != "rtsp":
        raise HTTPException(status_code=422, detail="منبع ضبط باید فعال و از نوع RTSP باشد")
    if runtime.live_branch is None or not runtime.live_branch.enabled or not runtime.live_branch.has_source(source.source_uri):
        raise HTTPException(status_code=409, detail="شاخه زنده منبع برای ضبط آماده نیست")
    if coordinator.spool.status().high_water_exceeded:
        raise HTTPException(status_code=507, detail="فضای موقت ضبط از حد مجاز عبور کرده است")
    try:
        job, created = store.create(
            source_uri=source.source_uri,
            scheduled_start_utc=payload.scheduled_start_utc,
            scheduled_end_utc=payload.scheduled_end_utc,
            created_by=current_user.id,
            idempotency_key=idempotency_key,
        )
    except RecordingOverlapError as exc:
        raise HTTPException(status_code=409, detail="این بازه با ضبط فعال دیگری برای همین منبع هم‌پوشانی دارد") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if created:
        try:
            scheduler.schedule_job(job)
        except Exception:
            coordinator.request_reconcile()
            response.status_code = status.HTTP_202_ACCEPTED
    else:
        response.status_code = status.HTTP_200_OK
    return _response(job, runtime)


@router.get("/sources", response_model=list[RecordingSourceOption])
def recording_source_options(
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    runtime=Depends(get_runtime),
) -> list[RecordingSourceOption]:
    del current_user
    live_branch = runtime.live_branch
    if not runtime.settings.recording_enabled or live_branch is None or not live_branch.enabled:
        return []
    return [
        RecordingSourceOption(
            source_ref=recording_source_ref(source.source_uri),
            source_name=safe_recording_source_name(source.name, source.source_uri),
            source_display=redacted_recording_source(source.source_uri),
        )
        for source in runtime.registry.list()
        if source.enabled and source.source_type == "rtsp" and live_branch.has_source(source.source_uri)
    ]


@router.get("", response_model=list[RecordingResponse])
def list_recordings(
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    store: Annotated[RecordingJobStore, Depends(get_recording_store)],
    recording_status: Annotated[str | None, Query(alias="status")] = None,
    source_ref: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    runtime=Depends(get_runtime),
) -> list[RecordingResponse]:
    del current_user
    if recording_status is not None and recording_status not in ALL_STATUSES:
        raise HTTPException(status_code=422, detail="وضعیت ضبط نامعتبر است")
    resolved_uri = None
    if source_ref is not None:
        source = resolve_recording_source(runtime.registry, source_ref)
        if source is None:
            raise HTTPException(status_code=422, detail="شناسه منبع ضبط نامعتبر است")
        resolved_uri = source.source_uri
    return [_response(job, runtime) for job in store.list(status=recording_status, source_uri=resolved_uri, limit=limit, offset=offset)]


@router.get("/{job_id}", response_model=RecordingResponse)
def get_recording(
    job_id: str,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    store: Annotated[RecordingJobStore, Depends(get_recording_store)],
    runtime=Depends(get_runtime),
) -> RecordingResponse:
    del current_user
    return _response(_require(store, job_id), runtime)


@router.post("/{job_id}/cancel", response_model=RecordingResponse)
def cancel_recording(
    job_id: str,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    store: Annotated[RecordingJobStore, Depends(get_recording_store)],
    runtime=Depends(get_runtime),
) -> RecordingResponse:
    del current_user
    try:
        coordinator = getattr(runtime, "recording_coordinator", None)
        return _response(coordinator.cancel(job_id) if coordinator is not None else store.cancel(job_id), runtime)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="درخواست ضبط یافت نشد") from exc


@router.get("/{job_id}/download", response_class=RedirectResponse)
@router.get("/{job_id}/content", response_class=RedirectResponse, include_in_schema=False)
def recording_content(
    job_id: str,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    store: Annotated[RecordingJobStore, Depends(get_recording_store)],
    content: Annotated[RecordingContentService | None, Depends(get_optional_recording_content_service)],
) -> RedirectResponse:
    del current_user
    job = _require(store, job_id)
    if not job.has_content:
        raise HTTPException(status_code=409, detail="محتوای قابل دریافت برای این ضبط آماده نیست")
    try:
        url = content.presigned_download(job.id, expires=timedelta(minutes=15))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="دریافت محتوای ضبط ممکن نیست") from exc
    return RedirectResponse(url=url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_recording(
    job_id: str,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    store: Annotated[RecordingJobStore, Depends(get_recording_store)],
    content: Annotated[RecordingContentService, Depends(get_recording_content_service)],
) -> Response:
    del current_user
    job = _require(store, job_id)
    try:
        if job.object_key and content is None:
            raise HTTPException(status_code=503, detail="سرویس ذخیره‌سازی ضبط در دسترس نیست")
        if job.object_key and content is not None:
            content.delete(job.id)
        store.delete(job.id)
    except InvalidRecordingTransition as exc:
        raise HTTPException(status_code=409, detail="درخواست ضبط فعال قابل حذف نیست") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="حذف محتوای ضبط کامل نشد") from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
