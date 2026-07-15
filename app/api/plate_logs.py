from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query, status, HTTPException
from pydantic import BaseModel, Field, field_validator


from app.runtime import Runtime

router = APIRouter(prefix="/api/v1/plate-logs", tags=["plate-logs"])


class PlateLogCreate(BaseModel):
    camera_id: str = Field(min_length=1, max_length=200)
    time: datetime
    plate: str = Field(min_length=1, max_length=100)

    @field_validator("camera_id", "plate")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be blank")
        return value


class PlateLogResponse(BaseModel):
    camera: str
    time: datetime
    plate: str


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


@router.post(
    "",
    response_model=PlateLogResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_plate_log(
    payload: PlateLogCreate,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, str]:
    from pathlib import Path
    from uuid import uuid4

    detected_time = payload.time
    timestamp = detected_time.strftime("%Y%m%d_%H%M%S_%f")

    safe_camera_id = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(payload.camera_id)
    )

    file_name = (
        f"{safe_camera_id}_{timestamp}_{uuid4().hex[:8]}.jpg"
    )

    snapshot_directory = Path("saved_media") / "plate_snapshots"
    snapshot_directory.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_directory / file_name
    snapshot_url = f"/media/plate_snapshots/{file_name}"

    # snapshot_saved = model.save_snapshot(str(snapshot_path))
    snapshot_saved = True
    if not snapshot_saved:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Plate snapshot could not be saved.",
        )

    try:
        return runtime.plate_logs.insert(
            camera=payload.camera_id,
            time=_utc_iso(payload.time),
            plate=payload.plate,
            snapshot_url=snapshot_url,
        )
    except Exception:
        # Remove the image if database insertion fails.
        snapshot_path.unlink(missing_ok=True)
        raise


@router.get("", response_model=list[PlateLogResponse])
def list_plate_logs(
    camera_id: str | None = None,
    plate: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
    runtime: Runtime = Depends(get_runtime),
) -> list[dict[str, str]]:
    return runtime.plate_logs.list(camera=camera_id, plate=plate, limit=limit)
