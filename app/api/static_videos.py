from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.source_registry import SourceRecord
from app.core.video_ingestor import VIDEO_SUFFIXES
from app.runtime import Runtime

router = APIRouter(prefix="/api/v1/static-videos", tags=["static-videos"])


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


def _safe_filename(filename: str) -> str:
    """Sanitize a filename, stripping path separators and dangerous characters."""
    # Remove path separators
    cleaned = filename.replace("/", "_").replace("\\", "_")
    # Strip any leading dots, spaces, or dashes
    cleaned = cleaned.strip(". -")
    # Keep only the last 200 chars to avoid absurdly long names
    if len(cleaned) > 200:
        stem, ext = Path(cleaned).stem, Path(cleaned).suffix
        cleaned = stem[: 200 - len(ext) - 1] + ext
    return cleaned if cleaned else "upload"


def _validate_video(filename: str, content_type: str | None) -> str | None:
    """Check that the filename has a supported video extension.

    Returns the normalized suffix (e.g. '.mp4') or None if unsupported.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        return None
    return suffix


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_static_video(
    file: UploadFile = File(..., description="Video file (.mp4, .avi, .mov, .mkv, .m4v, .webm)"),
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    """Upload a static video file and save it to the media store.

    Returns the file metadata including the server-side path (``source_uri``)
    that can be passed to ``POST /api/v1/cameras`` with ``source_type=static_video``.
    """
    # Validate extension
    suffix = _validate_video(file.filename or "", file.content_type)
    if suffix is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported video format. Allowed: {', '.join(sorted(VIDEO_SUFFIXES))}",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded file is empty",
        )

    # Size check (2 GB limit for video files)
    max_bytes = runtime.settings.max_upload_bytes_per_image * 200  # ~2 GB
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum {max_bytes // (1024*1024)} MB",
        )

    # Build safe path
    upload_dir: Path = runtime.settings.static_video_upload_path.resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_filename(file.filename or f"video{suffix}")
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    dest = upload_dir / unique_name

    try:
        dest.write_bytes(raw)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {exc}",
        ) from exc

    return {
        "file_id": unique_name,
        "filename": safe_name,
        "source_uri": str(dest),
        "size_bytes": len(raw),
        "content_type": file.content_type or "application/octet-stream",
    }


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_static_video_source(
    file: UploadFile = File(..., description="Video file (.mp4, .avi, .mov, .mkv)"),
    camera_id: str | None = None,
    name: str | None = None,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    """Upload a static video and immediately create a camera source for it.

    This is a convenience endpoint that combines upload + camera creation.
    The camera is created with ``source_type=static_video`` and the uploaded
    file path as ``source_uri``. The ``StaticVideoFileIngestor`` will
    automatically pick it up for processing.
    """
    upload = await upload_static_video(file=file, runtime=runtime)
    source_uri = upload["source_uri"]

    source_id = camera_id or f"static_{uuid.uuid4().hex[:12]}"
    source_name = name or upload["filename"]

    try:
        record = runtime.registry.create(
            SourceRecord(
                source_id=source_id,
                name=source_name,
                enabled=True,
                tasks=set(),  # play-only by default; user can add tasks via PATCH
                source_uri=source_uri,
                source_type="static_video",
                metadata={"original_filename": upload["filename"]},
            )
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return {
        "camera_id": record.source_id,
        "name": record.name,
        "source_uri": source_uri,
        "file_id": upload["file_id"],
        "size_bytes": upload["size_bytes"],
    }


@router.get("")
def list_static_video_sources(
    runtime: Runtime = Depends(get_runtime),
) -> list[dict[str, Any]]:
    """List all static video camera sources."""
    sources = runtime.registry.list_by_type("static_video")
    return [
        {
            "camera_id": s.source_id,
            "name": s.name,
            "enabled": s.enabled,
            "source_uri": s.source_uri,
            "tasks": sorted(t.value for t in s.tasks),
            "frame_width": s.frame_width,
            "frame_height": s.frame_height,
        }
        for s in sources
    ]
