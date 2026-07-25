from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from pydantic import BaseModel, Field

from app.core.source_registry import SourceRecord
from app.core.static_video_store import StaticVideoRecord
from app.core.video_ingestor import VIDEO_SUFFIXES
from app.runtime import Runtime

router = APIRouter(prefix="/api/v1/static-videos", tags=["static-videos"])


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


class StaticVideoUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=300)


def _safe_filename(filename: str) -> str:
    """Sanitize a filename, stripping path separators and dangerous characters."""
    cleaned = filename.replace("/", "_").replace("\\", "_")
    cleaned = cleaned.strip(". -")
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


async def _save_upload(
    file: UploadFile,
    runtime: Runtime,
) -> dict[str, Any]:
    """Validate, write, and return the file metadata."""
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

    max_bytes = runtime.settings.max_upload_bytes_per_image * 200  # ~2 GB
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum {max_bytes // (1024*1024)} MB",
        )

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
    }


def _api_response(record: StaticVideoRecord) -> dict[str, Any]:
    return {
        "source_uri": record.source_uri,
        "name": record.name,
        "source_type": record.source_type,
    }


@router.get("")
def list_static_videos(
    runtime: Runtime = Depends(get_runtime),
) -> list[dict[str, Any]]:
    """List all uploaded static video files."""
    return [_api_response(r) for r in runtime.static_video_store.list()]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_static_video(
    file: UploadFile = File(..., description="Video file (.mp4, .avi, .mov, .mkv, .m4v, .webm)"),
    name: str | None = None,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    """Upload a static video file and register it for processing.

    The file is saved to the media store, a ``static_videos`` record is
    created, and a corresponding camera source is registered so the
    ``StaticVideoFileIngestor`` can pick it up.
    """
    upload = await _save_upload(file, runtime)
    source_uri = upload["source_uri"]
    video_name = name or upload["filename"]

    # Create the API-facing static_videos record
    record = runtime.static_video_store.create(
        name=video_name,
        source_uri=source_uri,
    )

    # Create the processing-facing camera source (for the ingestor)
    try:
        runtime.registry.create(
            SourceRecord(
                source_uri=source_uri,
                name=video_name,
                source_type="static_video",
                metadata={"original_filename": upload["filename"]},
            )
        )
    except ValueError:
        pass  # already exists — fine for idempotent re-creation

    return {
        **_api_response(record),
        "file_size_bytes": upload["size_bytes"],
    }


@router.patch("/{source_uri:path}")
async def update_static_video(
    source_uri: str,
    payload: StaticVideoUpdate,
    file: UploadFile | None = None,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    """Update a static video's metadata or replace its file."""
    current = runtime.static_video_store.get(source_uri)
    if current is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Static video not found",
        )

    changes: dict[str, Any] = {}
    if payload.name is not None:
        changes["name"] = payload.name

    old_source_uri = source_uri
    if file is not None:
        upload = await _save_upload(file, runtime)
        changes["source_uri"] = upload["source_uri"]

    if not changes:
        return _api_response(current)

    if "source_uri" in changes:
        # Primary-key change: delete old row, insert new
        record = runtime.static_video_store.create(
            name=changes.get("name", current.name),
            source_uri=changes["source_uri"],
        )
        runtime.static_video_store.delete(old_source_uri)
        # Re-register the camera source under the new path
        try:
            runtime.registry.delete(old_source_uri)
        except KeyError:
            pass
        try:
            runtime.registry.create(
                SourceRecord(
                    source_uri=changes["source_uri"],
                    name=record.name,
                    source_type="static_video",
                )
            )
        except ValueError:
            pass
    else:
        # Name-only update — source_uri unchanged
        record = runtime.static_video_store.update(old_source_uri, **changes)
        if "name" in changes:
            try:
                runtime.registry.update(old_source_uri, name=changes["name"])
            except KeyError:
                pass

    return _api_response(record)


@router.delete("/{source_uri:path}", status_code=status.HTTP_204_NO_CONTENT)
def delete_static_video(
    source_uri: str,
    runtime: Runtime = Depends(get_runtime),
) -> Response:
    """Delete a static video record, deregister its camera source, and remove the file.

    The ingestor will automatically stop processing this source on its next
    loop iteration.
    """
    record = runtime.static_video_store.get(source_uri)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Static video not found",
        )

    # Deregister the camera source from the ingestor
    try:
        runtime.registry.delete(source_uri)
    except KeyError:
        pass

    # Delete the static_videos DB record
    runtime.static_video_store.delete(source_uri)

    # Try to delete the file from disk
    try:
        fpath = Path(source_uri)
        if fpath.is_file():
            fpath.unlink(missing_ok=True)
    except OSError:
        pass  # non-critical — file may be in use or already removed

    return Response(status_code=status.HTTP_204_NO_CONTENT)
