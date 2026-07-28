from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import cv2
from fastapi import APIRouter, Depends, HTTPException, status

from app.config import settings
from app.core.auth import get_current_user
from app.core.detection_log_store import DetectionLogStore
from app.core.detection_media import DetectionMediaStorage, InvalidMediaKey, MEDIA_STATUS_READY
from app.core.auth_store import UserRecord


router = APIRouter(prefix="/api/v1", tags=["Detection Logs"])


def get_runtime() -> Any:
    from app.main import runtime

    return runtime


def get_detection_log_store() -> DetectionLogStore:
    return get_runtime().detection_log_store


def _resolve_saved_video_path(video_path: str) -> Path:
    try:
        resolved = DetectionMediaStorage(settings.saved_media_path).resolve(video_path)
    except InvalidMediaKey as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="مسیر ویدیوی ذخیره‌شده معتبر نیست",
        ) from exc
    if resolved is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="مسیر ویدیوی ذخیره‌شده معتبر نیست",
        )
    return resolved


def _extract_video_frames(
    video_path: Path,
    detection_id: int,
    frame_interval: int,
    max_frames: int | None,
    public_video_path: str,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="ویدیوی ذخیره‌شده قابل خواندن نیست",
        )

    frames: list[dict[str, Any]] = []
    frame_count = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_index = frame_count
            frame_count += 1
            if frame_index % frame_interval == 0:
                encoded_success, encoded = cv2.imencode(".jpg", frame)
                if not encoded_success:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail="استخراج تصویر از ویدیو با خطا مواجه شد",
                    )
                encoded_bytes = encoded.tobytes()
                frames.append(
                    {
                        "frame_number": frame_index,
                        "frame_index": frame_index,
                        "image_base64": base64.b64encode(encoded_bytes).decode("utf-8"),
                        "image_size_kb": round(len(encoded_bytes) / 1024, 2),
                    }
                )
                if max_frames is not None and len(frames) >= max_frames:
                    break
    finally:
        capture.release()

    return {
        "success": True,
        "detection_id": detection_id,
        "video_path": public_video_path,
        "total_frames_in_video": frame_count,
        "frames_extracted": len(frames),
        "frame_interval": frame_interval,
        "frames": frames,
    }


@router.get("/extract-frames/{detection_id}/")
def extract_frames(
    detection_id: int,
    frame_interval: int = 1,
    max_frames: int | None = None,
    _: UserRecord = Depends(get_current_user),
) -> dict[str, Any]:
    if frame_interval < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="frame_interval باید حداقل ۱ باشد",
        )
    if max_frames is not None and max_frames < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="max_frames باید حداقل ۱ باشد",
        )

    record = get_detection_log_store().get(detection_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="لاگ تشخیص یافت نشد")
    saved_video = record.face_video_or_unknown_faces
    if record.face_video_status != MEDIA_STATUS_READY:
        raise HTTPException(
            status_code=(
                status.HTTP_409_CONFLICT
                if record.face_video_status == "writing"
                else status.HTTP_404_NOT_FOUND
            ),
            detail="ویدیوی چهره هنوز آماده نیست",
        )
    if not saved_video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="برای این لاگ ویدیای ذخیره‌شده‌ای وجود ندارد",
        )
    video_path = _resolve_saved_video_path(saved_video)
    if not video_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="فایل ویدیای ذخیره‌شده یافت نشد",
        )
    return _extract_video_frames(
        video_path,
        detection_id,
        frame_interval,
        max_frames,
        f"/api/v1/logs/{detection_id}/face-video",
    )
