from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

from app.api import extract_frames
from app.core.detection_log_store import DetectionLogRecord


def _record(face_video: str | None, video: str | None = None) -> DetectionLogRecord:
    return DetectionLogRecord(
        id=7,
        source_system="test",
        source_event_key=None,
        source_human_log_id=None,
        personnel_id=None,
        person="Unknown",
        confidence=0.9,
        detection_time="2026-07-26T12:00:00Z",
        ref_img_id=None,
        room_id=None,
        camera_id="camera-01",
        access_granted=False,
        counts_for_attendance=True,
        log_type="test",
        import_source_parts=None,
        face_image=None,
        face_thumbnail=None,
        body_image=None,
        snapshot_image=None,
        video=video,
        face_video_or_unknown_faces=face_video,
        video_status="ready" if video else "missing",
        face_video_status="ready" if face_video else "missing",
        media_finalized_at=None,
        created_by=None,
        updated_by=None,
        created_at_utc="2026-07-26T12:00:00Z",
        updated_at_utc="2026-07-26T12:00:00Z",
    )


def test_extract_frames_returns_exact_response_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 24))
    for index in range(4):
        writer.write(np.full((24, 32, 3), index * 40, dtype=np.uint8))
    writer.release()

    monkeypatch.setattr(
        extract_frames,
        "settings",
        SimpleNamespace(saved_media_path=tmp_path),
    )
    monkeypatch.setattr(
        extract_frames,
        "get_detection_log_store",
        lambda: SimpleNamespace(get=lambda _id: _record("/media/video.avi")),
    )

    response = extract_frames.extract_frames(
        detection_id=7,
        frame_interval=2,
        max_frames=1,
        _=SimpleNamespace(),
    )

    assert set(response) == {
        "success",
        "detection_id",
        "video_path",
        "total_frames_in_video",
        "frames_extracted",
        "frame_interval",
        "frames",
    }
    assert response["success"] is True
    assert response["detection_id"] == 7
    assert response["video_path"] == "/api/v1/logs/7/face-video"
    assert response["frames_extracted"] == 1
    assert response["frame_interval"] == 2
    assert set(response["frames"][0]) == {
        "frame_number",
        "frame_index",
        "image_base64",
        "image_size_kb",
    }
    assert response["frames"][0]["frame_number"] == 0


def test_extract_frames_requires_saved_video(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        extract_frames,
        "get_detection_log_store",
        lambda: SimpleNamespace(get=lambda _id: _record(None)),
    )

    with pytest.raises(HTTPException) as error:
        extract_frames.extract_frames(7, _=SimpleNamespace())
    assert error.value.status_code == 404


def test_extract_frames_does_not_fallback_to_full_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        extract_frames,
        "settings",
        SimpleNamespace(saved_media_path=tmp_path),
    )
    monkeypatch.setattr(
        extract_frames,
        "get_detection_log_store",
        lambda: SimpleNamespace(
            get=lambda _id: _record(None, video="/media/full-video.avi")
        ),
    )

    with pytest.raises(HTTPException) as error:
        extract_frames.extract_frames(7, _=SimpleNamespace())
    assert error.value.status_code == 404

pytestmark = pytest.mark.unit
