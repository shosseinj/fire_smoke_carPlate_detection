from __future__ import annotations

from datetime import datetime, timezone

import cv2
import numpy as np

from app.api.plate_logs import _response
from app.core.detection_media import DetectionMediaStorage


def test_plate_response_embeds_thumbnail_and_exposes_protected_urls(tmp_path) -> None:
    media = DetectionMediaStorage(tmp_path)
    image = np.full((480, 640, 3), 90, dtype=np.uint8)
    snapshot = tmp_path / "plate" / "snapshots" / "plate.jpg"
    snapshot.parent.mkdir(parents=True)
    assert cv2.imwrite(str(snapshot), image)
    video = tmp_path / "plate" / "videos" / "plate.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"ready-video")

    response = _response(
        {
            "id": 7,
            "detection_time": datetime(2026, 7, 29, tzinfo=timezone.utc),
            "snapshot_key": media.key_for_path(snapshot),
            "video_key": media.key_for_path(video),
            "snapshot_url": "/media/plate/snapshots/plate.jpg",
        },
        media,
    )

    assert response["snapshot_thumbnail"].startswith("data:image/jpeg;base64,")
    assert response["snap_shot_url"] == "/api/v1/plate-logs/7/snapshot"
    assert response["video_url"] == "/api/v1/plate-logs/7/video"
    assert "snapshot_key" not in response
    assert "video_key" not in response
    assert "snapshot_url" not in response


def test_plate_response_hides_missing_media(tmp_path) -> None:
    response = _response(
        {
            "id": 8,
            "detection_time": "2026-07-29T00:00:00+00:00",
            "snapshot_key": "plate/snapshots/missing.jpg",
            "video_key": None,
        },
        DetectionMediaStorage(tmp_path),
    )

    assert response["snapshot_thumbnail"] is None
    assert response["snap_shot_url"] is None
    assert response["video_url"] is None


def test_plate_collection_response_skips_media_file_checks(tmp_path) -> None:
    response = _response(
        {
            "id": 9,
            "detection_time": "2026-07-29T00:00:00+00:00",
            "snapshot_key": "plate/snapshots/not-checked.jpg",
            "video_key": "plate/videos/not-checked.mp4",
        },
        DetectionMediaStorage(tmp_path),
        check_media=False,
    )

    assert response["snapshot_thumbnail"] is None
    assert response["snap_shot_url"] == "/api/v1/plate-logs/9/snapshot"
    assert response["video_url"] == "/api/v1/plate-logs/9/video"
