from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.core.media_utils import save_video_frames


def test_save_video_frames_creates_readable_multi_frame_mp4(tmp_path: Path) -> None:
    path = tmp_path / "detection.mp4"
    save_video_frames(
        (
            np.zeros((32, 48, 3), dtype=np.uint8),
            np.full((32, 48, 3), 255, dtype=np.uint8),
        ),
        path,
    )

    capture = cv2.VideoCapture(str(path))
    frame_count = 0
    try:
        opened, frame = capture.read()
        if opened:
            frame_count = 1
            while capture.grab():
                frame_count += 1
    finally:
        capture.release()

    assert opened
    assert frame is not None
    assert frame.shape[:2] == (32, 48)
    assert frame_count >= 2

import pytest

pytestmark = pytest.mark.unit
