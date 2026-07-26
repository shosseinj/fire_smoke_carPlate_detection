from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.core.media_utils import save_single_frame_video


def test_save_single_frame_video_creates_readable_mp4(tmp_path: Path) -> None:
    path = tmp_path / "detection.mp4"
    save_single_frame_video(np.zeros((32, 48, 3), dtype=np.uint8), path)

    capture = cv2.VideoCapture(str(path))
    try:
        opened, frame = capture.read()
    finally:
        capture.release()

    assert opened
    assert frame is not None
    assert frame.shape[:2] == (32, 48)
