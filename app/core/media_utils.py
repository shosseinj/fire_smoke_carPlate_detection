from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_single_frame_video(frame: np.ndarray, path: Path, fps: float = 1.0) -> None:
    """Save the supplied detection frame as a playable MP4 artifact."""
    if frame is None or frame.size == 0:
        raise ValueError("Video frame is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not create video: {path}")
    try:
        writer.write(frame)
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not save video: {path}")
