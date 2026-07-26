from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np


def save_video_frames(frames: Sequence[np.ndarray], path: Path, fps: float = 1.0) -> None:
    """Save a bounded sequence of frames as a playable MP4 artifact."""
    if not frames:
        raise ValueError("Video frames are empty")
    first = frames[0]
    if first is None or first.size == 0:
        raise ValueError("Video frame is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = first.shape[:2]
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
        for frame in frames:
            if frame is None or frame.size == 0 or frame.shape[:2] != (height, width):
                raise ValueError("Video frames must have the same non-empty dimensions")
            writer.write(frame)
    except Exception:
        writer.release()
        path.unlink(missing_ok=True)
        raise
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not save video: {path}")


def save_single_frame_video(frame: np.ndarray, path: Path, fps: float = 1.0) -> None:
    """Save the supplied detection frame as a playable MP4 artifact."""
    save_video_frames((frame,), path, fps)
