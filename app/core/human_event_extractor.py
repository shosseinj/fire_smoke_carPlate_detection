from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Sequence

@dataclass(frozen=True, slots=True)
class ExtractionResult:
    clip_path: Path
    snapshot_path: Path
    frames: int


def extract_human_media(
    segment_paths: Sequence[Path], segment_starts: Sequence[datetime],
    clip_start: datetime, clip_end: datetime, best_frame_at: datetime,
    clip_path: Path, snapshot_path: Path, *, max_segments: int = 16,
    max_duration_seconds: float = 120.0,
) -> ExtractionResult:
    """Decode selected segment frames and create an H.264 MP4 plus full-frame JPEG."""
    import av
    if not segment_paths or len(segment_paths) != len(segment_starts):
        raise ValueError("segment paths and starts must be non-empty and aligned")
    if len(segment_paths) > max_segments or clip_end <= clip_start or (clip_end - clip_start).total_seconds() > max_duration_seconds:
        raise ValueError("extraction bounds exceeded")
    output = av.open(str(clip_path), mode="w")
    stream = None
    output_rate = None
    frame_count = 0
    nearest = None
    nearest_delta = float("inf")
    try:
        for path, segment_start in zip(segment_paths, segment_starts):
            with av.open(str(path)) as source:
                video = next((item for item in source.streams if item.type == "video"), None)
                if video is None or video.width <= 0 or video.height <= 0:
                    raise ValueError(f"segment has no valid video stream: {path}")
                rate = video.average_rate
                if rate is None or float(rate) <= 0:
                    raise ValueError(f"segment has no positive frame rate: {path}")
                if stream is None:
                    stream = output.add_stream("libx264", rate=rate)
                    output_rate = float(rate)
                    stream.width, stream.height, stream.pix_fmt = video.width, video.height, "yuv420p"
                elif (stream.width, stream.height) != (video.width, video.height):
                    raise ValueError("segment dimensions differ")
                elif abs(float(rate) - float(output_rate)) > max(.01, float(output_rate) * .01):
                    raise ValueError("segment frame rates differ")
                first_time = None
                for frame in source.decode(video):
                    if frame.time is None:
                        continue
                    if first_time is None:
                        first_time = float(frame.time)
                    captured = segment_start + __import__("datetime").timedelta(seconds=float(frame.time) - first_time)
                    if captured < clip_start or captured >= clip_end:
                        continue
                    delta = abs((captured - best_frame_at).total_seconds())
                    if delta < nearest_delta:
                        nearest, nearest_delta = frame.to_ndarray(format="bgr24"), delta
                    frame.pts = frame_count
                    frame.time_base = Fraction(rate.denominator, rate.numerator)
                    for packet in stream.encode(frame):
                        output.mux(packet)
                    frame_count += 1
        if stream is None or frame_count == 0 or nearest is None:
            raise ValueError("requested interval contains no decodable frames")
        for packet in stream.encode():
            output.mux(packet)
    finally:
        output.close()
    image = av.VideoFrame.from_ndarray(nearest, format="bgr24")
    with av.open(str(snapshot_path), mode="w", format="image2") as snapshot:
        jpeg = snapshot.add_stream("mjpeg")
        jpeg.width, jpeg.height, jpeg.pix_fmt = image.width, image.height, "yuvj420p"
        for packet in jpeg.encode(image):
            snapshot.mux(packet)
        for packet in jpeg.encode():
            snapshot.mux(packet)
    # Verification is deliberately decode-based rather than trusting container close.
    with av.open(str(clip_path)) as check:
        video = next((item for item in check.streams if item.type == "video"), None)
        if video is None or video.codec_context.name != "h264" or next(check.decode(video=0), None) is None:
            raise ValueError("generated clip is not decodable")
    with av.open(str(snapshot_path)) as check:
        frame = next(check.decode(video=0), None)
        if frame is None or (frame.width, frame.height) != (stream.width, stream.height):
            raise ValueError("generated full-frame JPEG is invalid")
    return ExtractionResult(clip_path, snapshot_path, frame_count)
