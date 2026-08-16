from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from bisect import bisect_left
from fractions import Fraction
from pathlib import Path
from typing import Sequence

from app.core.detection_event_schemas import HumanTrackObservation

@dataclass(frozen=True, slots=True)
class ExtractionResult:
    clip_path: Path
    snapshot_path: Path
    frames: int


def _segment_capture_time(
    segment_start: datetime, segment_end: datetime,
    media_elapsed_seconds: float, media_duration_seconds: float,
) -> datetime:
    """Map MP4 PTS onto the measured durable wall-clock interval."""
    from datetime import timedelta
    wall_duration = max(0.0, (segment_end - segment_start).total_seconds())
    if media_duration_seconds <= 0 or wall_duration <= 0:
        return segment_start + timedelta(seconds=max(0.0, media_elapsed_seconds))
    progress = max(0.0, min(1.0, media_elapsed_seconds / media_duration_seconds))
    return segment_start + timedelta(seconds=progress * wall_duration)


def _interpolated_box(
    observations: Sequence[HumanTrackObservation], captured: datetime,
    output_width: int, output_height: int,
) -> tuple[int, int, int, int] | None:
    if not observations or captured < observations[0].captured_at_utc or captured > observations[-1].captured_at_utc:
        return None
    times = [item.captured_at_utc for item in observations]
    right = bisect_left(times, captured)
    if right == 0:
        left = right = 0
    elif right >= len(observations):
        left = right = len(observations) - 1
    else:
        left = right - 1
    before, after = observations[left], observations[right]
    span = (after.captured_at_utc - before.captured_at_utc).total_seconds()
    fraction = 0.0 if span <= 0 else (captured - before.captured_at_utc).total_seconds() / span
    before_box = tuple((value / (before.frame_width if index % 2 == 0 else before.frame_height))
                       for index, value in enumerate(before.bounding_box))
    after_box = tuple((value / (after.frame_width if index % 2 == 0 else after.frame_height))
                      for index, value in enumerate(after.bounding_box))
    normalized = tuple(a + (b - a) * fraction for a, b in zip(before_box, after_box))
    return tuple(
        max(0, min((output_width if index % 2 == 0 else output_height) - 1,
                   round(value * (output_width if index % 2 == 0 else output_height))))
        for index, value in enumerate(normalized)
    )  # type: ignore[return-value]


def extract_human_media(
    segment_paths: Sequence[Path], segment_starts: Sequence[datetime],
    clip_start: datetime, clip_end: datetime, best_frame_at: datetime,
    clip_path: Path, snapshot_path: Path, *, max_segments: int = 16,
    max_duration_seconds: float = 120.0,
    track_id: int | None = None,
    track_observations: Sequence[HumanTrackObservation] = (),
    segment_ends: Sequence[datetime] | None = None,
) -> ExtractionResult:
    """Decode selected segment frames and create an H.264 MP4 plus full-frame JPEG."""
    import av
    import cv2
    if not segment_paths or len(segment_paths) != len(segment_starts):
        raise ValueError("segment paths and starts must be non-empty and aligned")
    if segment_ends is not None and len(segment_ends) != len(segment_paths):
        raise ValueError("segment ends must align with segment paths")
    if len(segment_paths) > max_segments or clip_end <= clip_start or (clip_end - clip_start).total_seconds() > max_duration_seconds:
        raise ValueError("extraction bounds exceeded")
    output = av.open(str(clip_path), mode="w")
    stream = None
    output_rate = None
    frame_count = 0
    nearest = None
    nearest_delta = float("inf")
    try:
        for segment_index, (path, segment_start) in enumerate(zip(segment_paths, segment_starts)):
            with av.open(str(path)) as source:
                video = next((item for item in source.streams if item.type == "video"), None)
                if video is None or video.width <= 0 or video.height <= 0:
                    raise ValueError(f"segment has no valid video stream: {path}")
                rate = video.average_rate
                if rate is None or float(rate) <= 0:
                    raise ValueError(f"segment has no positive frame rate: {path}")
                media_duration = (
                    float(video.duration * video.time_base)
                    if video.duration is not None and video.time_base is not None else 0.0
                )
                if stream is None:
                    stream = output.add_stream("libx264", rate=rate)
                    output_rate = rate
                    stream.width, stream.height, stream.pix_fmt = video.width, video.height, "yuv420p"
                elif (stream.width, stream.height) != (video.width, video.height):
                    raise ValueError("segment dimensions differ")
                first_time = None
                for frame in source.decode(video):
                    if frame.time is None:
                        continue
                    if first_time is None:
                        first_time = float(frame.time)
                    media_elapsed = float(frame.time) - first_time
                    captured = (
                        _segment_capture_time(
                            segment_start, segment_ends[segment_index],
                            media_elapsed, media_duration,
                        )
                        if segment_ends is not None else
                        segment_start + __import__("datetime").timedelta(seconds=media_elapsed)
                    )
                    if captured < clip_start or captured >= clip_end:
                        continue
                    output_frame = frame
                    if track_id is not None and track_observations:
                        first_observation = track_observations[0].captured_at_utc
                        last_observation = track_observations[-1].captured_at_utc
                        if first_observation <= captured <= last_observation:
                            image = frame.to_ndarray(format="bgr24")
                            box = _interpolated_box(track_observations, captured, frame.width, frame.height)
                            if box is None:
                                raise ValueError("track observations do not cover selected frame")
                            p1, p2 = box[:2], box[2:]
                            thickness = max(2, round(min(frame.width, frame.height) / 360))
                            cv2.rectangle(image, p1, p2, (0, 255, 0), thickness)
                            label = f"Track ID: {track_id}"
                            font_scale = max(0.5, min(frame.width, frame.height) / 720)
                            (label_width, label_height), baseline = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                            )
                            label_bottom = max(label_height + baseline + 2, p1[1])
                            label_top = label_bottom - label_height - baseline - 2
                            cv2.rectangle(
                                image, (p1[0], label_top),
                                (min(frame.width - 1, p1[0] + label_width + 4), label_bottom),
                                (0, 255, 0), -1,
                            )
                            cv2.putText(
                                image, label, (p1[0] + 2, label_bottom - baseline - 1),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness,
                                cv2.LINE_AA,
                            )
                            output_frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                    delta = abs((captured - best_frame_at).total_seconds())
                    if delta < nearest_delta:
                        nearest, nearest_delta = output_frame.to_ndarray(format="bgr24"), delta
                    output_frame.pts = frame_count
                    output_frame.time_base = Fraction(output_rate.denominator, output_rate.numerator)
                    for packet in stream.encode(output_frame):
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
