from datetime import datetime, timedelta, timezone
from fractions import Fraction

import numpy as np
import pytest

av = pytest.importorskip("av")

from app.core.human_event_extractor import extract_human_media


def _write_video(path, rate: int, frames: int) -> None:
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=rate); stream.width = 64; stream.height = 48; stream.pix_fmt = "yuv420p"
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), index * 10, np.uint8), format="bgr24")
            frame.pts = index; frame.time_base = Fraction(1, rate)
            for packet in stream.encode(frame): output.mux(packet)
        for packet in stream.encode(): output.mux(packet)


def test_extracts_decodable_h264_and_full_size_jpeg(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    _write_video(source, 10, 10)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = extract_human_media([source], [now], now, now + timedelta(seconds=.9),
                                 now + timedelta(seconds=.4), tmp_path / "clip.mp4", tmp_path / "snapshot.jpg")
    assert result.frames == 9
    with av.open(str(result.snapshot_path)) as image:
        frame = next(image.decode(video=0))
        assert (frame.width, frame.height) == (64, 48)


def test_extracts_adjacent_segments_with_different_reported_rates(tmp_path) -> None:
    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    _write_video(first, 10, 10)
    _write_video(second, 12, 12)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    result = extract_human_media(
        [first, second], [now, now + timedelta(seconds=1)],
        now, now + timedelta(seconds=1.9), now + timedelta(seconds=1.4),
        tmp_path / "clip.mp4", tmp_path / "snapshot.jpg",
    )

    assert result.frames > 10
