from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from app.core.human_log_store import HumanLogStore
from app.core.types import FramePacket, TaskName, TaskResult
from app.database import Database, metadata


def packet(frame_index: int) -> FramePacket:
    return FramePacket(
        source_id="camera-01",
        frame=np.zeros((120, 160, 3), dtype=np.uint8),
        round_sequence=frame_index,
        frame_index=frame_index,
        captured_monotonic=float(frame_index),
        captured_at_utc=f"2026-07-18T00:00:0{frame_index}+00:00",
    )


def result(
    source: FramePacket,
    name: str,
    score: float,
    *,
    face_quality: float | None = None,
) -> TaskResult:
    faces = []
    if face_quality is not None:
        faces.append(
            {
                "track_id": 13,
                "bbox": [40, 20, 90, 75],
                "landmarks": [[50, 35], [75, 35], [63, 48], [53, 63], [73, 63]],
                "quality_valid": True,
                "quality_score": face_quality,
                "quality_metrics": {"yaw": 2.0, "pitch": 8.0, "roll": 1.0},
            }
        )
    return TaskResult.success(
        task=TaskName.FACE_RECOGNITION,
        packet=source,
        processing_ms=1.0,
        data={
            "tracking_session_id": "session-a",
            "faces": faces,
            "humans": [
                {
                    "track_id": 13,
                    "bbox": [10, 10, 100, 110],
                    "person": name,
                    "recognition_score": score,
                    "ref_img_id": "reference-1" if name != "Unknown" else None,
                }
            ],
        },
    )


def test_one_log_per_human_track_is_upgraded_after_recognition(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store = HumanLogStore(postgres_database, tmp_path / "media")
    try:
        first = packet(1)
        recognized = packet(2)
        later_lower_quality = packet(3)
        store.observe_result(first, result(first, "Unknown", 0.0))
        store.observe_result(
            recognized,
            result(recognized, "Alice", 0.93, face_quality=0.90),
        )
        store.observe_result(
            later_lower_quality,
            result(later_lower_quality, "Alice", 0.80, face_quality=0.60),
        )
        store.flush()
        store.close()

        rows = store.list(camera="camera-01", track_id=13)
        assert len(rows) == 1
        assert rows[0]["name"] == "Alice"
        assert rows[0]["first_seen"] == "2026-07-18T00:00:01Z"
        assert rows[0]["last_seen"] == "2026-07-18T00:00:03Z"
        assert rows[0]["recognition_score"] == 0.93
        assert rows[0]["snapshot_quality"] == 0.97
        assert rows[0]["best_face_quality"] == 0.90
        assert "best_face_yaw" not in rows[0]
        assert "best_face_pitch" not in rows[0]
        assert "best_face_roll" not in rows[0]
        assert rows[0]["snapshot_url"].startswith("/media/human_snapshots/")
        assert rows[0]["video_url"].startswith("/media/human_videos/")
        assert rows[0]["face_video_url"].startswith("/media/human_face_videos/")
        assert rows[0]["full_frame_video_frames"] == 3
        assert rows[0]["accepted_face_frames"] == 2
        snapshot = tmp_path / "media" / "human_snapshots" / Path(
            rows[0]["snapshot_url"]
        ).name
        assert snapshot.is_file()
        human_video = tmp_path / "media" / "human_videos" / Path(
            rows[0]["video_url"]
        ).name
        face_video = tmp_path / "media" / "human_face_videos" / Path(
            rows[0]["face_video_url"]
        ).name
        for video in (human_video, face_video):
            assert video.is_file()
            assert video.stat().st_size > 0
            capture = cv2.VideoCapture(str(video))
            try:
                ok, frame = capture.read()
                assert ok is True
                assert frame is not None
                if video == human_video:
                    assert frame.shape[:2] == (120, 160)
                else:
                    assert frame.shape[:2] == (112, 112)
            finally:
                capture.release()
        assert store.status()["saved_snapshots"] == 2
    finally:
        store.close()


def test_best_face_is_saved_when_full_frame_sample_is_not_due(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store = HumanLogStore(
        postgres_database,
        tmp_path / "media",
        video_fps=10.0,
    )
    try:
        first = packet(1)
        accepted_face = replace(packet(2), captured_monotonic=1.01)

        store.observe_result(first, result(first, "Unknown", 0.0))
        store.observe_result(
            accepted_face,
            result(accepted_face, "Alice", 0.93, face_quality=0.90),
        )
        store.flush()
        store.close()

        row = store.list(track_id=13)[0]
        assert row["full_frame_video_frames"] == 1
        assert row["accepted_face_frames"] == 1
        assert row["face_video_url"].startswith("/media/human_face_videos/")
        face_video = tmp_path / "media" / "human_face_videos" / Path(
            row["face_video_url"]
        ).name
        capture = cv2.VideoCapture(str(face_video))
        try:
            ok, saved_face = capture.read()
            assert ok is True
            assert saved_face is not None
            assert saved_face.shape[:2] == (112, 112)
        finally:
            capture.release()
    finally:
        store.close()


def test_postgresql_human_log_schema_uses_current_fields() -> None:
    columns = metadata.tables["human_logs"].c
    assert "full_frame_video_frames" in columns
    assert "human_video_frames" not in columns
    assert "best_face_yaw" not in columns
    assert "best_face_pitch" not in columns
    assert "best_face_roll" not in columns


def test_media_is_cropped_and_encoded_from_native_source_resolution(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    source_frame = np.zeros((360, 480, 3), dtype=np.uint8)
    source_frame[30:330, 60:420] = (40, 160, 240)
    inference = replace(packet(1), metadata={"source_frame": source_frame})
    detected = result(inference, "Alice", 0.95, face_quality=0.95)
    detected.data["humans"][0]["source_bbox"] = [30, 30, 330, 330]
    detected.data["faces"][0]["source_bbox"] = [120, 60, 270, 225]
    detected.data["faces"][0]["source_landmarks"] = [
        [150, 105],
        [225, 105],
        [189, 144],
        [159, 189],
        [219, 189],
    ]
    store = HumanLogStore(postgres_database, tmp_path / "media")
    try:
        store.observe_result(inference, detected)
        store.flush()
        store.close()

        row = store.list(track_id=13)[0]
        snapshot_path = tmp_path / "media" / "human_snapshots" / Path(
            row["snapshot_url"]
        ).name
        snapshot = cv2.imread(str(snapshot_path))
        assert snapshot is not None
        assert snapshot.shape[0] > inference.frame.shape[0]
        assert snapshot.shape[1] > inference.frame.shape[1]

        full_video = tmp_path / "media" / "human_videos" / Path(
            row["video_url"]
        ).name
        full_capture = cv2.VideoCapture(str(full_video))
        face_video = tmp_path / "media" / "human_face_videos" / Path(
            row["face_video_url"]
        ).name
        face_capture = cv2.VideoCapture(str(face_video))
        try:
            full_ok, full_frame = full_capture.read()
            face_ok, face_frame = face_capture.read()
            assert full_ok is True
            assert full_frame.shape[:2] == (360, 480)
            assert face_ok is True
            assert face_frame.shape[:2] == (166, 166)
        finally:
            full_capture.release()
            face_capture.release()
    finally:
        store.close()
