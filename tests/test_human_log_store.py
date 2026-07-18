from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sqlite3

import cv2
import numpy as np

from app.core.human_log_store import HumanLogStore
from app.core.types import FramePacket, TaskName, TaskResult


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


def test_one_log_per_human_track_is_upgraded_after_recognition(tmp_path: Path) -> None:
    store = HumanLogStore(tmp_path / "logs.sqlite3", tmp_path / "media")
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
        assert rows[0]["first_seen"] == "2026-07-18T00:00:01+00:00"
        assert rows[0]["last_seen"] == "2026-07-18T00:00:03+00:00"
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
                    assert frame.shape[:2] == (224, 224)
            finally:
                capture.release()
        assert store.status()["saved_snapshots"] == 2
    finally:
        store.close()


def test_best_face_is_saved_when_full_frame_sample_is_not_due(tmp_path: Path) -> None:
    store = HumanLogStore(
        tmp_path / "logs.sqlite3",
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
            assert saved_face.shape[:2] == (224, 224)
        finally:
            capture.release()
    finally:
        store.close()


def test_legacy_human_log_schema_removes_pose_and_crop_video_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "logs.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE human_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                camera TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                name TEXT NOT NULL DEFAULT 'Unknown',
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                recognition_score REAL NOT NULL DEFAULT 0,
                ref_img_id TEXT,
                snapshot_url TEXT NOT NULL DEFAULT '',
                video_url TEXT NOT NULL DEFAULT '',
                face_video_url TEXT NOT NULL DEFAULT '',
                snapshot_quality REAL NOT NULL DEFAULT 0,
                best_face_quality REAL NOT NULL DEFAULT 0,
                best_face_yaw REAL,
                best_face_pitch REAL,
                best_face_roll REAL,
                human_video_frames INTEGER NOT NULL DEFAULT 0,
                accepted_face_frames INTEGER NOT NULL DEFAULT 0,
                UNIQUE(session_id, camera, track_id)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO human_logs (
                session_id, camera, track_id, first_seen, last_seen,
                human_video_frames
            ) VALUES ('old-session', 'camera-01', 13, 'start', 'end', 7)
            """
        )

    store = HumanLogStore(path, tmp_path / "media")
    try:
        row = store.list(track_id=13)[0]
        assert row["full_frame_video_frames"] == 7
        with sqlite3.connect(path) as connection:
            columns = [
                item[1]
                for item in connection.execute("PRAGMA table_info(human_logs)")
            ]
        assert "human_video_frames" not in columns
        assert "best_face_yaw" not in columns
        assert "best_face_pitch" not in columns
        assert "best_face_roll" not in columns
    finally:
        store.close()
