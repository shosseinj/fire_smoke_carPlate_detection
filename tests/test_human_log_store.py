from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import queue
from unittest.mock import patch

import cv2
import numpy as np

from app.core.human_log_store import HumanLogStore
from app.core.detection_log_store import DetectionLogStore
from app.core.jalali_utils import utc_iso_to_jalali_datetime
from app.core.location_store import LocationStore
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


def colored_packet(frame_index: int, bgr: tuple[int, int, int]) -> FramePacket:
    return replace(
        packet(frame_index),
        frame=np.full((120, 160, 3), bgr, dtype=np.uint8),
    )


def assert_jpeg_has_dominant_channel(path: Path, channel: int) -> None:
    image = cv2.imread(str(path))
    assert image is not None, path
    means = image.mean(axis=(0, 1))
    assert int(np.argmax(means)) == channel, (path, means)


def test_persistence_name_formats_only_exact_unknown_sentinel() -> None:
    assert HumanLogStore._persistence_name("Unknown", 13) == "Unknown #13"
    assert HumanLogStore._persistence_name("Alice", 13) == "Alice"
    assert HumanLogStore._persistence_name("1234567891", 13) == "1234567891"


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
        expired = packet(4)
        expired_result = result(expired, "Alice", 0.80)
        expired_result.data["humans"] = []
        expired_result.data["faces"] = []
        expired_result.data["disappeared_humans"] = [
            {
                "track_id": 13,
                "bbox": [10, 10, 100, 110],
                "person": "Alice",
                "recognition_score": 0.80,
                "ref_img_id": "reference-1",
                "confidence": 0.0,
            }
        ]
        store.observe_result(expired, expired_result)
        store.flush()
        store.close()

        rows = store.list(camera="camera-01", track_id=13)
        assert len(rows) == 1
        assert rows[0]["name"] == "Alice"
        assert rows[0]["first_seen"] == utc_iso_to_jalali_datetime(
            "2026-07-18T00:00:01Z"
        )
        assert rows[0]["last_seen"] == utc_iso_to_jalali_datetime(
            "2026-07-18T00:00:04Z"
        )
        assert rows[0]["recognition_score"] == 0.93
        assert rows[0]["snapshot_quality"] == 0.97
        assert rows[0]["best_face_quality"] == 0.90
        assert "best_face_yaw" not in rows[0]
        assert "best_face_pitch" not in rows[0]
        assert "best_face_roll" not in rows[0]
        with postgres_database.connection() as connection:
            media_row = connection.execute(
                "SELECT snapshot_url, video_url, face_video_url FROM human_logs "
                "WHERE session_id = ? AND camera = ? AND track_id = ?",
                ("session-a", "camera-01", 13),
            ).fetchone()
        assert media_row is not None
        assert str(media_row["snapshot_url"]).startswith("human/body_images/")
        assert str(media_row["video_url"]).startswith("human/videos/")
        assert str(media_row["face_video_url"]).startswith("human/face_videos/")
        assert rows[0]["full_frame_video_frames"] == 3
        assert rows[0]["accepted_face_frames"] == 2
        snapshot = tmp_path / "media" / "human" / "body_images" / Path(
            str(media_row["snapshot_url"])
        ).name
        assert snapshot.is_file()
        human_video = tmp_path / "media" / "human" / "videos" / Path(
            str(media_row["video_url"])
        ).name
        face_video = tmp_path / "media" / "human" / "face_videos" / Path(
            str(media_row["face_video_url"])
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
        assert store.status()["saved_snapshots"] == 1
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

        with postgres_database.connection() as connection:
            row = connection.execute(
                "SELECT snapshot_url, video_url, face_video_url, "
                "full_frame_video_frames, accepted_face_frames FROM human_logs "
                "WHERE session_id = ? AND camera = ? AND track_id = ?",
                ("session-a", "camera-01", 13),
            ).fetchone()
        assert row is not None
        assert row["full_frame_video_frames"] == 1
        assert row["accepted_face_frames"] == 1
        with postgres_database.connection() as connection:
            media_row = connection.execute(
                "SELECT face_video_url FROM human_logs "
                "WHERE session_id = ? AND camera = ? AND track_id = ?",
                ("session-a", "camera-01", 13),
            ).fetchone()
        assert media_row is not None
        assert str(media_row["face_video_url"]).startswith("human/face_videos/")
        face_video = tmp_path / "media" / "human" / "face_videos" / Path(
            str(media_row["face_video_url"])
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


def test_disappeared_track_is_visible_in_detection_log_filter(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    national_code = "1234567891"
    with postgres_database.connection() as connection:
        existing = connection.execute(
            "SELECT id FROM personnel WHERE national_code = ?",
            (national_code,),
        ).fetchone()
        inserted_personnel = existing is None
        if existing is None:
            connection.execute(
                "INSERT INTO personnel (fname, lname, national_code) VALUES (?, ?, ?)",
                ("Test First", "Test Last", national_code),
            )
    detection_logs = DetectionLogStore(postgres_database)
    store = HumanLogStore(
        postgres_database,
        tmp_path / "media",
        detection_log_store=detection_logs,
    )
    disappeared = packet(4)
    final_result = result(disappeared, "Alice", 0.93, face_quality=0.90)
    final_result.data["humans"] = []
    final_result.data["disappeared_humans"] = [
        {
            "track_id": 13,
            "bbox": [10, 10, 100, 110],
            "person": national_code,
            "recognition_score": 0.93,
            "ref_img_id": "qdrant-point",
            "confidence": 0.0,
        }
    ]
    try:
        store.observe_result(disappeared, final_result)
        store.flush()
        records, _ = detection_logs.list_filter(
            camera_id="camera-01",
            log_type="camera_rtsp",
        )
        assert len(records) == 1
        assert records[0].person == national_code
        assert records[0].personnel_id is not None
        assert records[0].source_human_log_id is not None
        assert records[0].source_event_key == "human-track:session-a:camera-01:13"
    finally:
        store.close()
        if inserted_personnel:
            with postgres_database.connection() as connection:
                connection.execute(
                    "DELETE FROM personnel WHERE national_code = ?",
                    (national_code,),
                )


def test_polygon_gated_face_evidence_is_reused_when_track_disappears(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    detection_logs = DetectionLogStore(postgres_database)
    store = HumanLogStore(
        postgres_database,
        tmp_path / "media",
        detection_log_store=detection_logs,
    )
    observed = packet(6)
    disappeared = packet(7)
    final_result = result(disappeared, "Alice", 0.93)
    final_result.data["humans"] = []
    final_result.data["faces"] = []
    final_result.data["disappeared_humans"] = [
        {
            "track_id": 13,
            "bbox": [0, 0, 0, 0],
            "person": "Alice",
            "recognition_score": 0.93,
            "ref_img_id": None,
            "confidence": 0.0,
        }
    ]
    with postgres_database.connection() as connection:
        room_row = connection.execute(
            "SELECT id FROM rooms ORDER BY id LIMIT 1"
        ).fetchone()
    assert room_row is not None
    room_id = int(room_row["id"])
    try:
        # This represents a regular frame admitted by an explicit room polygon.
        store.observe_result(
            observed,
            result(observed, "Alice", 0.93, face_quality=0.90),
            persist_human_log=False,
            room_ids_by_track={13: room_id},
        )
        store.observe_result(
            disappeared,
            final_result,
            room_ids_by_track={},
        )
        store.flush()

        row = store.list(track_id=13)[0]
        assert row["name"] == "Alice"
        records, _ = detection_logs.list_filter(
            camera_id="camera-01",
            log_type="camera_rtsp",
        )
        assert len(records) == 1
        assert records[0].person == "Unknown"
        assert records[0].face_image
        assert records[0].room_id == room_id
        assert records[0].camera_id == "camera-01"
        assert records[0].video
        assert records[0].face_video_or_unknown_faces
        face_path = tmp_path / "media" / "human" / "detected_faces" / Path(
            records[0].face_image
        ).name
        assert face_path.is_file()
        person_video = tmp_path / "media" / Path(records[0].video)
        face_video = tmp_path / "media" / Path(
            records[0].face_video_or_unknown_faces
        )
        for video_path in (person_video, face_video):
            assert video_path.is_file()
            capture = cv2.VideoCapture(str(video_path))
            try:
                ok, video_frame = capture.read()
                assert ok is True
                assert video_frame is not None
            finally:
                capture.release()
    finally:
        store.close()


def test_outside_polygon_track_does_not_create_human_or_detection_log(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    detection_logs = DetectionLogStore(postgres_database)
    store = HumanLogStore(
        postgres_database,
        tmp_path / "media",
        detection_log_store=detection_logs,
    )
    observed = packet(8)
    disappeared = packet(9)
    final_result = result(disappeared, "Alice", 0.93)
    final_result.data["humans"] = []
    final_result.data["faces"] = []
    final_result.data["disappeared_humans"] = [
        {
            "track_id": 13,
            "bbox": [10, 10, 100, 110],
            "person": "Alice",
            "recognition_score": 0.93,
            "ref_img_id": None,
            "confidence": 0.0,
        }
    ]
    try:
        store.observe_result(
            observed,
            result(observed, "Alice", 0.93, face_quality=0.90),
            persist_human_log=False,
            room_ids_by_track={},
        )
        store.observe_result(
            disappeared,
            final_result,
            room_ids_by_track={},
        )
        store.flush()

        assert store.list(track_id=13) == []
        records, total = detection_logs.list_filter(camera_id="camera-01")
        assert total == 0
        assert records == []
    finally:
        store.close()


def test_track_without_valid_face_does_not_create_detection_media(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    detection_logs = DetectionLogStore(postgres_database)
    media_root = tmp_path / "media"
    store = HumanLogStore(
        postgres_database,
        media_root,
        detection_log_store=detection_logs,
    )
    disappeared = packet(4)
    final_result = result(disappeared, "Unknown", 0.0)
    final_result.data["humans"] = []
    final_result.data["faces"] = []
    final_result.data["disappeared_humans"] = [
        {
            "track_id": 13,
            "bbox": [10, 10, 100, 110],
            "person": "Unknown",
            "recognition_score": 0.0,
            "ref_img_id": None,
            "confidence": 0.0,
        }
    ]
    try:
        store.observe_result(disappeared, final_result)
        store.flush()
        assert detection_logs.get_by_source_event_key(
            "human-track:session-a:camera-01:13"
        ) is None
        for directory in ("detected_faces", "body_images", "full_frame_images"):
            assert not list((media_root / "human" / directory).glob("*.jpg"))
    finally:
        store.close()


def test_disappeared_known_track_saves_reference_and_current_image_side_by_side(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    national_code = "1234567892"
    media_root = tmp_path / "media"
    media_root.mkdir(parents=True, exist_ok=True)
    reference_path = media_root / "human" / "reference_images" / "reference.jpg"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(reference_path), np.full((40, 30, 3), (20, 80, 160), dtype=np.uint8))
    with postgres_database.connection() as connection:
        person_cursor = connection.execute(
            "INSERT INTO personnel (fname, lname, national_code) VALUES (?, ?, ?)",
            ("Known", "Person", national_code),
        )
        image_cursor = connection.execute(
            "INSERT INTO personnel_images "
            "(personnel_id, storage_key, is_primary) VALUES (?, ?, 1)",
            (person_cursor.lastrowid, "human/reference_images/reference.jpg"),
        )
        ref_img_id = int(image_cursor.lastrowid)

    store = HumanLogStore(postgres_database, media_root)
    disappeared = packet(5)
    final_result = result(
        disappeared, national_code, 0.94, face_quality=0.90
    )
    final_result.data["humans"] = []
    final_result.data["disappeared_humans"] = [
        {
            "track_id": 13,
            "bbox": [10, 10, 100, 110],
            "person": national_code,
            "recognition_score": 0.94,
            "ref_img_id": ref_img_id,
            "confidence": 0.0,
        }
    ]
    try:
        store.observe_result(disappeared, final_result)
        store.flush()
        row = store.list(track_id=13)[0]
        assert row["name"] == "Known Person"
        with postgres_database.connection() as connection:
            media_row = connection.execute(
                "SELECT snapshot_url FROM human_logs "
                "WHERE session_id = ? AND camera = ? AND track_id = ?",
                ("session-a", "camera-01", 13),
            ).fetchone()
        assert media_row is not None
        snapshot_path = media_root / "human" / "body_images" / Path(
            str(media_row["snapshot_url"])
        ).name
        saved = cv2.imread(str(snapshot_path))
        assert saved is not None
        assert saved.shape[1] > 90
    finally:
        store.close()
        with postgres_database.connection() as connection:
            connection.execute(
                "DELETE FROM personnel WHERE national_code = ?",
                (national_code,),
            )


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
        expired = replace(packet(2), metadata={"source_frame": source_frame})
        expired_result = result(expired, "Alice", 0.95)
        expired_result.data["humans"] = []
        expired_result.data["faces"] = []
        expired_result.data["disappeared_humans"] = [
            {
                "track_id": 13,
                "source_bbox": [30, 30, 330, 330],
                "person": "Alice",
                "recognition_score": 0.95,
                "ref_img_id": "reference-1",
                "confidence": 0.0,
            }
        ]
        store.observe_result(expired, expired_result)
        store.flush()
        store.close()

        with postgres_database.connection() as connection:
            row = connection.execute(
                "SELECT snapshot_url, video_url, face_video_url FROM human_logs "
                "WHERE session_id = ? AND camera = ? AND track_id = ?",
                ("session-a", "camera-01", 13),
            ).fetchone()
        assert row is not None
        snapshot_path = tmp_path / "media" / "human" / "body_images" / Path(
            str(row["snapshot_url"])
        ).name
        snapshot = cv2.imread(str(snapshot_path))
        assert snapshot is not None
        assert snapshot.shape[0] > inference.frame.shape[0]
        assert snapshot.shape[1] > inference.frame.shape[1]

        full_video = tmp_path / "media" / "human" / "videos" / Path(
            str(row["video_url"])
        ).name
        full_capture = cv2.VideoCapture(str(full_video))
        face_video = tmp_path / "media" / "human" / "face_videos" / Path(
            str(row["face_video_url"])
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


def test_finalized_still_evidence_is_atomic_and_idempotent(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    """Face, body, and full frame must be one save-once evidence bundle."""
    media_root = tmp_path / "media"
    detection_logs = DetectionLogStore(postgres_database)
    store = HumanLogStore(
        postgres_database,
        media_root,
        detection_log_store=detection_logs,
    )

    # Blue is an earlier, lower-quality candidate. Green is the selected best
    # frame. The red disappearance frame must not be mixed into its evidence.
    earlier = colored_packet(1, (255, 0, 0))
    selected = colored_packet(2, (0, 255, 0))
    disappeared = colored_packet(3, (0, 0, 255))
    final_result = result(disappeared, "Alice", 0.93)
    final_result.data["humans"] = []
    final_result.data["faces"] = []
    final_result.data["disappeared_humans"] = [
        {
            "track_id": 13,
            "bbox": [10, 10, 100, 110],
            "person": "Alice",
            "recognition_score": 0.93,
            "ref_img_id": None,
            "confidence": 0.0,
        }
    ]

    try:
        store.observe_result(
            earlier,
            result(earlier, "Alice", 0.80, face_quality=0.60),
            counts_for_attendance=False,
        )
        store.observe_result(
            selected,
            result(selected, "Alice", 0.93, face_quality=0.95),
            counts_for_attendance=False,
        )
        store.observe_result(
            disappeared,
            final_result,
            counts_for_attendance=False,
        )
        store.flush()

        records, _ = detection_logs.list_filter(
            camera_id="camera-01",
            log_type="camera_rtsp",
        )
        assert len(records) == 1
        record = records[0]
        assert record.counts_for_attendance is False
        with postgres_database.connection() as connection:
            human_row = connection.execute(
                "SELECT counts_for_attendance FROM human_logs "
                "WHERE session_id = ? AND camera = ? AND track_id = ?",
                ("session-a", "camera-01", 13),
            ).fetchone()
        assert human_row is not None
        assert bool(human_row["counts_for_attendance"]) is False
        first_keys = (record.face_image, record.body_image, record.snapshot_image)
        assert all(first_keys)
        expected_directories = (
            media_root / "human" / "detected_faces",
            media_root / "human" / "body_images",
            media_root / "human" / "full_frame_images",
        )
        for key, directory in zip(first_keys, expected_directories, strict=True):
            path = directory / Path(str(key)).name
            assert_jpeg_has_dominant_channel(path, 1)
            assert len(list(directory.glob("*.jpg"))) == 1

        # A duplicate tracker-expiry notification must reuse the same database
        # keys and must not create or replace any still-image files.
        store.observe_result(
            disappeared,
            final_result,
            counts_for_attendance=False,
        )
        store.flush()
        duplicate = detection_logs.get_by_source_event_key(
            "human-track:session-a:camera-01:13"
        )
        assert duplicate is not None
        assert (
            duplicate.face_image,
            duplicate.body_image,
            duplicate.snapshot_image,
        ) == first_keys
        for directory in expected_directories:
            assert len(list(directory.glob("*.jpg"))) == 1
    finally:
        store.close()


def test_active_track_video_is_not_split_by_idle_cleanup_and_counts_are_persisted(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store = HumanLogStore(
        postgres_database,
        tmp_path / "media",
        video_fps=5.0,
        video_idle_seconds=1.0,
    )
    room = LocationStore(postgres_database).create_room("Video zone")
    key = ("session-a", "camera-01", 13)
    try:
        first = packet(1)
        first_result = result(first, "Alice", 0.93, face_quality=0.90)
        store.observe_result(
            first,
            first_result,
            persist_human_log=False,
            room_ids_by_track={13: room.id},
        )
        store.observe_result(
            first,
            first_result,
            room_ids_by_track={13: room.id},
        )
        store.flush()

        state = store._media[key]
        first_video_key = state.video_key
        state.last_event_monotonic = 0.0
        store._close_idle_media()
        assert store._media[key] is state

        second = packet(2)
        second_result = result(second, "Alice", 0.93, face_quality=0.85)
        store.observe_result(
            second,
            second_result,
            persist_human_log=False,
            room_ids_by_track={13: room.id},
        )
        store.observe_result(
            second,
            second_result,
            room_ids_by_track={13: room.id},
        )
        store.flush()

        assert store._media[key].video_key == first_video_key
        assert len(list((tmp_path / "media" / "human" / "videos").glob("*.mp4"))) == 1
        with postgres_database.connection() as connection:
            row = connection.execute(
                "SELECT full_frame_video_frames, accepted_face_frames "
                "FROM human_logs WHERE session_id = ? AND camera = ? AND track_id = ?",
                key,
            ).fetchone()
        assert row is not None
        assert row["full_frame_video_frames"] == 2
        assert row["accepted_face_frames"] == 2
    finally:
        store.close()


def test_full_frame_video_includes_bounded_pre_and_post_roll(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store = HumanLogStore(
        postgres_database,
        tmp_path / "media",
        video_fps=5.0,
        video_pre_roll_frames=2,
        video_post_roll_frames=2,
        video_pre_roll_max_bytes=8 * 1024 * 1024,
    )
    room = LocationStore(postgres_database).create_room("Roll zone")
    try:
        for frame_index in (1, 2):
            outside = colored_packet(frame_index, (frame_index * 20, 0, 0))
            outside_result = result(outside, "Unknown", 0.0)
            outside_result.data["humans"] = []
            outside_result.data["faces"] = []
            store.observe_result(
                outside,
                outside_result,
                persist_human_log=False,
                room_ids_by_track={},
            )

        entered = colored_packet(3, (0, 80, 0))
        entered_result = result(entered, "Alice", 0.93)
        store.observe_result(
            entered,
            entered_result,
            persist_human_log=False,
            room_ids_by_track={13: room.id},
        )
        store.observe_result(
            entered,
            entered_result,
            room_ids_by_track={13: room.id},
        )

        valid_face = colored_packet(4, (0, 100, 0))
        store.observe_result(
            valid_face,
            result(valid_face, "Alice", 0.93, face_quality=0.90),
            persist_human_log=False,
            room_ids_by_track={13: room.id},
        )
        outside = colored_packet(5, (0, 0, 100))
        store.observe_result(
            outside,
            result(outside, "Alice", 0.93),
            persist_human_log=False,
            room_ids_by_track={},
        )

        disappeared = packet(6)
        disappeared_result = result(disappeared, "Alice", 0.93)
        disappeared_result.data["humans"] = []
        disappeared_result.data["faces"] = []
        disappeared_result.data["disappeared_humans"] = [
            {
                "track_id": 13,
                "bbox": [10, 10, 100, 110],
                "person": "Alice",
                "recognition_score": 0.93,
                "ref_img_id": "reference-1",
                "confidence": 0.0,
            }
        ]
        store.observe_result(
            disappeared,
            disappeared_result,
            room_ids_by_track={},
        )
        assert store.status()["pending_post_roll_tracks"] == 1

        for frame_index in (7, 8):
            tail = colored_packet(frame_index, (0, 140 + frame_index * 10, 0))
            tail_result = result(tail, "Unknown", 0.0)
            tail_result.data["humans"] = []
            tail_result.data["faces"] = []
            store.observe_result(
                tail,
                tail_result,
                persist_human_log=False,
                room_ids_by_track={},
            )
        store.flush()

        videos = list((tmp_path / "media" / "human" / "videos").glob("*.mp4"))
        assert len(videos) == 1
        capture = cv2.VideoCapture(str(videos[0]))
        try:
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 7
            decoded: list[np.ndarray] = []
            while True:
                ok, decoded_frame = capture.read()
                if not ok:
                    break
                decoded.append(decoded_frame)
            assert len(decoded) == 7
            assert all(
                int(np.argmax(frame.mean(axis=(0, 1)))) == 0
                for frame in decoded[:2]
            )
            assert all(
                int(np.argmax(frame.mean(axis=(0, 1)))) == 1
                for frame in decoded[-2:]
            )
        finally:
            capture.release()
        with postgres_database.connection() as connection:
            row = connection.execute(
                "SELECT full_frame_video_frames FROM human_logs "
                "WHERE session_id = ? AND camera = ? AND track_id = ?",
                ("session-a", "camera-01", 13),
            ).fetchone()
        assert row is not None
        assert row["full_frame_video_frames"] == 7
        status = store.status()
        assert status["video_pre_roll_frames"] == 2
        assert status["video_post_roll_frames"] == 2
        assert status["pending_post_roll_tracks"] == 0
        assert status["pre_roll_buffered_bytes"] <= status["pre_roll_max_bytes"]
    finally:
        store.close()


def test_one_track_visiting_two_polygons_creates_two_detection_logs(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    detection_logs = DetectionLogStore(postgres_database)
    store = HumanLogStore(
        postgres_database,
        tmp_path / "media",
        detection_log_store=detection_logs,
    )
    locations = LocationStore(postgres_database)
    room_a = locations.create_room("Visited zone A")
    room_b = locations.create_room("Visited zone B")
    try:
        first = packet(1)
        first_result = result(first, "Alice", 0.93, face_quality=0.90)
        store.observe_result(
            first,
            first_result,
            persist_human_log=False,
            room_ids_by_track={13: room_a.id},
            observed_room_ids_by_track={13: {room_a.id}},
        )
        store.observe_result(
            first,
            first_result,
            room_ids_by_track={13: room_a.id},
            observed_room_ids_by_track={13: {room_a.id}},
        )

        second = packet(2)
        second_result = result(second, "Alice", 0.93, face_quality=0.85)
        store.observe_result(
            second,
            second_result,
            persist_human_log=False,
            room_ids_by_track={13: room_b.id},
            observed_room_ids_by_track={13: {room_b.id}},
        )
        store.observe_result(
            second,
            second_result,
            room_ids_by_track={13: room_b.id},
            observed_room_ids_by_track={13: {room_b.id}},
        )

        disappeared = packet(3)
        disappeared_result = result(disappeared, "Alice", 0.93)
        disappeared_result.data["humans"] = []
        disappeared_result.data["faces"] = []
        disappeared_result.data["disappeared_humans"] = [
            {
                "track_id": 13,
                "bbox": [10, 10, 100, 110],
                "person": "Alice",
                "recognition_score": 0.93,
                "ref_img_id": "reference-1",
                "confidence": 0.0,
            }
        ]
        store.observe_result(
            disappeared,
            disappeared_result,
            room_ids_by_track={},
            observed_room_ids_by_track={},
        )
        store.flush()

        records, total = detection_logs.list_filter(
            camera_id="camera-01",
            log_type="camera_rtsp",
        )
        assert total == 2
        assert {record.room_id for record in records} == {room_a.id, room_b.id}
        assert len({record.source_human_log_id for record in records}) == 1
        assert len({record.face_image for record in records}) == 1
        assert len({record.body_image for record in records}) == 1
        assert len({record.snapshot_image for record in records}) == 1
        assert len({record.video for record in records}) == 1
        assert all(record.face_image for record in records)
        assert all(record.body_image for record in records)
        assert all(record.snapshot_image for record in records)
        assert {
            record.source_event_key for record in records
        } == {
            "human-track:session-a:camera-01:13",
            f"human-track:session-a:camera-01:13:room:{room_b.id}",
        }
    finally:
        store.close()


def test_unknown_track_uses_frames_before_first_seen_and_after_disappearance(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    detection_logs = DetectionLogStore(postgres_database)
    store = HumanLogStore(
        postgres_database,
        tmp_path / "media",
        video_fps=5.0,
        video_pre_roll_frames=2,
        video_post_roll_frames=2,
        detection_log_store=detection_logs,
    )
    room = LocationStore(postgres_database).create_room("Unknown admission zone")

    def blank_result(source: FramePacket) -> TaskResult:
        item = result(source, "Unknown", 0.0)
        item.data["humans"] = []
        item.data["faces"] = []
        return item

    try:
        # These frames occur before the tracker sees any human.
        for frame_index in (1, 2):
            before = colored_packet(frame_index, (frame_index * 30, 0, 0))
            store.observe_result(
                before,
                blank_result(before),
                persist_human_log=False,
                room_ids_by_track={},
            )

        # Unknown and no valid face: polygon admission alone must start logging.
        entered = colored_packet(3, (0, 90, 0))
        entered_result = result(entered, "Unknown", 0.0)
        store.observe_result(
            entered,
            entered_result,
            persist_human_log=False,
            room_ids_by_track={13: room.id},
            observed_room_ids_by_track={13: {room.id}},
        )
        store.observe_result(
            entered,
            entered_result,
            room_ids_by_track={13: room.id},
            observed_room_ids_by_track={13: {room.id}},
        )

        disappeared = colored_packet(4, (0, 0, 90))
        disappeared_result = blank_result(disappeared)
        disappeared_result.data["disappeared_humans"] = [
            {
                "track_id": 13,
                "bbox": [10, 10, 100, 110],
                "person": "Unknown",
                "recognition_score": 0.0,
                "ref_img_id": None,
                "confidence": 0.0,
            }
        ]
        store.observe_result(
            disappeared,
            disappeared_result,
            room_ids_by_track={},
            observed_room_ids_by_track={},
        )
        store.flush()
        assert store.status()["pending_post_roll_tracks"] == 1
        assert detection_logs.list_filter(camera_id="camera-01")[1] == 0

        for frame_index in (5, 6):
            after = colored_packet(frame_index, (0, 0, frame_index * 25))
            store.observe_result(
                after,
                blank_result(after),
                persist_human_log=False,
                room_ids_by_track={},
            )
        store.flush()

        records, total = detection_logs.list_filter(camera_id="camera-01")
        assert total == 1
        assert records[0].person == "Unknown #13"
        assert records[0].room_id == room.id
        assert records[0].face_image is None
        assert records[0].body_image
        assert records[0].snapshot_image
        human_rows = store.list(track_id=13)
        assert len(human_rows) == 1
        assert human_rows[0]["name"] == "Unknown #13"

        # Re-finalizing an idempotent bridge uses the same persistence format
        # on the detection-log update path as on initial creation.
        store.video_post_roll_frames = 0
        store.observe_result(
            disappeared,
            disappeared_result,
            room_ids_by_track={13: room.id},
            observed_room_ids_by_track={13: {room.id}},
        )
        store.flush()
        updated = detection_logs.get_by_source_event_key(
            "human-track:session-a:camera-01:13"
        )
        assert updated is not None
        assert updated.person == "Unknown #13"
        assert store.status()["pending_post_roll_tracks"] == 0
        videos = list((tmp_path / "media" / "human" / "videos").glob("*.mp4"))
        assert len(videos) == 1
        capture = cv2.VideoCapture(str(videos[0]))
        try:
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
        finally:
            capture.release()
    finally:
        store.close()


def test_face_seen_after_polygon_exit_upgrades_identity_and_all_still_evidence(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    media_root = tmp_path / "media"
    detection_logs = DetectionLogStore(postgres_database)
    store = HumanLogStore(
        postgres_database,
        media_root,
        video_post_roll_frames=2,
        detection_log_store=detection_logs,
    )
    room = LocationStore(postgres_database).create_room("Late face zone")

    def blank_result(source: FramePacket) -> TaskResult:
        item = result(source, "Unknown", 0.0)
        item.data["humans"] = []
        item.data["faces"] = []
        return item

    try:
        inside = colored_packet(1, (255, 0, 0))
        inside_result = result(inside, "Unknown", 0.0)
        store.observe_result(
            inside,
            inside_result,
            persist_human_log=False,
            room_ids_by_track={13: room.id},
            observed_room_ids_by_track={13: {room.id}},
        )
        store.observe_result(
            inside,
            inside_result,
            room_ids_by_track={13: room.id},
            observed_room_ids_by_track={13: {room.id}},
        )

        # The face and identity become available only after polygon exit.
        outside_face = colored_packet(2, (0, 255, 0))
        store.observe_result(
            outside_face,
            result(outside_face, "Alice", 0.94, face_quality=0.95),
            persist_human_log=False,
            room_ids_by_track={},
            exited_track_ids={13},
        )

        disappeared = colored_packet(3, (0, 0, 255))
        disappeared_result = blank_result(disappeared)
        disappeared_result.data["disappeared_humans"] = [
            {
                "track_id": 13,
                "bbox": [10, 10, 100, 110],
                "person": "Alice",
                "recognition_score": 0.94,
                "ref_img_id": "reference-1",
                "confidence": 0.0,
            }
        ]
        store.observe_result(
            disappeared,
            disappeared_result,
            room_ids_by_track={},
        )
        for frame_index in (4, 5):
            tail = packet(frame_index)
            store.observe_result(
                tail,
                blank_result(tail),
                persist_human_log=False,
                room_ids_by_track={},
            )
        store.flush()

        records, total = detection_logs.list_filter(camera_id="camera-01")
        assert total == 1
        record = records[0]
        assert record.person == "Alice"
        assert record.room_id == room.id
        assert record.face_image
        assert record.body_image
        assert record.snapshot_image
        assert_jpeg_has_dominant_channel(
            media_root / "human" / "detected_faces" / Path(record.face_image).name,
            1,
        )
        assert_jpeg_has_dominant_channel(
            media_root / "human" / "body_images" / Path(record.body_image).name,
            1,
        )
        assert_jpeg_has_dominant_channel(
            media_root
            / "human"
            / "full_frame_images"
            / Path(record.snapshot_image).name,
            1,
        )
    finally:
        store.close()


def test_ranked_faces_are_bounded_and_snapshots_use_exact_best_face_frame(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    media_root = tmp_path / "media"
    detection_logs = DetectionLogStore(postgres_database)
    store = HumanLogStore(
        postgres_database,
        media_root,
        face_candidate_limit=3,
        snapshot_min_improvement=0.01,
        detection_log_store=detection_logs,
    )
    locations = LocationStore(postgres_database)
    building = locations.create_building("Ranked evidence building")
    section = locations.create_section(
        "Ranked evidence section", building_id=building.id
    )
    room = locations.create_room("Ranked evidence room", section_id=section.id)
    observations = [
        (1, (255, 0, 0), 0.90),
        # The old snapshot-quality threshold rejected this real improvement:
        # (0.92 - 0.90) * 0.30 is smaller than 0.01.
        (2, (0, 255, 0), 0.92),
        (3, (0, 0, 255), 0.70),
        (4, (255, 255, 0), 0.80),
    ]
    disappeared = colored_packet(5, (255, 0, 255))
    disappeared_result = result(disappeared, "Alice", 0.93)
    disappeared_result.data["humans"] = []
    disappeared_result.data["faces"] = []
    disappeared_result.data["disappeared_humans"] = [
        {
            "track_id": 13,
            "bbox": [10, 10, 100, 110],
            "person": "Alice",
            "recognition_score": 0.93,
            "ref_img_id": None,
            "confidence": 0.0,
        }
    ]

    try:
        for frame_index, color, quality in observations:
            observed = colored_packet(frame_index, color)
            store.observe_result(
                observed,
                result(observed, "Alice", 0.93, face_quality=quality),
                room_ids_by_track={13: room.id},
                counts_for_attendance=False,
            )
            with store._lock:
                candidates = store._face_candidates[
                    ("session-a", "camera-01", 13)
                ]
                assert len(candidates) <= 3

        store.observe_result(
            disappeared,
            disappeared_result,
            room_ids_by_track={13: room.id},
            counts_for_attendance=False,
        )
        store.flush()

        record = detection_logs.get_by_source_event_key(
            "human-track:session-a:camera-01:13"
        )
        assert record is not None
        assert_jpeg_has_dominant_channel(
            media_root / "human" / "detected_faces" / Path(record.face_image).name,
            1,
        )
        assert_jpeg_has_dominant_channel(
            media_root / "human" / "body_images" / Path(record.body_image).name,
            1,
        )
        assert_jpeg_has_dominant_channel(
            media_root
            / "human"
            / "full_frame_images"
            / Path(record.snapshot_image).name,
            1,
        )

        ranked = sorted((media_root / "human" / "face_videos").glob("*_rank_*.jpg"))
        assert len(ranked) == 3
        assert "frame_0000000002" in ranked[0].stem
        assert "frame_0000000001" in ranked[1].stem
        assert "frame_0000000004" in ranked[2].stem
        assert_jpeg_has_dominant_channel(ranked[0], 1)
        status = store.status()
        assert status["saved_ranked_face_candidates"] == 3
        assert status["ranked_candidate_tracks"] == 0
    finally:
        store.close()


def test_ranked_face_state_rolls_back_when_media_queue_is_full(
    tmp_path: Path,
    postgres_database: Database,
) -> None:
    store = HumanLogStore(
        postgres_database,
        tmp_path / "media",
        face_candidate_limit=2,
    )
    first = colored_packet(1, (255, 0, 0))
    rejected = colored_packet(2, (0, 255, 0))
    invalid = colored_packet(3, (0, 0, 255))
    invalid_result = result(invalid, "Alice", 0.93, face_quality=0.99)
    invalid_result.data["faces"][0]["quality_valid"] = False
    key = ("session-a", "camera-01", 13)
    try:
        store.observe_result(
            first,
            result(first, "Alice", 0.93, face_quality=0.80),
        )
        store.flush()

        with patch.object(store._queue, "put_nowait", side_effect=queue.Full):
            store.observe_result(
                rejected,
                result(rejected, "Alice", 0.93, face_quality=0.95),
            )

        store.observe_result(invalid, invalid_result)
        store.flush()
        with store._lock:
            assert [candidate.frame_index for candidate in store._face_candidates[key]] == [1]
            assert store._still_evidence[key].frame_index == 1
            assert store._best_face_scores[key] == 0.80
    finally:
        store.close()
