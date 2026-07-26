from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from app.core.recent_detection_service import _build_payload_from_enriched_row


def _runtime(media_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        settings=SimpleNamespace(saved_media_path=media_root),
        general_settings=SimpleNamespace(
            get=lambda: SimpleNamespace(
                face_rec_score=0.45,
                confirmation_threshold=0.75,
            )
        ),
        registry=SimpleNamespace(get=lambda _camera_id: None),
        personnel_store=SimpleNamespace(
            get_image=lambda _image_id: None,
            list_images=lambda _personnel_id: [],
        ),
    )


def _row(snapshot_image: str | None = None) -> dict[str, object]:
    return {
        "id": 1,
        "person": "Unknown",
        "personnel_id": None,
        "confidence": 0.0,
        "detection_time": "2026-07-26T12:00:00+00:00",
        "camera_id": "camera-01",
        "snapshot_image": snapshot_image,
        "body_image": None,
        "face_image": None,
        "room_id": None,
        "room_name": None,
        "access_granted": False,
        "counts_for_attendance": True,
    }


def test_recent_detection_omits_entry_when_face_is_missing(tmp_path: Path) -> None:
    snapshot = tmp_path / "human_snapshots" / "body.jpg"
    snapshot.parent.mkdir(parents=True)
    assert cv2.imwrite(str(snapshot), np.full((40, 30, 3), 120, dtype=np.uint8))

    payload = _build_payload_from_enriched_row(
        _runtime(tmp_path),
        _row("/media/human_snapshots/body.jpg"),
    )

    assert payload is None


def test_recent_detection_prefers_face_over_human_snapshot(tmp_path: Path) -> None:
    face = tmp_path / "detected_faces" / "face.jpg"
    snapshot = tmp_path / "human_snapshots" / "body.jpg"
    face.parent.mkdir(parents=True)
    snapshot.parent.mkdir(parents=True)
    assert cv2.imwrite(str(face), np.full((20, 20, 3), 20, dtype=np.uint8))
    assert cv2.imwrite(str(snapshot), np.full((40, 30, 3), 120, dtype=np.uint8))

    row = _row("/media/human_snapshots/body.jpg")
    row["face_image"] = "/media/detected_faces/face.jpg"
    payload = _build_payload_from_enriched_row(_runtime(tmp_path), row)

    assert payload is not None
    assert payload["face_image_base64"]
    assert set(payload) == {
        "id",
        "area",
        "person",
        "full_name",
        "confidence",
        "detection_time",
        "face_image_base64",
        "access_granted",
        "counts_for_attendance",
        "classification",
    }
