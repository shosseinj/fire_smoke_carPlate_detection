from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from app.core.recent_detection_service import (
    _build_payload_from_enriched_row,
    build_recent_detections_message,
)


def _runtime(media_root: Path, database: object | None = None) -> SimpleNamespace:
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
        database=database,
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


def test_recent_detection_uses_body_snapshot_when_face_is_missing(tmp_path: Path) -> None:
    snapshot = tmp_path / "body_images" / "body.jpg"
    snapshot.parent.mkdir(parents=True)
    assert cv2.imwrite(str(snapshot), np.full((40, 30, 3), 120, dtype=np.uint8))

    payload = _build_payload_from_enriched_row(
        _runtime(tmp_path),
        _row("/media/body_images/body.jpg"),
    )

    assert payload is not None
    assert payload["face_image_base64"] is None
    assert payload["body_image_base64"]
    assert payload["image_kind"] == "body"


def test_recent_detection_prefers_human_snapshot_over_face(tmp_path: Path) -> None:
    face = tmp_path / "detected_faces" / "face.jpg"
    snapshot = tmp_path / "body_images" / "body.jpg"
    face.parent.mkdir(parents=True)
    snapshot.parent.mkdir(parents=True)
    assert cv2.imwrite(str(face), np.full((20, 20, 3), 20, dtype=np.uint8))
    assert cv2.imwrite(str(snapshot), np.full((40, 30, 3), 120, dtype=np.uint8))

    row = _row("/media/body_images/body.jpg")
    row["face_image"] = "/media/detected_faces/face.jpg"
    payload = _build_payload_from_enriched_row(_runtime(tmp_path), row)

    assert payload is not None
    assert payload["face_image_base64"] is None
    assert payload["body_image_base64"]
    assert payload["image_kind"] == "body"


def test_known_body_snapshot_uses_upper_section_and_ref_img_id_reference(
    tmp_path: Path,
) -> None:
    body = tmp_path / "body_images" / "body.jpg"
    reference = tmp_path / "reference_images" / "reference.jpg"
    body.parent.mkdir(parents=True)
    reference.parent.mkdir(parents=True)
    assert cv2.imwrite(str(body), np.full((100, 80, 3), 120, dtype=np.uint8))
    assert cv2.imwrite(str(reference), np.full((20, 10, 3), 20, dtype=np.uint8))

    class Connection:
        def __enter__(self) -> "Connection":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, _query: str, params: tuple[int]) -> "Connection":
            assert params == (7,)
            return self

        def fetchone(self) -> dict[str, str]:
            return {"storage_key": "reference_images/reference.jpg"}

    row = _row("/media/body_images/body.jpg")
    row.update({"person": "Alice", "ref_img_id": "7"})
    payload = _build_payload_from_enriched_row(
        _runtime(tmp_path, SimpleNamespace(connection=lambda: Connection())),
        row,
    )

    assert payload is not None
    assert payload["image_kind"] == "body"
    encoded = base64.b64decode(str(payload["body_image_base64"]))
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[0] == 224
    assert decoded.shape[1] == 448
    # The stable composite contract is reference on the left, body on the right.
    assert int(decoded[112, 112].mean()) < 40
    assert int(decoded[112, 336].mean()) > 100


def test_recent_detections_message_contains_database_log(tmp_path: Path) -> None:
    snapshot = tmp_path / "body_images" / "body.jpg"
    snapshot.parent.mkdir(parents=True)
    assert cv2.imwrite(str(snapshot), np.full((40, 30, 3), 120, dtype=np.uint8))

    row = _row("/media/body_images/body.jpg")

    class Connection:
        def __enter__(self) -> "Connection":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, _query: str, _params: tuple[int]) -> "Connection":
            return self

        def fetchall(self) -> list[dict[str, object]]:
            return [row]

    message = build_recent_detections_message(
        SimpleNamespace(
            database=SimpleNamespace(connection=lambda: Connection()),
            settings=SimpleNamespace(saved_media_path=tmp_path),
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
    )

    assert message is not None
    assert message["type"] == "recent_detections"
    assert message["count"] == 1
    assert message["detections"][0]["body_image_base64"]
