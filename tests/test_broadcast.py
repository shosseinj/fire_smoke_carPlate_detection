from __future__ import annotations

import time

import cv2
import numpy as np

from app.core.broadcast import AnnotatedBroadcastHub
from app.core.types import FramePacket, TaskName, TaskResult
from app.core.worker import TaskWorker


def packet(tasks: list[str]) -> FramePacket:
    return FramePacket(
        source_id="camera-07",
        frame=np.zeros((180, 320, 3), dtype=np.uint8),
        round_sequence=1,
        frame_index=12,
        captured_monotonic=time.monotonic(),
        captured_at_utc="2026-07-15T00:00:00+00:00",
        metadata={"assigned_tasks": tasks},
    )


def result(task: TaskName, data: dict) -> TaskResult:
    return TaskResult(
        task=task,
        source_id="camera-07",
        round_sequence=1,
        frame_index=12,
        captured_at_utc="2026-07-15T00:00:00+00:00",
        processed_at_utc="2026-07-15T00:00:01+00:00",
        processing_ms=2.0,
        data=data,
    )


def test_dual_task_frame_is_broadcast_only_after_both_exact_results() -> None:
    hub = AnnotatedBroadcastHub(enabled=True)
    subscriber_id, target = hub.subscribe()
    source_packet = packet(["fire_smoke", "plate_recognition"])
    fire_result = result(
        TaskName.FIRE_SMOKE,
        {
            "severity": "high",
            "tracks": [
                {"label": "fire", "confidence": 0.91, "bbox": [30, 40, 120, 130]}
            ],
        },
    )
    plate_result = result(
        TaskName.PLATE_RECOGNITION,
        {
            "plates": [
                {
                    "plate": "12B34567",
                    "detector_confidence": 0.88,
                    "bbox": [170, 90, 270, 135],
                }
            ]
        },
    )

    hub.publish_result(source_packet, fire_result)
    assert hub.latest("camera-07") is None

    hub.publish_result(source_packet, plate_result)
    encoded = hub.latest("camera-07")
    assert encoded is not None
    assert encoded.tasks == ("fire_smoke", "plate_recognition")
    delivered = target.get(timeout=1.0)
    assert delivered is not None
    assert delivered.source_id == "camera-07"
    target.task_done()
    image = cv2.imdecode(np.frombuffer(encoded.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    assert image.shape == (180, 320, 3)
    assert int(image.sum()) > 0

    hub.unsubscribe(subscriber_id)
    assert hub.status()["websocket_subscribers"] == 0

    hub.set_enabled(False)
    assert hub.latest("camera-07") is None
    assert hub.status()["enabled"] is False


def test_positive_results_print_detection_logs(capsys) -> None:
    TaskWorker._print_positive_detection(
        result(
            TaskName.FIRE_SMOKE,
            {
                "severity": "medium",
                "tracks": [{"label": "smoke"}, {"label": "fire"}],
            },
        )
    )
    TaskWorker._print_positive_detection(
        result(
            TaskName.PLATE_RECOGNITION,
            {"plates": [{"plate": "12B34567"}]},
        )
    )
    TaskWorker._print_positive_detection(
        result(
            TaskName.FACE_RECOGNITION,
            {"faces": [{"person": "Alice"}, {"person": "Unknown"}]},
        )
    )
    output = capsys.readouterr().out
    assert "task=fire_smoke" in output
    assert '"fire": 1' in output
    assert "task=plate_recognition" in output
    assert "12B34567" in output
    assert "task=face_recognition" in output
    assert "Alice" in output


def test_play_only_frame_is_broadcast_without_ai_result() -> None:
    hub = AnnotatedBroadcastHub(enabled=True)
    source_packet = packet([])

    hub.publish_passthrough(source_packet)

    encoded = hub.latest("camera-07")
    assert encoded is not None
    assert encoded.tasks == ()
    image = cv2.imdecode(np.frombuffer(encoded.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    assert int(image.sum()) > 0


def test_face_result_draws_recognized_identity() -> None:
    hub = AnnotatedBroadcastHub(enabled=True)
    source_packet = packet(["face_recognition"])
    hub.publish_result(
        source_packet,
        result(
            TaskName.FACE_RECOGNITION,
            {
                "faces": [
                    {
                        "bbox": [60, 45, 130, 135],
                        "person": "Alice",
                        "recognition_score": 0.92,
                        "track_id": 4,
                    }
                ]
            },
        ),
    )
    encoded = hub.latest("camera-07")
    assert encoded is not None
    assert encoded.tasks == ("face_recognition",)
    image = cv2.imdecode(np.frombuffer(encoded.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    assert int(image.sum()) > 0
