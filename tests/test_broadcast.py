from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.broadcast import get_runtime, router as broadcast_router
from app.core.broadcast import AnnotatedBroadcastHub
from app.core.types import FramePacket, TaskName, TaskResult
from app.core.worker import TaskWorker


def packet(tasks: list[str], source_id: str = "camera-07") -> FramePacket:
    return FramePacket(
        source_id=source_id,
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


def test_fire_frame_is_eager_then_upgraded_with_both_exact_results() -> None:
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
    fire_encoded = hub.wait_next("camera-07", 0, timeout=1.0)
    assert fire_encoded is not None
    first_version = fire_encoded.version
    fire_delivered = target.get(timeout=1.0)
    assert fire_delivered is not None
    assert fire_delivered.frame_index == source_packet.frame_index
    target.task_done()

    hub.publish_result(source_packet, plate_result)
    encoded = hub.wait_next("camera-07", first_version, timeout=1.0)
    assert encoded is not None
    assert encoded.version > first_version
    assert encoded.frame_index == source_packet.frame_index
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

    encoded = hub.wait_next("camera-07", 0, timeout=1.0)
    assert encoded is not None
    assert encoded.tasks == ()
    image = cv2.imdecode(np.frombuffer(encoded.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    assert int(image.sum()) > 0


def test_wall_rendition_is_bounded_and_full_rendition_keeps_camera_size() -> None:
    hub = AnnotatedBroadcastHub(
        enabled=True,
        wall_max_width=160,
        wall_max_height=160,
    )

    hub.publish_passthrough(packet([]))

    encoded = hub.wait_next("camera-07", 0, timeout=1.0)
    assert encoded is not None
    full_jpeg, full_width, full_height, full_profile = encoded.rendition(
        full_resolution=True
    )
    wall_jpeg, wall_width, wall_height, wall_profile = encoded.rendition(
        full_resolution=False
    )
    assert (full_width, full_height, full_profile) == (320, 180, "full")
    assert (wall_width, wall_height, wall_profile) == (160, 90, "wall")
    assert cv2.imdecode(
        np.frombuffer(full_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
    ).shape == (180, 320, 3)
    assert cv2.imdecode(
        np.frombuffer(wall_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
    ).shape == (90, 160, 3)
    status = hub.status()
    assert status["wall_max_width"] == 160
    assert status["wall_max_height"] == 160
    assert status["active_streams"]["camera-07"]["full_resolution"] == [320, 180]
    assert status["active_streams"]["camera-07"]["wall_resolution"] == [160, 90]
    assert status["wall_encoded_bytes"] < status["full_encoded_bytes"]


def test_dashboard_requests_wall_profile_and_reconnects_for_fullscreen_source() -> None:
    dashboard = (
        Path(__file__).parents[1] / "app" / "web" / "dashboard.html"
    ).read_text(encoding="utf-8")

    # JPEG fallback (default) connects broadcast WS without metadata_only to receive full JPEG frames
    assert 'new URLSearchParams()' in dashboard
    assert 'if (!useJpegFallback) {' in dashboard
    assert 'parameters.set("metadata_only", "true")' in dashboard
    assert 'parameters.set("fullscreen_source", fullscreenSourceId)' in dashboard
    assert 'fullscreenElement?.classList.contains("camera-card")' in dashboard
    assert "reconnectBroadcastSocket()" in dashboard
    assert 'header.render_profile || "Annotated"' in dashboard
    assert "if (broadcastSocket !== socket) return;" in dashboard
    assert "event.data instanceof ArrayBuffer" in dashboard


def test_websocket_sends_fullscreen_source_full_and_other_sources_as_wall() -> None:
    hub = AnnotatedBroadcastHub(
        enabled=True,
        wall_max_width=160,
        wall_max_height=160,
    )
    hub.publish_passthrough(packet([], "camera-07"))
    hub.publish_passthrough(packet([], "camera-08"))
    known_sources = {"camera-07", "camera-08"}
    runtime = SimpleNamespace(
        broadcast=hub,
        registry=SimpleNamespace(
            get=lambda source_id: object() if source_id in known_sources else None
        ),
    )
    app = FastAPI()
    app.include_router(broadcast_router)
    app.dependency_overrides[get_runtime] = lambda: runtime

    with TestClient(app) as client:
        with client.websocket_connect(
            "/api/v1/broadcast/ws?wall=true&fullscreen_source=camera-07"
        ) as websocket:
            received: dict[str, tuple[dict, bytes]] = {}
            for _ in known_sources:
                payload = websocket.receive_bytes()
                header_length = struct.unpack("!I", payload[:4])[0]
                header = json.loads(payload[4 : 4 + header_length])
                received[header["source_id"]] = (
                    header,
                    payload[4 + header_length :],
                )
            hub.set_enabled(False)

    fullscreen_header, fullscreen_jpeg = received["camera-07"]
    wall_header, wall_jpeg = received["camera-08"]
    assert fullscreen_header["render_profile"] == "full"
    assert (fullscreen_header["frame_width"], fullscreen_header["frame_height"]) == (
        320,
        180,
    )
    assert cv2.imdecode(
        np.frombuffer(fullscreen_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
    ).shape == (180, 320, 3)
    assert wall_header["render_profile"] == "wall"
    assert (wall_header["frame_width"], wall_header["frame_height"]) == (160, 90)
    assert cv2.imdecode(
        np.frombuffer(wall_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
    ).shape == (90, 160, 3)


def test_websocket_keeps_default_full_resolution_for_existing_clients() -> None:
    hub = AnnotatedBroadcastHub(enabled=True, wall_max_width=160, wall_max_height=160)
    hub.publish_passthrough(packet([]))
    runtime = SimpleNamespace(
        broadcast=hub,
        registry=SimpleNamespace(get=lambda source_id: object()),
    )
    app = FastAPI()
    app.include_router(broadcast_router)
    app.dependency_overrides[get_runtime] = lambda: runtime

    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/broadcast/ws") as websocket:
            payload = websocket.receive_bytes()
            header_length = struct.unpack("!I", payload[:4])[0]
            header = json.loads(payload[4 : 4 + header_length])
            hub.set_enabled(False)

    assert header["render_profile"] == "full"
    assert (header["frame_width"], header["frame_height"]) == (320, 180)


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
    encoded = hub.wait_next("camera-07", 0, timeout=1.0)
    assert encoded is not None


def test_human_result_draws_human_bounding_box() -> None:
    hub = AnnotatedBroadcastHub(enabled=True, async_render=False)
    source_packet = packet(["face_recognition"])
    hub.publish_result(
        source_packet,
        result(
            TaskName.FACE_RECOGNITION,
            {
                "humans": [
                    {
                        "bbox": [60, 45, 130, 135],
                        "person": "Unknown",
                        "track_id": 4,
                    }
                ],
                "faces": [],
            },
        ),
    )

    encoded = hub.latest("camera-07")
    assert encoded is not None
    assert encoded.tasks == ("face_recognition",)
    image = cv2.imdecode(np.frombuffer(encoded.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    assert int(image.sum()) > 0
    image = cv2.imdecode(np.frombuffer(encoded.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    assert float(image[45:135, 60:130].mean()) > 0.0


def test_face_overlay_is_retained_briefly_for_intermediate_frames() -> None:
    hub = AnnotatedBroadcastHub(enabled=True, async_render=False, face_overlay_ttl_ms=500)
    face_packet = packet(["face_recognition"])
    hub.publish_result(
        face_packet,
        result(
            TaskName.FACE_RECOGNITION,
            {"humans": [{"bbox": [60, 45, 130, 135], "track_id": 4}], "faces": []},
        ),
    )
    next_packet = packet(["face_recognition", "fire_smoke"])
    next_packet = FramePacket(
        source_id=next_packet.source_id,
        frame=next_packet.frame,
        round_sequence=2,
        frame_index=13,
        captured_monotonic=time.monotonic(),
        captured_at_utc=next_packet.captured_at_utc,
        metadata=next_packet.metadata,
    )
    hub.publish_result(
        next_packet,
        TaskResult(
            task=TaskName.FIRE_SMOKE,
            source_id=next_packet.source_id,
            round_sequence=2,
            frame_index=13,
            captured_at_utc=next_packet.captured_at_utc,
            processed_at_utc=next_packet.captured_at_utc,
            processing_ms=1.0,
            data={"tracks": []},
        ),
    )

    encoded = hub.latest("camera-07")
    assert encoded is not None
    assert hub.status()["face_overlay_cache_hits"] == 1
