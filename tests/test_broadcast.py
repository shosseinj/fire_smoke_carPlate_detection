from __future__ import annotations

import json
import struct
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.broadcast import (
    _source_frame_batch_payload,
    _source_frame_payload,
    get_runtime,
    router as broadcast_router,
)
from app.core.broadcast import (
    AnnotatedBroadcastHub,
    EncodedBroadcastFrame,
    SourceDrawSettings,
)
from app.core.source_registry import SourceChange, SourceRecord
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


def test_passthrough_frame_with_assigned_tasks_starts_as_source_only() -> None:
    hub = AnnotatedBroadcastHub(enabled=True, async_render=False)
    source_packet = packet(["face_recognition", "fire_smoke"])

    hub.publish_passthrough(source_packet)

    encoded = hub.latest("camera-07")
    assert encoded is not None
    assert encoded.tasks == ()


def test_video_wall_passthrough_is_available_before_ai_result() -> None:
    hub = AnnotatedBroadcastHub(enabled=True)
    source_packet = packet(["face_recognition"])

    hub.publish_passthrough(source_packet)

    encoded = hub.wait_next("camera-07", 0, timeout=1.0)
    assert encoded is not None
    assert encoded.tasks == ()
    assert cv2.imdecode(
        np.frombuffer(encoded.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
    ) is not None


def test_passthrough_render_does_not_mutate_worker_frame() -> None:
    hub = AnnotatedBroadcastHub(enabled=True, async_render=False)
    source_packet = packet(["plate_recognition"])
    original = source_packet.frame.copy()

    hub.publish_passthrough(source_packet)

    assert np.array_equal(source_packet.frame, original)


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
    assert 'header.render_profile === "full" ? "کامل"' in dashboard
    assert "if (broadcastSocket !== socket) return;" in dashboard
    assert "event.data instanceof ArrayBuffer" in dashboard
    assert "incomingFrameIndex < stats.frameIndex" in dashboard
    assert 'img.addEventListener("load"' in dashboard
    assert 'parameters.set("batch", "true")' in dashboard
    assert 'header.type === "source_frame_batch"' in dashboard
    assert "previewAvailable && !useJpegFallback" in dashboard


def test_source_frame_batch_envelope_contains_complete_frame_records() -> None:
    frame = EncodedBroadcastFrame(
        version=1,
        source_id="camera-07",
        frame_index=9,
        jpeg=b"full",
        wall_jpeg=b"wall",
        frame_width=640,
        frame_height=640,
        wall_width=240,
        wall_height=240,
        tasks=(),
        updated_monotonic=time.monotonic(),
    )
    record = _source_frame_payload(frame, wall=True, fullscreen_source=None)
    payload = _source_frame_batch_payload([record, record])
    outer_header_length = struct.unpack("!I", payload[:4])[0]
    outer_header = json.loads(payload[4 : 4 + outer_header_length])
    assert outer_header == {"type": "source_frame_batch", "count": 2}
    offset = 4 + outer_header_length
    first_record_length = struct.unpack("!I", payload[offset : offset + 4])[0]
    assert payload[offset + 4 : offset + 4 + first_record_length] == record


def test_dashboard_uses_source_uri_task_manager_identity() -> None:
    dashboard = (
        Path(__file__).parents[1] / "app" / "web" / "dashboard.html"
    ).read_text(encoding="utf-8")

    assert "source.source_id" not in dashboard
    assert "item.source_id" not in dashboard
    assert 'card.dataset.sourceId = source.source_uri' in dashboard
    assert 'item.source_uri === sourceId' in dashboard
    assert 'status.textContent = "منبع در مدیریت وظایف غیرفعال است"' in dashboard


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
            hub.publish_passthrough(packet([], "camera-07"))
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


def test_fullscreen_subscription_skips_cached_wall_only_frame() -> None:
    hub = AnnotatedBroadcastHub(
        enabled=True,
        wall_max_width=160,
        wall_max_height=160,
        async_render=False,
    )
    wall_subscriber_id, wall_queue = hub.subscribe(wall=True)
    hub.publish_passthrough(packet([], "camera-07"))
    cached = wall_queue.get_nowait()
    assert isinstance(cached, EncodedBroadcastFrame)
    assert (cached.frame_width, cached.frame_height) == (160, 90)

    fullscreen_subscriber_id, fullscreen_queue = hub.subscribe(
        wall=True,
        fullscreen_source="camera-07",
    )
    assert fullscreen_queue.empty()

    next_packet = packet([], "camera-07")
    hub.publish_passthrough(next_packet)
    fullscreen = fullscreen_queue.get_nowait()

    assert isinstance(fullscreen, EncodedBroadcastFrame)
    assert (fullscreen.frame_width, fullscreen.frame_height) == (320, 180)
    hub.unsubscribe(fullscreen_subscriber_id)
    hub.unsubscribe(wall_subscriber_id)


def test_source_only_fullscreen_subscription_skips_cached_wall_only_frame() -> None:
    hub = AnnotatedBroadcastHub(
        enabled=True,
        wall_max_width=160,
        wall_max_height=160,
        async_render=False,
    )
    wall_subscriber_id, wall_queue = hub.subscribe_source_only(wall=True)
    hub.publish_source_only(packet([], "camera-07"))
    cached = wall_queue.get_nowait()
    assert isinstance(cached, EncodedBroadcastFrame)
    assert (cached.frame_width, cached.frame_height) == (160, 90)

    fullscreen_subscriber_id, fullscreen_queue = hub.subscribe_source_only(
        wall=True,
        fullscreen_source="camera-07",
    )
    assert fullscreen_queue.empty()

    next_packet = packet([], "camera-07")
    hub.publish_source_only(next_packet)
    fullscreen = fullscreen_queue.get_nowait()

    assert isinstance(fullscreen, EncodedBroadcastFrame)
    assert (fullscreen.frame_width, fullscreen.frame_height) == (320, 180)
    hub.unsubscribe_source_only(fullscreen_subscriber_id)
    hub.unsubscribe_source_only(wall_subscriber_id)


def test_source_only_broadcast_never_replaces_new_frame_with_stale_frame() -> None:
    hub = AnnotatedBroadcastHub(enabled=True, async_render=False)
    subscriber_id, target = hub.subscribe_source_only(wall=True)
    newest = replace(packet([], "camera-07"), frame_index=20)
    stale = replace(packet([], "camera-07"), frame_index=19)

    hub.publish_source_only(newest)
    delivered = target.get_nowait()
    hub.publish_source_only(stale)

    assert isinstance(delivered, EncodedBroadcastFrame)
    assert delivered.frame_index == 20
    assert target.empty()
    assert hub._source_only_latest["camera-07"].frame_index == 20
    hub.unsubscribe_source_only(subscriber_id)


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


def test_source_video_wall_websocket_sends_unannotated_frames() -> None:
    hub = AnnotatedBroadcastHub(
        enabled=True, wall_max_width=160, wall_max_height=160, async_render=False
    )
    hub.publish_source_only(packet(["face_recognition"]))
    runtime = SimpleNamespace(
        broadcast=hub,
        registry=SimpleNamespace(get=lambda source_id: object()),
    )
    app = FastAPI()
    app.include_router(broadcast_router)
    app.dependency_overrides[get_runtime] = lambda: runtime

    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/video-wall/ws") as websocket:
            payload = websocket.receive_bytes()
            header_length = struct.unpack("!I", payload[:4])[0]
            header = json.loads(payload[4 : 4 + header_length])
            image = cv2.imdecode(
                np.frombuffer(payload[4 + header_length :], dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            assert header["type"] == "source_frame"
            assert header["ai_processed"] is False
            assert header["render_profile"] == "wall"
            assert image is not None
            hub.set_enabled(False)


def test_dashboard_can_switch_between_source_only_and_ai_streams() -> None:
    dashboard = (Path(__file__).parents[1] / "app" / "web" / "dashboard.html").read_text(
        encoding="utf-8"
    )
    assert '"/api/v1/video-wall/ws"' in dashboard
    assert '"/api/v1/broadcast/ws"' in dashboard
    assert "sourceOnlyWall" in dashboard

def test_source_change_with_previous_uri_clears_old_broadcast_state() -> None:
    hub = AnnotatedBroadcastHub(enabled=True, async_render=False)
    hub.publish_passthrough(packet([], "video-old.mp4"))
    hub.publish_passthrough(packet([], "video-new.mp4"))
    hub._latest_face_results["video-old.mp4"] = (0.0, result(TaskName.FIRE_SMOKE, {}))  # type: ignore[index]
    hub._source_zones["video-old.mp4"] = [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]]
    hub._source_draw_settings["video-old.mp4"] = SourceDrawSettings()

    hub.publish_source_change(
        SourceChange(
            action="updated",
            source_uri="video-new.mp4",
            previous_source_uri="video-old.mp4",
            revision=2,
            record=SourceRecord(source_uri="video-new.mp4", name="New"),
        )
    )

    assert "video-old.mp4" not in hub._pending
    assert "video-old.mp4" not in hub._latest
    assert "video-old.mp4" not in hub._latest_face_results
    assert "video-old.mp4" not in hub._source_zones
    assert "video-old.mp4" not in hub._source_draw_settings
    assert "video-new.mp4" not in hub._pending
    assert "video-new.mp4" not in hub._latest


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


def test_fire_overlay_is_retained_briefly_for_intermediate_frames() -> None:
    hub = AnnotatedBroadcastHub(
        enabled=True,
        async_render=False,
        face_overlay_ttl_ms=500,
    )
    fire_packet = packet(["fire_smoke"])
    hub.publish_result(
        fire_packet,
        result(
            TaskName.FIRE_SMOKE,
            {
                "tracks": [
                    {
                        "label": "fire",
                        "confidence": 0.9,
                        "bbox": [30, 40, 120, 130],
                    }
                ]
            },
        ),
    )
    next_packet = replace(
        fire_packet,
        round_sequence=2,
        frame_index=13,
        captured_monotonic=time.monotonic(),
    )

    hub.publish_passthrough(next_packet)

    encoded = hub.latest("camera-07")
    assert encoded is not None
    assert encoded.frame_index == 13
    assert encoded.tasks == ("fire_smoke",)
    assert hub.status()["overlay_cache_hits"] == 1


# ── Polygon drawing tests ────────────────────────────────────────────


def test_draw_zones_default_enabled() -> None:
    hub = AnnotatedBroadcastHub(enabled=True)
    assert hub.draw_zones is True


def test_set_draw_zones_toggles_flag() -> None:
    hub = AnnotatedBroadcastHub(enabled=True)
    hub.set_draw_zones(False)
    assert hub.draw_zones is False
    hub.set_draw_zones(True)
    assert hub.draw_zones is True


def test_set_source_zones_stores_polygons() -> None:
    hub = AnnotatedBroadcastHub(enabled=True)
    zones = [[[0, 0], [0, 100], [100, 100], [100, 0]]]
    hub.set_source_zones("cam-01", zones)
    assert hub._source_zones["cam-01"] == zones


def test_clear_source_zones_removes_polygons() -> None:
    hub = AnnotatedBroadcastHub(enabled=True)
    hub.set_source_zones("cam-01", [[[0, 0], [0, 100], [100, 100], [100, 0]]])
    hub.clear_source_zones("cam-01")
    assert "cam-01" not in hub._source_zones


def test_draw_zones_renders_polygon_on_frame() -> None:
    """Verify polygon drawing modifies pixels when draw_zones=True and zones exist."""
    hub = AnnotatedBroadcastHub(enabled=True, async_render=False)
    zones = [[[10, 10], [10, 100], [100, 100], [100, 10]]]
    hub.set_source_zones("camera-07", zones)
    hub.draw_zones = True
    source_packet = packet(["face_recognition"])
    hub.publish_result(
        source_packet,
        result(TaskName.FACE_RECOGNITION, {"humans": [], "faces": []}),
    )
    encoded = hub.latest("camera-07")
    assert encoded is not None
    image = cv2.imdecode(np.frombuffer(encoded.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    # The polygon area should have non-zero pixel values from the drawn overlay
    poly_region = image[10:100, 10:100]
    assert float(poly_region.mean()) > 0.0


def test_draw_zones_suppressed_when_flag_false() -> None:
    """Verify no polygon drawn when draw_zones=False."""
    hub = AnnotatedBroadcastHub(enabled=True, async_render=False)
    zones = [[[10, 10], [10, 100], [100, 100], [100, 10]]]
    hub.set_source_zones("camera-07", zones)
    hub.draw_zones = False
    source_packet = packet(["face_recognition"])
    hub.publish_result(
        source_packet,
        result(TaskName.FACE_RECOGNITION, {"humans": [], "faces": []}),
    )
    encoded = hub.latest("camera-07")
    assert encoded is not None
    image = cv2.imdecode(np.frombuffer(encoded.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    # Without draw_zones, no overlay should be drawn. The header overlay
    # still exists, but the polygon region should have no additional color
    # from the specific polygon fill colors (100, 100, 255 BGR=blue).
    # We verify the area outside header has similar mean to a no-polygon render.
    ref_hub = AnnotatedBroadcastHub(enabled=True, async_render=False)
    ref_hub.publish_result(
        packet(["face_recognition"]),
        result(TaskName.FACE_RECOGNITION, {"humans": [], "faces": []}),
    )
    ref_encoded = ref_hub.latest("camera-07")
    assert ref_encoded is not None
    ref_image = cv2.imdecode(np.frombuffer(ref_encoded.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert ref_image is not None
    # Compare header-excluded region (below pixel row 76)
    region = image[80:, 10:100]
    ref_region = ref_image[80:, 10:100]
    assert abs(float(region.mean()) - float(ref_region.mean())) < 5.0


def test_draw_zones_no_op_when_no_zones() -> None:
    """Verify rendering works normally with no zones set."""
    hub = AnnotatedBroadcastHub(enabled=True, async_render=False)
    hub.draw_zones = True
    source_packet = packet(["face_recognition"])
    hub.publish_result(
        source_packet,
        result(TaskName.FACE_RECOGNITION, {"humans": [], "faces": []}),
    )
    encoded = hub.latest("camera-07")
    assert encoded is not None
    assert encoded.tasks == ("face_recognition",)


def test_source_draw_human_false_suppresses_human_boxes() -> None:
    hub = AnnotatedBroadcastHub(enabled=True, async_render=False)
    hub.set_source_draw_settings("camera-07", SourceDrawSettings(draw_human=False))
    source_packet = packet(["face_recognition"])
    hub.publish_result(
        source_packet,
        result(
            TaskName.FACE_RECOGNITION,
            {
                "humans": [{"bbox": [60, 45, 130, 135], "person": "Unknown", "track_id": 4}],
                "faces": [],
            },
        ),
    )
    encoded = hub.latest("camera-07")
    assert encoded is not None
    image = cv2.imdecode(np.frombuffer(encoded.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    assert float(image[80:135, 60:130].mean()) == 0.0


def test_source_draw_zone_false_suppresses_zone_overlay() -> None:
    hub = AnnotatedBroadcastHub(enabled=True, async_render=False)
    hub.set_source_zones("camera-07", [[[10, 10], [10, 100], [100, 100], [100, 10]]])
    hub.set_source_draw_settings("camera-07", SourceDrawSettings(draw_zone=False))
    hub.publish_result(packet(["face_recognition"]), result(TaskName.FACE_RECOGNITION, {"humans": [], "faces": []}))
    encoded = hub.latest("camera-07")
    assert encoded is not None
    image = cv2.imdecode(np.frombuffer(encoded.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    assert float(image[80:, 10:100].mean()) == 0.0


def test_source_draw_fire_and_smoke_independently_filter_hazard_boxes() -> None:
    hub = AnnotatedBroadcastHub(enabled=True, async_render=False)
    hub.set_source_draw_settings(
        "camera-07",
        SourceDrawSettings(draw_fire=False, draw_smoke=True),
    )
    hub.publish_result(
        packet(["fire_smoke"]),
        result(
            TaskName.FIRE_SMOKE,
            {
                "tracks": [
                    {"label": "fire", "confidence": 0.9, "bbox": [20, 80, 80, 140], "confirmed": True},
                    {"label": "smoke", "confidence": 0.8, "bbox": [120, 80, 180, 140], "confirmed": True},
                ],
            },
        ),
    )
    encoded = hub.latest("camera-07")
    assert encoded is not None
    image = cv2.imdecode(np.frombuffer(encoded.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    assert float(image[80:140, 20:80].mean()) == 0.0
    assert float(image[80:140, 120:180].mean()) > 0.0


def test_fire_overlay_draws_even_before_confirmation() -> None:
    hub = AnnotatedBroadcastHub(enabled=True, async_render=False)
    hub.publish_result(
        packet(["fire_smoke"]),
        result(
            TaskName.FIRE_SMOKE,
            {
                "tracks": [
                    {"label": "fire", "confidence": 0.9, "bbox": [20, 80, 80, 140], "confirmed": False, "alert_active": False},
                ],
            },
        ),
    )
    encoded = hub.latest("camera-07")
    assert encoded is not None
    image = cv2.imdecode(np.frombuffer(encoded.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    assert float(image[80:140, 20:80].mean()) > 0.0


def test_fire_overlay_draws_raw_detections_when_tracks_are_not_ready() -> None:
    hub = AnnotatedBroadcastHub(enabled=True, async_render=False)
    hub.publish_result(
        packet(["fire_smoke"]),
        result(
            TaskName.FIRE_SMOKE,
            {
                "detections": [
                    {"label": "fire", "confidence": 0.9, "bbox": [20, 80, 80, 140]},
                ],
                "tracks": [],
            },
        ),
    )
    encoded = hub.latest("camera-07")
    assert encoded is not None
    image = cv2.imdecode(np.frombuffer(encoded.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    assert float(image[80:140, 20:80].mean()) > 0.0


def test_source_draw_vehicle_and_plate_can_disable_each_box_type() -> None:
    payload = {
        "plates": [
            {
                "plate": "12B34567",
                "detector_confidence": 0.88,
                "bbox": [170, 90, 230, 130],
                "vehicle_bbox": [140, 70, 260, 160],
            }
        ]
    }
    enabled_hub = AnnotatedBroadcastHub(enabled=True, async_render=False)
    enabled_hub.set_source_draw_settings(
        "camera-07",
        SourceDrawSettings(draw_vehicle=True, draw_plate=True),
    )
    enabled_hub.publish_result(packet(["plate_recognition"]), result(TaskName.PLATE_RECOGNITION, payload))
    enabled_encoded = enabled_hub.latest("camera-07")
    assert enabled_encoded is not None
    enabled_image = cv2.imdecode(np.frombuffer(enabled_encoded.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert enabled_image is not None

    disabled_hub = AnnotatedBroadcastHub(enabled=True, async_render=False)
    disabled_hub.set_source_draw_settings(
        "camera-07",
        SourceDrawSettings(draw_vehicle=False, draw_plate=True),
    )
    disabled_hub.publish_result(packet(["plate_recognition"]), result(TaskName.PLATE_RECOGNITION, payload))
    disabled_encoded = disabled_hub.latest("camera-07")
    assert disabled_encoded is not None
    disabled_image = cv2.imdecode(np.frombuffer(disabled_encoded.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert disabled_image is not None

    assert float(disabled_image[90:130, 170:230].mean()) > 0.0
    enabled_vehicle_region = float(enabled_image[70:160, 140:160].mean())
    disabled_vehicle_region = float(disabled_image[70:160, 140:160].mean())
    assert enabled_vehicle_region > disabled_vehicle_region + 2.0


def test_source_only_renderer_uses_configured_bounded_worker_pool() -> None:
    hub = AnnotatedBroadcastHub(
        enabled=True,
        source_only_render_threads=3,
        render_threads=2,
    )
    try:
        state = hub.status()
        assert state["source_only_render_threads"] == 3
        assert state["render_threads"] == 2
    finally:
        hub.close()
