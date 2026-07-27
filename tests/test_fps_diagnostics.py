from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from app.core.deepstream_ingestor import DeepStreamIngestor, DeepStreamSourceState
from app.core.fps_diagnostics import build_fps_report


def snapshot(
    *,
    decoded: int,
    received: int,
    submitted: int,
    worker_accepted: int,
    worker_processed: int,
    worker_replaced: int = 0,
    pending_sources: int = 0,
    rate_limited: int = 0,
    pre_submit_replaced: int = 0,
) -> dict:
    return {
        "video_ingestor": {
            "backend": "deepstream",
            "fps_control": "sources.fps",
            "sources": {
                "camera-01": {
                    "fps_mode": "override",
                    "delivery_target_fps": 5.0,
                    "decoded_samples": decoded,
                    "rate_limited_frames": rate_limited,
                    "received_frames": received,
                    "pre_submit_replacements": pre_submit_replaced,
                    "submitted_frames": submitted,
                    "last_frame_age_seconds": 0.05,
                }
            },
        },
        "workers": {
            "fire_smoke": {
                "counters": {
                    "frames": worker_processed,
                    "failed_batches": 0,
                    "last_batch_size": 1,
                    "last_batch_ms": 20.0,
                    "last_error": None,
                    "frames_by_source": {"camera-01": worker_processed},
                    "failed_frames_by_source": {},
                },
                "buffer": {
                    "accepted": worker_accepted,
                    "stale_replaced": worker_replaced,
                    "pending_sources": pending_sources,
                    "accepted_by_source": {"camera-01": worker_accepted},
                    "stale_replaced_by_source": {"camera-01": worker_replaced},
                },
            }
        },
        "broadcast": {
            "rendered_frames": submitted,
            "full_encoded_bytes": submitted * 100_000,
            "wall_encoded_bytes": submitted * 20_000,
            "encode_failures": 0,
        },
    }


def test_fps_report_identifies_configured_five_fps_cap() -> None:
    before = snapshot(
        decoded=100,
        received=20,
        submitted=20,
        worker_accepted=20,
        worker_processed=20,
        rate_limited=80,
    )
    after = snapshot(
        decoded=160,
        received=30,
        submitted=30,
        worker_accepted=30,
        worker_processed=30,
        rate_limited=130,
    )

    report = build_fps_report(
        before,
        after,
        sample_seconds=2.0,
        expected_fps=25.0,
        camera_tasks={"camera-01": ("fire_smoke",)},
    )

    camera = report["cameras"]["camera-01"]
    assert camera["appsink_sample_fps"] == 30.0
    assert camera["rate_limited_fps"] == 25.0
    assert camera["admitted_fps"] == 5.0
    assert camera["router_frame_fps"] == 5.0
    assert camera["target_utilization_percent"] == 100.0
    assert camera["expected_utilization_percent"] == 20.0
    assert camera["causes"] == ["configured_ingest_cap"]
    assert report["overall"] == "limited"
    assert report["primary_causes"] == [
        {"code": "configured_ingest_cap", "affected_sections": 1}
    ]
    assert report["fps_control"] == "sources.fps"
    assert report["overridden_fps_sources"] == 1
    assert report["native_fps_sources"] == 0


def test_fps_report_uses_native_source_fps_without_a_global_cap() -> None:
    before = snapshot(
        decoded=100,
        received=100,
        submitted=100,
        worker_accepted=100,
        worker_processed=100,
    )
    after = snapshot(
        decoded=150,
        received=150,
        submitted=150,
        worker_accepted=150,
        worker_processed=150,
    )
    for value in (before, after):
        source = value["video_ingestor"]["sources"]["camera-01"]
        source["delivery_target_fps"] = None
        source["fps_mode"] = "native"
        source["native_fps"] = 25.0

    report = build_fps_report(
        before,
        after,
        sample_seconds=2.0,
        expected_fps=25.0,
        camera_tasks={"camera-01": ("fire_smoke",)},
    )

    camera = report["cameras"]["camera-01"]
    assert camera["fps_mode"] == "native"
    assert camera["configured_target_fps"] is None
    assert camera["native_fps"] == 25.0
    assert camera["router_frame_fps"] == 25.0
    assert "configured_ingest_cap" not in camera["causes"]
    assert report["native_fps_sources"] == 1


def test_fps_report_marks_a_native_source_with_no_frames_unavailable() -> None:
    before = snapshot(
        decoded=0,
        received=0,
        submitted=0,
        worker_accepted=0,
        worker_processed=0,
    )
    after = snapshot(
        decoded=0,
        received=0,
        submitted=0,
        worker_accepted=0,
        worker_processed=0,
    )
    for value in (before, after):
        source = value["video_ingestor"]["sources"]["camera-01"]
        source["delivery_target_fps"] = None
        source["fps_mode"] = "native"
        source["source_type"] = "rtsp"

    report = build_fps_report(
        before,
        after,
        sample_seconds=2.0,
        expected_fps=25.0,
        camera_tasks={"camera-01": ("fire_smoke",)},
    )

    camera = report["cameras"]["camera-01"]
    assert camera["source_type"] == "rtsp"
    assert camera["status"] == "limited"
    assert camera["causes"] == ["source_unavailable"]
    assert report["primary_causes"] == [
        {"code": "source_unavailable", "affected_sections": 1}
    ]


def test_fps_report_identifies_decode_and_worker_pressure() -> None:
    before = snapshot(
        decoded=100,
        received=100,
        submitted=100,
        worker_accepted=100,
        worker_processed=100,
    )
    after = snapshot(
        decoded=106,
        received=106,
        submitted=106,
        worker_accepted=150,
        worker_processed=120,
        worker_replaced=10,
        pending_sources=2,
    )
    before["video_ingestor"]["sources"]["camera-01"]["delivery_target_fps"] = 25.0
    after["video_ingestor"]["sources"]["camera-01"]["delivery_target_fps"] = 25.0

    report = build_fps_report(
        before,
        after,
        sample_seconds=2.0,
        expected_fps=25.0,
        camera_tasks={"camera-01": ("fire_smoke",)},
    )

    assert "source_decode_starvation" in report["cameras"]["camera-01"]["causes"]
    assert "task_worker_backpressure" in report["cameras"]["camera-01"]["causes"]
    assert report["cameras"]["camera-01"]["tasks_detail"]["fire_smoke"] == {
        "accepted_fps": 25.0,
        "processed_fps": 10.0,
        "latest_frame_replacement_fps": 5.0,
        "failed_fps": 0.0,
        "causes": ["task_worker_backpressure"],
    }
    assert report["workers"]["fire_smoke"]["input_fps"] == 25.0
    assert report["workers"]["fire_smoke"]["processed_fps"] == 10.0
    assert report["workers"]["fire_smoke"]["latest_frame_replacement_fps"] == 5.0
    assert report["workers"]["fire_smoke"]["causes"] == [
        "latest_frame_replacement",
        "processor_throughput_shortfall",
    ]


def test_deepstream_delivery_gate_counts_rate_limited_samples() -> None:
    gst = SimpleNamespace(FlowReturn=SimpleNamespace(OK="ok", ERROR="error"))
    sink = SimpleNamespace(emit=lambda name: object())
    state = DeepStreamSourceState(
        source_id="camera-01",
        source_uri="local.mp4",
        display_uri="local.mp4",
        source_type="video_file",
        pipeline=None,
        source=None,
        sink=sink,
        bus=None,
        bus_handler_id=0,
        pipeline_handler_id=0,
        source_pad_handler_id=0,
        sink_handler_id=0,
        frame_width=640,
        frame_height=640,
        delivery_target_fps=5.0,
        next_frame_due_monotonic=time.monotonic() + 0.2,
        last_frame_monotonic=time.monotonic(),
    )
    ingestor = object.__new__(DeepStreamIngestor)
    ingestor._gst = gst
    ingestor._glib = object()
    ingestor._lock = threading.RLock()
    ingestor._states = {state.source_id: state}

    assert ingestor._on_new_sample(sink, state.source_id) == "ok"
    assert state.decoded_samples == 1
    assert state.rate_limited_frames == 1
    assert state.received_frames == 0


def _admitting_deepstream_sample(
    *,
    delivery_target_fps: float | None,
    next_frame_due_monotonic: float = 0.0,
) -> tuple[DeepStreamIngestor, DeepStreamSourceState, SimpleNamespace]:
    structure = SimpleNamespace(
        get_value=lambda name: {
            "width": 1,
            "height": 1,
            "format": "BGRx",
        }[name]
    )
    caps = SimpleNamespace(get_structure=lambda _: structure)
    buffer = SimpleNamespace(
        extract_dup=lambda _offset, _size: bytes([1, 2, 3, 255]),
        get_size=lambda: 4,
        pts=-1,
    )
    sample = SimpleNamespace(
        get_caps=lambda: caps,
        get_buffer=lambda: buffer,
    )
    gst = SimpleNamespace(
        FlowReturn=SimpleNamespace(OK="ok", ERROR="error"),
        CLOCK_TIME_NONE=-1,
        SECOND=1_000_000_000,
    )
    sink = SimpleNamespace(emit=lambda _name: sample)
    state = DeepStreamSourceState(
        source_id="camera-01",
        source_uri="rtsp://example.test/live",
        display_uri="rtsp://example.test/live",
        source_type="rtsp",
        pipeline=None,
        source=None,
        sink=sink,
        bus=None,
        bus_handler_id=0,
        pipeline_handler_id=0,
        source_pad_handler_id=0,
        sink_handler_id=0,
        frame_width=1,
        frame_height=1,
        delivery_target_fps=delivery_target_fps,
        next_frame_due_monotonic=next_frame_due_monotonic,
        last_frame_monotonic=time.monotonic(),
    )
    ingestor = object.__new__(DeepStreamIngestor)
    ingestor._gst = gst
    ingestor._glib = object()
    ingestor._lock = threading.RLock()
    ingestor._states = {state.source_id: state}
    ingestor._frame_sequences = {}
    ingestor._failed_sources = set()
    ingestor._retry_after = {}
    ingestor._last_error = None
    return ingestor, state, sink


def test_deepstream_native_fps_mode_admits_each_decoded_sample() -> None:
    ingestor, state, sink = _admitting_deepstream_sample(
        delivery_target_fps=None,
    )

    assert ingestor._on_new_sample(sink, state.source_id) == "ok"
    assert state.decoded_samples == 1
    assert state.rate_limited_frames == 0
    assert state.received_frames == 1


def test_deepstream_fps_override_tolerates_nominal_timestamp_jitter() -> None:
    ingestor, state, sink = _admitting_deepstream_sample(
        delivery_target_fps=25.0,
        next_frame_due_monotonic=time.monotonic() + 0.001,
    )

    assert ingestor._on_new_sample(sink, state.source_id) == "ok"
    assert state.decoded_samples == 1
    assert state.rate_limited_frames == 0
    assert state.received_frames == 1
