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
            "target_fps": 5.0,
            "preview_fps": 25.0,
            "sources": {
                "camera-01": {
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
        preserve_source_resolution=False,
        delivery_target_fps=5.0,
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
