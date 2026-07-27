from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


def _counter_rate(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    key: str,
    sample_seconds: float,
) -> float | None:
    before_value = before.get(key)
    after_value = after.get(key)
    if not isinstance(before_value, (int, float)) or not isinstance(
        after_value, (int, float)
    ):
        return None
    if float(after_value) < float(before_value):
        return None
    return round((float(after_value) - float(before_value)) / sample_seconds, 3)


def _mapping_counter_rate(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    key: str,
    item: str,
    sample_seconds: float,
) -> float | None:
    before_mapping = before.get(key, {})
    after_mapping = after.get(key, {})
    if not isinstance(before_mapping, Mapping) or not isinstance(after_mapping, Mapping):
        return None
    return _counter_rate(
        {item: before_mapping.get(item, 0)},
        {item: after_mapping.get(item, 0)},
        item,
        sample_seconds,
    )


def _ratio_percent(value: float | None, target: float) -> float | None:
    if value is None or target <= 0:
        return None
    return round(100.0 * value / target, 1)


def _camera_report(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    sample_seconds: float,
    expected_fps: float,
    tasks: tuple[str, ...],
) -> dict[str, Any]:
    configured_target_fps = float(after.get("delivery_target_fps") or 0.0)
    native_fps = float(after.get("native_fps") or 0.0)
    comparison_target_fps = configured_target_fps or native_fps
    appsink_sample_fps = _counter_rate(
        before, after, "decoded_samples", sample_seconds
    )
    rate_limited_fps = _counter_rate(
        before, after, "rate_limited_frames", sample_seconds
    )
    admitted_fps = _counter_rate(before, after, "received_frames", sample_seconds)
    router_frame_fps = _counter_rate(
        before, after, "submitted_frames", sample_seconds
    )
    pre_router_replacement_fps = _counter_rate(
        before, after, "pre_submit_replacements", sample_seconds
    )
    if admitted_fps is None:
        admitted_fps = router_frame_fps

    causes: list[str] = []
    if (
        appsink_sample_fps == 0
        and (admitted_fps is None or admitted_fps == 0)
        and (router_frame_fps is None or router_frame_fps == 0)
    ):
        causes.append("source_unavailable")
    if (
        configured_target_fps > 0
        and configured_target_fps < expected_fps * 0.95
        and (rate_limited_fps is None or rate_limited_fps > 0)
    ):
        causes.append("configured_ingest_cap")
    if (
        comparison_target_fps > 0
        and appsink_sample_fps is not None
        and appsink_sample_fps < comparison_target_fps * 0.75
    ):
        causes.append("source_decode_starvation")
    if (
        comparison_target_fps > 0
        and admitted_fps is not None
        and admitted_fps < comparison_target_fps * 0.75
        and "source_decode_starvation" not in causes
    ):
        causes.append("ingestion_delivery_shortfall")
    if (
        admitted_fps is not None
        and router_frame_fps is not None
        and admitted_fps > 0
        and router_frame_fps < admitted_fps * 0.9
    ):
        causes.append("router_submission_shortfall")
    if pre_router_replacement_fps is not None and pre_router_replacement_fps > 0:
        causes.append("pre_router_replacement")

    observed_fps = (
        router_frame_fps if router_frame_fps is not None else admitted_fps
    )
    if causes:
        status = "limited"
    elif observed_fps is None or observed_fps == 0:
        status = "idle"
    else:
        status = "healthy"
    return {
        "status": status,
        "tasks": list(tasks),
        "source_type": after.get("source_type"),
        "fps_mode": after.get("fps_mode") or (
            "override" if configured_target_fps > 0 else "native"
        ),
        "configured_target_fps": configured_target_fps or None,
        "native_fps": native_fps or None,
        "expected_fps": expected_fps,
        "appsink_sample_fps": appsink_sample_fps,
        "rate_limited_fps": rate_limited_fps,
        "admitted_fps": admitted_fps,
        "pre_router_replacement_fps": pre_router_replacement_fps,
        "router_frame_fps": router_frame_fps,
        "target_utilization_percent": _ratio_percent(
            observed_fps,
            comparison_target_fps,
        ),
        "expected_utilization_percent": _ratio_percent(observed_fps, expected_fps),
        "last_frame_age_seconds": after.get("last_frame_age_seconds"),
        "causes": causes,
    }


def _worker_report(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    sample_seconds: float,
) -> dict[str, Any]:
    before_counters = before.get("counters", {})
    after_counters = after.get("counters", {})
    before_buffer = before.get("buffer", {})
    after_buffer = after.get("buffer", {})
    input_fps = _counter_rate(before_buffer, after_buffer, "accepted", sample_seconds)
    processed_fps = _counter_rate(before_counters, after_counters, "frames", sample_seconds)
    replacement_fps = _counter_rate(
        before_buffer, after_buffer, "stale_replaced", sample_seconds
    )
    failed_batch_fps = _counter_rate(
        before_counters, after_counters, "failed_batches", sample_seconds
    )
    pending_sources = int(after_buffer.get("pending_sources") or 0)
    causes: list[str] = []
    if replacement_fps is not None and replacement_fps > 0:
        causes.append("latest_frame_replacement")
    if (
        input_fps is not None
        and processed_fps is not None
        and input_fps > 0
        and processed_fps < input_fps * 0.9
        and pending_sources > 0
    ):
        causes.append("processor_throughput_shortfall")
    if failed_batch_fps is not None and failed_batch_fps > 0:
        causes.append("processor_failures")

    last_batch_size = int(after_counters.get("last_batch_size") or 0)
    last_batch_ms = float(after_counters.get("last_batch_ms") or 0.0)
    last_batch_capacity_fps = (
        round(last_batch_size * 1000.0 / last_batch_ms, 3)
        if last_batch_size > 0 and last_batch_ms > 0
        else None
    )
    return {
        "status": "limited" if causes else "healthy" if input_fps else "idle",
        "input_fps": input_fps,
        "processed_fps": processed_fps,
        "latest_frame_replacement_fps": replacement_fps,
        "pending_sources": pending_sources,
        "last_batch_size": last_batch_size,
        "last_batch_ms": round(last_batch_ms, 3),
        "last_batch_capacity_fps": last_batch_capacity_fps,
        "failed_batch_fps": failed_batch_fps,
        "last_error": after_counters.get("last_error"),
        "causes": causes,
    }


def _worker_source_report(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    source_id: str,
    sample_seconds: float,
) -> dict[str, Any]:
    before_counters = before.get("counters", {})
    after_counters = after.get("counters", {})
    before_buffer = before.get("buffer", {})
    after_buffer = after.get("buffer", {})
    accepted_fps = _mapping_counter_rate(
        before_buffer,
        after_buffer,
        "accepted_by_source",
        source_id,
        sample_seconds,
    )
    processed_fps = _mapping_counter_rate(
        before_counters,
        after_counters,
        "frames_by_source",
        source_id,
        sample_seconds,
    )
    replaced_fps = _mapping_counter_rate(
        before_buffer,
        after_buffer,
        "stale_replaced_by_source",
        source_id,
        sample_seconds,
    )
    failed_fps = _mapping_counter_rate(
        before_counters,
        after_counters,
        "failed_frames_by_source",
        source_id,
        sample_seconds,
    )
    causes = ["task_worker_backpressure"] if replaced_fps and replaced_fps > 0 else []
    if failed_fps and failed_fps > 0:
        causes.append("processor_failures")
    return {
        "accepted_fps": accepted_fps,
        "processed_fps": processed_fps,
        "latest_frame_replacement_fps": replaced_fps,
        "failed_fps": failed_fps,
        "causes": causes,
    }


def build_fps_report(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    sample_seconds: float,
    expected_fps: float,
    camera_tasks: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    duration = max(0.001, float(sample_seconds))
    before_video = before.get("video_ingestor", {})
    after_video = after.get("video_ingestor", {})
    before_sources = before_video.get("sources", {})
    after_sources = after_video.get("sources", {})
    cameras = {
        source_id: _camera_report(
            before_sources.get(source_id, {}),
            source_status,
            sample_seconds=duration,
            expected_fps=expected_fps,
            tasks=camera_tasks.get(source_id, ()),
        )
        for source_id, source_status in after_sources.items()
    }

    before_workers = before.get("workers", {})
    after_workers = after.get("workers", {})
    workers = {
        task: _worker_report(
            before_workers.get(task, {}),
            worker_status,
            sample_seconds=duration,
        )
        for task, worker_status in after_workers.items()
    }
    for source_id, camera in cameras.items():
        task_details = {
            task: _worker_source_report(
                before_workers.get(task, {}),
                after_workers.get(task, {}),
                source_id=source_id,
                sample_seconds=duration,
            )
            for task in camera["tasks"]
            if task in after_workers
        }
        camera["tasks_detail"] = task_details
        for detail in task_details.values():
            for cause in detail["causes"]:
                if cause not in camera["causes"]:
                    camera["causes"].append(cause)
        if camera["causes"]:
            camera["status"] = "limited"
    cause_counts = Counter(
        cause
        for section in (*cameras.values(), *workers.values())
        for cause in section["causes"]
    )
    guidance: list[str] = []
    if cause_counts["configured_ingest_cap"]:
        guidance.append(
            "A sources.fps override is below the requested FPS; raise that source row gradually and re-run this check while watching worker replacements and latency."
        )
    if cause_counts["source_decode_starvation"]:
        guidance.append(
            "One or more decoders are producing fewer frames than their configured target; inspect source FPS, connectivity, decode warnings, and frame age."
        )
    if cause_counts["source_unavailable"]:
        guidance.append(
            "One or more enabled sources delivered no frames during the sample; inspect only those camera connections while healthy sources continue independently."
        )
    if cause_counts["latest_frame_replacement"] or cause_counts[
        "processor_throughput_shortfall"
    ] or cause_counts["task_worker_backpressure"]:
        guidance.append(
            "At least one task worker is slower than incoming work; inspect batch latency, GPU utilization, model runtime, and per-task queue pressure before raising ingest FPS further."
        )
    if cause_counts["processor_failures"]:
        guidance.append("A processor failed during the sample; inspect its last_error and logs.")

    broadcast_before = before.get("broadcast", {})
    broadcast_after = after.get("broadcast", {})
    result = {
        "sample_seconds": round(duration, 3),
        "expected_fps_per_camera": expected_fps,
        "backend": after_video.get("backend"),
        "fps_control": "sources.fps",
        "native_fps_sources": sum(
            1
            for source in after_sources.values()
            if source.get("delivery_target_fps") is None
        ),
        "overridden_fps_sources": sum(
            1
            for source in after_sources.values()
            if source.get("delivery_target_fps") is not None
        ),
        "cameras": cameras,
        "workers": workers,
        "broadcast": {
            "rendered_fps": _counter_rate(
                broadcast_before,
                broadcast_after,
                "rendered_frames",
                duration,
            ),
            "full_encoded_megabits_per_second": round(
                8.0
                * (_counter_rate(
                    broadcast_before,
                    broadcast_after,
                    "full_encoded_bytes",
                    duration,
                ) or 0.0)
                / 1_000_000.0,
                3,
            ),
            "wall_encoded_megabits_per_second": round(
                8.0
                * (_counter_rate(
                    broadcast_before,
                    broadcast_after,
                    "wall_encoded_bytes",
                    duration,
                ) or 0.0)
                / 1_000_000.0,
                3,
            ),
            "encode_failures_per_second": _counter_rate(
                broadcast_before,
                broadcast_after,
                "encode_failures",
                duration,
            ),
        },
        "primary_causes": [
            {"code": code, "affected_sections": count}
            for code, count in cause_counts.most_common()
        ],
        "guidance": guidance,
    }
    result["overall"] = (
        "limited"
        if cause_counts
        else "healthy"
        if any(camera["status"] == "healthy" for camera in cameras.values())
        else "idle"
    )
    return result
