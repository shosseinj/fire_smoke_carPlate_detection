from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
from pathlib import Path

from app.core.rtsp_process_supervisor import _rtsp_child_main
from app.core.source_registry import RTSP, SourceRegistry, canonical_source_type


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    registry = SourceRegistry(os.environ["DATABASE_URL"])
    try:
        records = [
            record
            for record in registry.list()
            if record.enabled
            and canonical_source_type(record.source_uri, record.source_type) == RTSP
        ]
        records.sort(key=lambda record: record.source_uri)
        if args.index < 0 or args.index >= len(records):
            raise SystemExit(
                f"RTSP source index out of range: {args.index}/{len(records)}"
            )
        record = records[args.index]
    finally:
        registry.close()

    context = mp.get_context("spawn")
    prefix = (
        "vai_rtsp_gdb_"
        + hashlib.sha256(record.source_uri.encode()).hexdigest()[:16]
    )
    shared_transport = {
        "generation": 1,
        "shared_memory_names": (f"{prefix}_0", f"{prefix}_1"),
        "buffer_locks": (context.Lock(), context.Lock()),
        "active_buffer_index": context.Value("i", -1),
        "write_sequence": context.Value("Q", 0),
        "read_sequence": context.Value("Q", 0),
        "buffer_sequences": (
            context.Value("Q", 0),
            context.Value("Q", 0),
        ),
        "frames_written": context.Value("Q", 0),
        "metadata_dropped": context.Value("Q", 0),
        "overwritten_frames": context.Value("Q", 0),
        "write_copy_ns": context.Value("Q", 0),
    }
    fullscreen_transport = {
        **shared_transport,
        "shared_memory_names": (
            f"{prefix}_fullscreen_0",
            f"{prefix}_fullscreen_1",
        ),
        "buffer_locks": (context.Lock(), context.Lock()),
        "active_buffer_index": context.Value("i", -1),
        "write_sequence": context.Value("Q", 0),
        "read_sequence": context.Value("Q", 0),
        "buffer_sequences": (
            context.Value("Q", 0),
            context.Value("Q", 0),
        ),
        "frames_written": context.Value("Q", 0),
        "metadata_dropped": context.Value("Q", 0),
        "overwritten_frames": context.Value("Q", 0),
        "write_copy_ns": context.Value("Q", 0),
    }
    _rtsp_child_main(
        record.to_dict(),
        {
            "project_root": Path("/workspace"),
            "video_only_mode": True,
            "gpu_resize_enabled": True,
            "loop": True,
            "rtsp_transport": "tcp",
            "rtsp_latency_ms": 500,
            "rtsp_reconnect_seconds": 3.0,
            "rtsp_stall_timeout_seconds": 30,
            "skip_taskless_sources": True,
            "rtsp_enabled": True,
            "raw_enabled": False,
        },
        context.Queue(maxsize=8),
        context.Queue(maxsize=32),
        context.Event(),
        shared_transport,
        fullscreen_transport,
        context.Value("b", False),
    )


if __name__ == "__main__":
    main()
