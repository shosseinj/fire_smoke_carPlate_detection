from __future__ import annotations

import asyncio
import json
import re
import struct
import time
import urllib.parse
import urllib.request
from collections import Counter
from typing import Any

import websockets


def _redact(source_id: str) -> str:
    return re.sub(
        r"rtsps?://[^@/\s]+@",
        "rtsp://***:***@",
        source_id,
    )


def _records(message: bytes) -> list[dict[str, Any]]:
    if len(message) < 4:
        return []
    header_size = struct.unpack("!I", message[:4])[0]
    if not 0 < header_size <= len(message) - 4:
        return []
    header = json.loads(message[4 : 4 + header_size].decode("utf-8"))
    if header.get("type") != "source_frame_batch":
        return [header]
    result = []
    offset = 4 + header_size
    for _ in range(int(header.get("count") or 0)):
        record_size = struct.unpack("!I", message[offset : offset + 4])[0]
        offset += 4
        record = message[offset : offset + record_size]
        offset += record_size
        nested_size = struct.unpack("!I", record[:4])[0]
        result.append(json.loads(record[4 : 4 + nested_size].decode("utf-8")))
    return result


def _health() -> dict[str, Any]:
    with urllib.request.urlopen("http://127.0.0.1:9999/health") as response:
        return json.load(response)


async def _next_source_frame(websocket: Any, source_id: str) -> dict[str, Any]:
    while True:
        message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
        if not isinstance(message, bytes):
            continue
        for record in _records(message):
            if record.get("source_id") == source_id:
                return record


async def main() -> None:
    wall_url = (
        "ws://127.0.0.1:9999/api/v1/video-wall/ws"
        "?wall=true&batch=true"
    )
    async with websockets.connect(wall_url, max_size=None) as wall:
        wall_counts: Counter[str] = Counter()
        started = time.monotonic()
        while time.monotonic() - started < 10.0:
            message = await asyncio.wait_for(wall.recv(), timeout=10.0)
            if isinstance(message, bytes):
                for record in _records(message):
                    wall_counts[str(record.get("source_id"))] += 1

        health = await asyncio.to_thread(_health)
        stage_sources = list(
            health.get("broadcast", {})
            .get("source_only_stage_metrics", {})
        )
        rtsp = next((item for item in stage_sources if item.startswith("rtsp")), None)
        static = next((item for item in stage_sources if not item.startswith("rtsp")), None)
        for source_id in (rtsp, static):
            if source_id is None:
                continue
            before_health = await asyncio.to_thread(_health)
            before_wall = wall_counts[source_id]
            fullscreen_url = (
                "ws://127.0.0.1:9999/api/v1/video-wall/ws"
                "?wall=true&batch=false&fullscreen_source="
                + urllib.parse.quote(source_id, safe="")
            )
            async with websockets.connect(fullscreen_url, max_size=None) as fullscreen:
                full_record = await _next_source_frame(fullscreen, source_id)
                during_until = time.monotonic() + 2.0
                while time.monotonic() < during_until:
                    message = await asyncio.wait_for(wall.recv(), timeout=5.0)
                    if isinstance(message, bytes):
                        for record in _records(message):
                            wall_counts[str(record.get("source_id"))] += 1
                during_wall = wall_counts[source_id] - before_wall
            closed_at = time.monotonic()
            wall_record = await _next_source_frame(wall, source_id)
            resume_ms = (time.monotonic() - closed_at) * 1000.0
            after_health = await asyncio.to_thread(_health)

            if source_id.startswith("rtsp"):
                redacted = _redact(source_id)
                before_state = before_health["video_ingestor"]["sources"][redacted]
                after_state = after_health["video_ingestor"]["sources"][redacted]
                pid_stable = before_state["pid"] == after_state["pid"]
                generation_stable = (
                    before_state.get("generation")
                    == after_state.get("generation")
                )
            else:
                redacted = source_id
                before_state = before_health["static_video_ingestor"]["sources"][
                    source_id
                ]
                after_state = after_health["static_video_ingestor"]["sources"][
                    source_id
                ]
                pid_stable = True
                generation_stable = (
                    before_state["source_generation"]
                    == after_state["source_generation"]
                )
            print(
                json.dumps(
                    {
                        "source": redacted,
                        "wall_resolution": [
                            wall_record["frame_width"],
                            wall_record["frame_height"],
                        ],
                        "fullscreen_resolution": [
                            full_record["frame_width"],
                            full_record["frame_height"],
                        ],
                        "wall_frames_during_fullscreen": during_wall,
                        "first_wall_after_exit_ms": round(resume_ms, 3),
                        "pid_stable": pid_stable,
                        "generation_stable": generation_stable,
                        "restart_count": after_state.get("restart_count", 0),
                    },
                    ensure_ascii=True,
                )
            )


if __name__ == "__main__":
    asyncio.run(main())
