from __future__ import annotations

import argparse
import asyncio
import json
import re
import struct
import time
from collections import Counter

import websockets


async def probe(url: str, duration: float) -> None:
    frames = 0
    by_source: Counter[str] = Counter()
    started = time.monotonic()
    async with websockets.connect(url, max_size=None) as websocket:
        while time.monotonic() - started < duration:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            if isinstance(message, bytes):
                frames += 1
                if len(message) >= 4:
                    header_size = struct.unpack("!I", message[:4])[0]
                    if 0 < header_size <= len(message) - 4:
                        try:
                            header = json.loads(
                                message[4 : 4 + header_size].decode("utf-8")
                            )
                            records = [header]
                            if header.get("type") == "source_frame_batch":
                                records = []
                                offset = 4 + header_size
                                for _ in range(int(header.get("count") or 0)):
                                    if offset + 4 > len(message):
                                        break
                                    record_size = struct.unpack(
                                        "!I", message[offset : offset + 4]
                                    )[0]
                                    offset += 4
                                    record = message[offset : offset + record_size]
                                    offset += record_size
                                    if len(record) < 4:
                                        continue
                                    nested_size = struct.unpack("!I", record[:4])[0]
                                    records.append(
                                        json.loads(
                                            record[4 : 4 + nested_size].decode("utf-8")
                                        )
                                    )
                            for record_header in records:
                                source_id = str(
                                    record_header.get("source_id") or "<unknown>"
                                )
                                source_id = re.sub(
                                    r"rtsps?://[^@/\s]+@",
                                    "rtsp://***:***@",
                                    source_id,
                                )
                                by_source[source_id] += 1
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            pass
    elapsed = time.monotonic() - started
    print(f"frames={frames} duration_seconds={elapsed:.3f}")
    for source_id, count in sorted(by_source.items()):
        print(f"source={source_id} frames={count} fps={count / elapsed:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()
    asyncio.run(probe(args.url, args.duration))


if __name__ == "__main__":
    main()
