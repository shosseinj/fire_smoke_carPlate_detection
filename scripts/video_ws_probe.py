from __future__ import annotations

import argparse
import asyncio
import time

import websockets


async def probe(url: str, duration: float) -> None:
    frames = 0
    started = time.monotonic()
    async with websockets.connect(url, max_size=None) as websocket:
        while time.monotonic() - started < duration:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            if isinstance(message, bytes):
                frames += 1
    print(f"frames={frames} duration_seconds={time.monotonic() - started:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()
    asyncio.run(probe(args.url, args.duration))


if __name__ == "__main__":
    main()
