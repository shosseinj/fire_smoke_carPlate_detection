from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
from urllib.parse import urlsplit

from sqlalchemy import create_engine, text


def _redact(value: str) -> str:
    return re.sub(r"rtsps?://[^@/\s]+@", "rtsp://***:***@", value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--duration", type=int, default=15)
    parser.add_argument(
        "--codec", choices=("discover", "h264", "h265"), default="discover"
    )
    parser.add_argument("--redacted-only", action="store_true")
    args = parser.parse_args()

    engine = create_engine(os.environ["DATABASE_URL"])
    with engine.connect() as connection:
        urls = connection.execute(
            text(
                "SELECT source_uri FROM sources "
                "WHERE source_uri LIKE 'rtsp%' ORDER BY source_uri"
            )
        ).scalars().all()
    if args.index < 0 or args.index >= len(urls):
        raise SystemExit(f"source index out of range: {args.index}/{len(urls)}")
    source_uri = str(urls[args.index])
    if args.redacted_only:
        print(_redact(source_uri))
        return 0
    parsed = urlsplit(source_uri)
    try:
        with socket.create_connection(
            (parsed.hostname or "", parsed.port or 554), timeout=3.0
        ):
            tcp_status = "connected"
    except OSError as exc:
        tcp_status = f"failed:{type(exc).__name__}"
    command = [
        "timeout",
        str(args.duration),
        "gst-launch-1.0",
        "-v",
        "rtspsrc",
        f"location={source_uri}",
        "protocols=tcp",
        "latency=300",
        "!",
    ]
    if args.codec == "h264":
        command.extend(["rtph264depay", "!", "h264parse", "!"])
    elif args.codec == "h265":
        command.extend(["rtph265depay", "!", "h265parse", "!"])
    command.extend(["fakesink", "sync=false"])
    environment = dict(os.environ)
    environment["GST_DEBUG"] = "3"
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
    )
    print(f"source_index={args.index} source=redacted codec={args.codec}")
    print(
        f"endpoint={parsed.hostname}:{parsed.port or 554} "
        f"tcp_preflight={tcp_status}"
    )
    print(f"exit_code={completed.returncode}")
    print("\n".join(_redact(completed.stdout).splitlines()[-160:]))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
