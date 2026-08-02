#!/usr/bin/env python3
"""Static guardrail for forbidden CPU-oriented operations in the new live branch.

The workflow agent should extend TARGET_HINTS when it discovers the production files.
This script intentionally fails on suspicious tokens in files whose path/name indicates
live-branch or broadcast-gpu ownership.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
TARGET_FILES = {Path("app/core/live_branch.py")}
FORBIDDEN = (
    "appsink",
    "cv2.",
    "opencv",
    "numpy",
    "np.",
    "pillow",
    "from PIL",
    "jpeg",
    "jpg",
    "videoconvert",
    "videoscale",
    "x264enc",
    "openh264enc",
    "gst_buffer_map",
    ".tobytes(",
    ".cpu(",
)
SKIP_PARTS = {".git", ".venv", "venv", "node_modules", "__pycache__", "tests"}

violations: list[str] = []
for path in ROOT.rglob("*"):
    if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
        continue
    relative_path = path.relative_to(ROOT)
    if relative_path not in TARGET_FILES:
        continue
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    lowered = text.lower()
    for token in FORBIDDEN:
        if token in {"videoconvert", "videoscale"}:
            # nvvideoconvert is the required GPU element; reject only the
            # software element name as a token, not its NVIDIA variant.
            import re
            found = re.search(rf"(?<!nv){re.escape(token)}\b", lowered)
        else:
            found = token.lower() in lowered
        if found:
            violations.append(f"{path.relative_to(ROOT)}: forbidden token {token!r}")

if violations:
    print("Live-branch GPU guardrail FAILED:")
    print("\n".join(f"- {item}" for item in violations))
    sys.exit(1)

print("Live-branch GPU guardrail passed.")
