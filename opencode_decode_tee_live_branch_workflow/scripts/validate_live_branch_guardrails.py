#!/usr/bin/env python3
"""Static guardrail for forbidden CPU-oriented operations in the new live branch.

The workflow agent should extend TARGET_HINTS when it discovers the production files.
This script intentionally fails on suspicious tokens in files whose path/name indicates
live-branch or broadcast-gpu ownership.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGET_HINTS = ("live_branch", "live-branch", "broadcast_gpu", "broadcast-gpu")
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
    lowered_path = str(path.relative_to(ROOT)).lower()
    if not any(hint in lowered_path for hint in TARGET_HINTS):
        continue
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    lowered = text.lower()
    for token in FORBIDDEN:
        if token.lower() in lowered:
            violations.append(f"{path.relative_to(ROOT)}: forbidden token {token!r}")

if violations:
    print("Live-branch GPU guardrail FAILED:")
    print("\n".join(f"- {item}" for item in violations))
    sys.exit(1)

print("Live-branch GPU guardrail passed.")
