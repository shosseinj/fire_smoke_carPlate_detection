from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
FEATURE = ROOT / "app" / "broadcast_gpu"

FORBIDDEN = {
    "appsink": re.compile(r"\bappsink\b", re.I),
    "cpu videoconvert": re.compile(r"(?<!nv)videoconvert", re.I),
    "videoscale": re.compile(r"\bvideoscale\b", re.I),
    "opencv": re.compile(r"\b(cv2|opencv)\b", re.I),
    "numpy": re.compile(r"\b(numpy|np\.)", re.I),
    "jpeg": re.compile(r"\b(jpegenc|imencode|\.jpg)\b", re.I),
    "pixel buffer map": re.compile(r"\b(buffer\.map|Gst\.Buffer\.map)\b", re.I),
}

if not FEATURE.exists():
    print("BLOCKED: app/broadcast_gpu does not exist yet; run after implementation.")
    sys.exit(2)

violations: list[str] = []
for path in FEATURE.rglob("*.py"):
    text = path.read_text(encoding="utf-8", errors="replace")
    for label, pattern in FORBIDDEN.items():
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            violations.append(f"{path.relative_to(ROOT)}:{line}: forbidden {label}: {match.group(0)}")

config_files = [ROOT / "app" / "config.py", ROOT / ".env.example"]
joined = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in config_files if p.exists())
if "GPU_BROADCAST_ENABLED" not in joined:
    violations.append("Missing GPU_BROADCAST_ENABLED configuration")
if re.search(r"GPU_BROADCAST_ENABLED\s*=\s*(true|1|yes)", joined, re.I):
    violations.append("GPU_BROADCAST_ENABLED must default to false")

if violations:
    print("Guardrail violations:")
    for item in violations:
        print(f" - {item}")
    sys.exit(1)

print("Guardrails passed: no forbidden CPU pixel path found and feature flag is present.")
