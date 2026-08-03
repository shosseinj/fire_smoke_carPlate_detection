---
description: Validates GPU memory path, on-demand lifecycle, FPS, resolution, and failure isolation
mode: subagent
---

Do not modify implementation unless explicitly reassigned.

Validate with available NVIDIA/DeepStream runtime:

1. Flag disabled: no new routes/publishers affect baseline and no NVENC activity is attributable to this feature.
2. Wall closed: no wall/full branch in PLAYING state.
3. Wall opened: requested low-resolution profiles negotiate in NVMM and start shared publishers.
4. Tile clicked: fullscreen negotiates source-native width/height and source FPS.
5. Multiple clients: publisher count remains one per source/profile/mode.
6. Last client disconnect: branch stops after grace period.
7. AI metrics continue before, during, and after branch operations.
8. Inspect GStreamer DOT graphs and logs for `memory:NVMM`, NVIDIA converter/encoder, absence of appsink/CPU pixel path.
9. Use `nvidia-smi dmon` or equivalent to correlate NVENC activity with demand.

Write `.agentic/broadcast/runtime-validation.md`. Mark unavailable hardware checks BLOCKED with the exact missing prerequisite.
