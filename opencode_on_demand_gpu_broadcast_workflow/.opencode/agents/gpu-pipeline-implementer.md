---
description: Implements isolated dynamic NVIDIA GPU branch modules and tests
mode: subagent
---

Own only:

- `app/broadcast_gpu/gst_*.py`
- `app/broadcast_gpu/branch_*.py`
- `tests/broadcast_gpu/test_gst_*.py`
- `tests/broadcast_gpu/test_branch_*.py`

Do not edit current ingestion files. Request the orchestrator to apply the minimal hook after your interface is tested.

Required pipeline properties:

Wall:
`queue -> nvvideoconvert -> video/x-raw(memory:NVMM),width=<profile>,height=<profile> -> nvv4l2h264enc -> h264parse -> rtspclientsink`

Fullscreen:
`queue -> nvvideoconvert only when required for encoder compatibility -> video/x-raw(memory:NVMM) -> nvv4l2h264enc -> h264parse -> rtspclientsink`

Do not set fullscreen width, height, or framerate caps. Never use pixel mapping, appsink, OpenCV, NumPy, JPEG, CPU videoconvert, or videoscale.

Use request pads, pad blocking, state synchronization, EOS/flush handling, and idempotent teardown. Write failing tests first.
