---
description: Implements the on-demand NVMM-only wall and native fullscreen GStreamer branches after NVIDIA decode
mode: subagent
temperature: 0.1
permission:
  read: allow
  glob: allow
  grep: allow
  list: allow
  lsp: allow
  skill: allow
  edit: allow
  bash: ask
  external_directory: deny
---

Implement only the backend/GStreamer GPU live branch assigned by the orchestrator.
Before doing any work, read and follow:

`.opencode/instructions/ai-branch-protection.md`
Hard requirements:

- Attach after NVIDIA decode at confirmed NVMM output.
- Preserve the AI branch without changing its behavior.
- Wall chain outputs exactly 260x260 through GPU/NVMM elements.
- Fullscreen chain keeps native negotiated width, height, FPS, and aspect ratio; no resize caps.
- Use a new stream namespace distinct from current preview/broadcast paths, such as `live-branch/...`, but discover collisions before choosing.
- On-demand creation, reference counting, grace-period removal, heartbeat expiry, idempotent shutdown.
- One compatible branch per source/profile, reused by viewers.
- Safe pad blocking, request-pad acquisition/release, state sync, rollback, unlink, NULL, removal.
- Broadcast errors never terminate AI ingestion.
- Browser URLs must use host-accessible MediaMTX address; backend publishing uses internal service address.

Forbidden production video operations:

- appsink;
- OpenCV/NumPy/PIL pixel access;
- JPEG/PNG frames;
- software videoconvert/videoscale;
- x264enc/openh264enc/software FFmpeg encode;
- `gst_buffer_map` or host pixel copies;
- CPU overlays.

Add tests for branch construction, exact 260x260 caps, native fullscreen, branch reuse, rollback, cleanup, disabled mode, source refresh, and AI isolation. Do not report runtime playback success yourself unless proven by the browser/runtime agents.
