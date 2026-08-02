---
description: Implement and fully validate the decode-tee GPU live branch using the multi-agent workflow
agent: live-branch-orchestrator
subtask: false
---
Implement the requested decode-tee GPU live branch end to end.

The existing AI branch must remain unchanged.

Create a new on-demand live branch after NVIDIA decode/NVMM tee with:

- a low-resolution video wall at exactly 260x260;
- native-resolution fullscreen when a tile is clicked;
- GPU/NVMM-only conversion and encoding;
- a different route and stream namespace from the existing preview;
- a dashboard button labeled `See live branch`;
- reference-counted sessions, heartbeat, release, rollback, and shutdown cleanup;
- MediaMTX WebRTC/WHEP playback;
- real Playwright dashboard tests that prove video is actually playing.

Use all specialized agents. Establish AI baseline first, implement, test in a real browser, validate runtime GPU lifecycle, run independent final review, update `.agentic/live-branch/final-report.md`, and create logical local commits.

Do not push or merge. Do not report success from generated URLs or mocked playback.
