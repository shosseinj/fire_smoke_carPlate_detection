---
description: Implements private MinIO storage for finalized recording files
mode: subagent
temperature: 0.1
---
Read `.agentic/project-state/state.json` and the approved recording architecture first.

Implement only the MinIO storage foundation and direct dependencies:
- configuration and secrets;
- private bucket initialization;
- upload, stat verification, delete, and presigned download service;
- deterministic object keys;
- Docker service, persistent volume, health check;
- focused tests.

Do not implement recording, Redis scheduling, or dashboard controls.
Do not alter DeepStream, NVMM tee, MediaMTX, or WHEP.
Do not expose credentials, commit, push, or merge.

Report files changed, tests, configuration, risks, and blockers.
