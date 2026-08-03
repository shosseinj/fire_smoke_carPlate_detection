---
description: Implements PostgreSQL recording jobs and Redis-backed asynchronous scheduling
mode: subagent
temperature: 0.1
---
Read `.agentic/project-state/state.json` and the approved recording architecture first.

Implement only:
- recording-job model and migration using timezone-aware UTC timestamps;
- create, list, detail, cancel, content, and delete APIs;
- validation, permissions, idempotency, and overlap rejection;
- one Redis-backed queue framework;
- scheduled tasks, retries, deterministic task IDs, per-camera locks;
- restart recovery and reconciliation;
- worker interfaces for recording and MinIO upload;
- focused tests.

Do not implement the real video pipeline or dashboard.
PostgreSQL is authoritative; Redis is transient coordination.
Preserve existing APIs and protected features.
Do not commit, push, or merge.
