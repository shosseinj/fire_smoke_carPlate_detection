---
description: Adds scheduled recording controls to the existing live dashboard
mode: subagent
temperature: 0.1
---
Read `.agentic/project-state/state.json` and existing recording APIs first.

Modify only the live dashboard and unavoidable API-client integration.

Add:
- searchable camera-name selector that submits camera_id;
- local start and end date/time controls;
- calculated duration and validation;
- schedule action;
- job table and status;
- cancel, permitted retry, and ready play/download actions.

Convert local times using the configured IANA timezone and send timezone-aware ISO-8601 values.
Reuse the existing polling, WebSocket, or SSE mechanism.
Do not redesign unrelated UI or change DeepStream, Redis, or MinIO internals.
Add focused frontend tests.
Do not commit, push, or merge.
