---
description: Validates the complete scheduled recording flow and protected regressions
mode: subagent
temperature: 0.1
permission:
  edit: deny
---
Read `.agentic/project-state/state.json` and inspect the complete unmerged change.

Do not edit production code.

Validate:
- dashboard scheduling and timezone conversion;
- PostgreSQL persistence;
- Redis scheduling, locks, retry, cancellation, and restart recovery;
- recording start, stop, finalization, and local spool behavior;
- MinIO upload verification and authenticated content access;
- AI, NVMM tee, MediaMTX, WHEP, reconnect, and camera settings regressions;
- Redis/MinIO outage behavior and multiple-camera independence.

Return exact test evidence, failures, unsafe scope expansion, unresolved blockers,
rollback instructions, and whether the feature is safe to keep.
