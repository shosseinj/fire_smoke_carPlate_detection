param(
    [string]$ProjectRoot = "."
)

$ErrorActionPreference = "Stop"

$agentsDir = Join-Path $ProjectRoot ".opencode\agents"
$commandsDir = Join-Path $ProjectRoot ".opencode\commands"

New-Item -ItemType Directory -Force -Path $agentsDir | Out-Null
New-Item -ItemType Directory -Force -Path $commandsDir | Out-Null

$files = @{
    (Join-Path $agentsDir "video-recording-architect.md") = @'
---
description: Designs the smallest safe architecture for scheduled asynchronous video recording
mode: subagent
temperature: 0.1
permission:
  edit: deny
---
Read `.agentic/project-state/state.json` first.

Inspect the current FastAPI, DeepStream, NVMM tee, MediaMTX/WHEP, camera, dashboard,
PostgreSQL, Docker, worker, Redis, and media-storage implementation.

Do not edit production code.

Design the smallest staged architecture that lets a user select a camera and local
start/end time in the live dashboard, records asynchronously, uploads finalized
video to MinIO, and preserves AI and live playback.

Prefer reusing the existing stream. Explicitly state whether another RTSP session
is required.

Return architecture, affected components, stages, risks, tests, rollback, and blockers.
'@

    (Join-Path $agentsDir "minio-storage-agent.md") = @'
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
'@

    (Join-Path $agentsDir "redis-recording-agent.md") = @'
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
'@

    (Join-Path $agentsDir "live-recording-agent.md") = @'
---
description: Implements scheduled recording execution without disrupting AI or live playback
mode: subagent
temperature: 0.1
---
Read `.agentic/project-state/state.json`, the approved architecture, and recording-job contracts first.

Implement only actual recording execution:
- start and stop at scheduled UTC timestamps;
- record the selected camera;
- support cancellation;
- write finalized files to persistent local spool;
- enqueue asynchronous MinIO upload;
- delete local files only after verified upload;
- retry and recover idempotently.

Priority:
1. safely reuse the existing DeepStream pipeline;
2. otherwise reuse the existing MediaMTX output;
3. open another RTSP session only if unavoidable and document why.

Never block DeepStream callbacks.
Failures must not stop AI, decoded frames, MediaMTX, or WHEP.
Add focused and regression tests.
Do not commit, push, or merge.
'@

    (Join-Path $agentsDir "recording-dashboard-agent.md") = @'
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
'@

    (Join-Path $agentsDir "video-recording-validator.md") = @'
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
'@

    (Join-Path $commandsDir "video-step-1.md") = @'
---
description: Design scheduled asynchronous video recording
agent: video-recording-architect
subtask: true
---
Design the scheduled asynchronous video-recording feature for this repository.
Do not edit production code.

Additional instructions: $ARGUMENTS
'@

    (Join-Path $commandsDir "video-step-2.md") = @'
---
description: Implement MinIO video-storage foundation
agent: minio-storage-agent
subtask: true
---
Implement Step 2, the MinIO storage foundation, using the approved Step 1 architecture.

Additional instructions: $ARGUMENTS
'@

    (Join-Path $commandsDir "video-step-3.md") = @'
---
description: Implement Redis recording-job infrastructure
agent: redis-recording-agent
subtask: true
---
Implement Step 3, PostgreSQL recording jobs and Redis asynchronous scheduling,
using the approved contracts from earlier steps.

Additional instructions: $ARGUMENTS
'@

    (Join-Path $commandsDir "video-step-4.md") = @'
---
description: Implement live scheduled recording execution
agent: live-recording-agent
subtask: true
---
Implement Step 4, actual scheduled recording and asynchronous MinIO upload handoff,
using the approved architecture and existing recording-job interfaces.

Additional instructions: $ARGUMENTS
'@

    (Join-Path $commandsDir "video-step-5.md") = @'
---
description: Add recording controls to the live dashboard
agent: recording-dashboard-agent
subtask: true
---
Implement Step 5, the scheduled-recording dashboard controls, using the existing APIs.

Additional instructions: $ARGUMENTS
'@

    (Join-Path $commandsDir "video-step-6.md") = @'
---
description: Validate scheduled recording end to end
agent: video-recording-validator
subtask: true
---
Validate Step 6, the full scheduled asynchronous video-recording feature and all
protected live/AI regressions. Do not edit production code.

Additional instructions: $ARGUMENTS
'@
}

foreach ($entry in $files.GetEnumerator()) {
    Set-Content -LiteralPath $entry.Key -Value $entry.Value -Encoding UTF8
    Write-Host "Created $($entry.Key)"
}

Write-Host ""
Write-Host "Done. Restart OpenCode, then run:"
Write-Host "  /video-step-1"
Write-Host "  /video-step-2"
Write-Host "  /video-step-3"
Write-Host "  /video-step-4"
Write-Host "  /video-step-5"
Write-Host "  /video-step-6"
