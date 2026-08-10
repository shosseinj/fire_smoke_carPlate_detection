# Live Video Save Development Rules

This repository is a real-time GPU application.
The current development phase is ONLY for asynchronously saving live frames/video using Redis and MinIO.

## Hard rules

1. Work only on the `live` branch.
2. Never checkout, merge, rebase, reset, cherry-pick, force-update, or modify `ai-branch`.
3. Do not change unrelated API endpoints, AI/inference code, stream processing, GPU logic, or existing behavior unless the user explicitly approves it.
4. Diagnose the existing broken Redis/MinIO implementation BEFORE rewriting it.
5. Make one small change at a time.
6. After every code change run:
   - `git diff --stat`
   - `git diff`
   - `python tools/check_changes.py`
7. If `check_changes.py` reports an unexpected file, STOP and report it. Do not silently repair or revert it.
8. Never hide or discard pre-existing user changes.
9. Real-time behavior is critical. Frame/video persistence must not block the live/GPU processing path.
10. Prefer queue/enqueue work on the live path and perform MinIO I/O in a separate worker/background process/task appropriate to the existing architecture.
11. Do not introduce a new framework if the existing project already has a suitable Redis/job/worker mechanism.
12. Before changing architecture, show the suspected failure point and evidence from logs/code flow.

## Development order

### Step 1 — Observe
Map the current path:
`frame produced -> enqueue/publish -> Redis -> worker/consumer -> encode/assemble -> MinIO upload`
Do not edit yet.

### Step 2 — Find the first broken boundary
Check each boundary independently and identify the FIRST point where expected data stops flowing.
Examples: producer never enqueues, Redis key/stream mismatch, worker not running, serialization mismatch, temp video never closes, MinIO upload fails, bucket/object path error.

### Step 3 — Minimal fix
Change only the smallest component necessary to fix that boundary.

### Step 4 — Verify
Verify both:
- new save path works
- live/GPU path and unrelated endpoints remain unchanged

### Step 5 — Report
Always report:
- root cause found
- files changed
- protected/unexpected files changed (must be none)
- test performed
- result
- next step

## Git protection

The phase is controlled by `.workflow/phase.json` created by `tools/start_phase.py`.
Only paths explicitly listed there are allowed to change during this phase.
All other changed paths must be reported.
