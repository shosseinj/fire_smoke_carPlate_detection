# On-Demand GPU Broadcast Implementation Plan

> **For agentic workers:** Implement task-by-task with fresh subagents and independent review gates. Use checkbox state in `.agentic/broadcast/task-board.md`.

**Goal:** Add isolated, reference-counted, on-demand NVMM/NVENC video-wall and native fullscreen publishers without changing existing AI behavior.

**Architecture:** New modules under `app/broadcast_gpu/` own demand, sessions, dynamic GStreamer bins, MediaMTX paths, and API contracts. Existing code receives only feature-flagged additive lifecycle/route registration and a minimal decoded-NVMM attachment hook.

**Tech Stack:** Python 3.10+, FastAPI, GStreamer, NVIDIA DeepStream, NVMM, NVENC, MediaMTX WHEP, existing frontend JavaScript.

## Global constraints

- `GPU_BROADCAST_ENABLED=false` by default.
- No CPU pixel mapping or Python frame access in the broadcast path.
- Wall uses GPU resize to one bounded profile.
- Fullscreen preserves native size, aspect ratio, timestamps, and FPS.
- No viewer means no active converter/encoder/publisher bin.
- Existing AI/JPEG/recording paths remain behaviorally unchanged.

---

### Task 1: Baseline and integration map

- [ ] Create feature worktree/branch.
- [ ] Record clean/dirty status and baseline commit.
- [ ] Run targeted existing tests: video ingestor, DeepStream ingestor, media preview, broadcast, dashboard surface, and static-video lifecycle.
- [ ] Map exact decoded-NVMM attachment point and source lifecycle.
- [ ] Freeze files allowed for minimal modification.
- [ ] Commit only inventory/state documents.

### Task 2: Demand registry and session lifecycle

**Create:** `app/broadcast_gpu/demand.py`, `sessions.py`, `schemas.py`, and tests.

- [ ] Write failing tests for 0→1 start, sharing, 1→0 delayed stop, heartbeat expiry, duplicate release, and grace cancellation.
- [ ] Implement immutable demand keys and bounded registries.
- [ ] Implement async session creation/release/heartbeat/expiry.
- [ ] Run targeted tests.
- [ ] Commit `feat: add on-demand broadcast session registry`.

### Task 3: Dynamic GPU branch builder

**Create:** `app/broadcast_gpu/gst_factory.py`, `branches.py`, and tests.

- [ ] Write failing topology tests using fake Gst elements.
- [ ] Implement wall bin with NVMM profile caps and NVIDIA encoder.
- [ ] Implement fullscreen bin with no size/FPS caps.
- [ ] Implement request-pad attach, state sync, safe detach, and branch-local bus/error handling.
- [ ] Add static forbidden-token tests.
- [ ] Run targeted tests.
- [ ] Commit `feat: add isolated NVMM broadcast branches`.

### Task 4: Branch manager and source adapter

**Create:** `app/broadcast_gpu/manager.py`, `source_adapter.py`, and tests.

- [ ] Write failing tests for idempotent start/stop and source-not-ready behavior.
- [ ] Implement manager keyed by camera/mode/profile.
- [ ] Implement adapter protocol so current ingestor exposes only attachment-point operations.
- [ ] Add minimal feature-flagged hook to current source construction without altering existing branch properties.
- [ ] Prove disabled topology equivalence in tests.
- [ ] Commit `feat: connect optional broadcast attachment point`.

### Task 5: FastAPI session endpoints

**Create:** `app/broadcast_gpu/api.py`; minimally register router/lifespan.

- [ ] Write failing API tests for wall create, fullscreen create, heartbeat, release, invalid cameras/profiles, and disabled flag.
- [ ] Implement response payloads containing session IDs and WHEP URLs.
- [ ] Add expiry task to feature-owned lifespan.
- [ ] Verify no endpoint starts a branch until demand is accepted.
- [ ] Commit `feat: expose on-demand broadcast sessions`.

### Task 6: Frontend wall and fullscreen

**Create:** feature-owned JS/CSS where possible; minimally wire dashboard.

- [ ] Write contract tests for feature disabled, wall activation, visible tiles, fullscreen click, close, heartbeat, and reconnect.
- [ ] Add WHEP video elements without removing JPEG fallback.
- [ ] Pause/release selected wall demand during fullscreen when configured.
- [ ] Restore tile demand after fullscreen close.
- [ ] Commit `feat: add on-demand WebRTC wall and fullscreen`.

### Task 7: Isolation and regression gate

- [ ] Run all targeted baseline tests with feature disabled.
- [ ] Run new tests with feature enabled and fake branch manager.
- [ ] Run `scripts/validate_gpu_broadcast_guardrails.py`.
- [ ] Review git diff against allowed modification list.
- [ ] Independent isolation reviewer approves.
- [ ] Commit fixes only after review evidence.

### Task 8: NVIDIA runtime validation

- [ ] Start MediaMTX and application on NVIDIA runtime.
- [ ] Verify zero branch/NVENC demand with wall closed.
- [ ] Open 24-camera wall and verify bounded low-resolution profiles.
- [ ] Open one fullscreen stream and verify source-native resolution/FPS.
- [ ] Confirm no additional Python/CPU frame mapping.
- [ ] Disconnect clients and verify delayed teardown.
- [ ] Confirm AI throughput and health remain stable.
- [ ] Save DOT graphs, logs, counters, and commands in final report.

### Task 9: Final handoff

- [ ] Update documentation and `.env.example` with disabled defaults.
- [ ] Record rollback: set `GPU_BROADCAST_ENABLED=false` and restart.
- [ ] List commits and unmerged branch.
- [ ] Do not merge without user approval.
