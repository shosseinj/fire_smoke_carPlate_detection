---
description: Independently audit an implemented decode-tee GPU live branch and print exact tested access URLs
agent: live-branch-orchestrator
subtask: false
---
Audit the existing decode-tee live branch without broad redesign.

Run the AI guardian, architecture inspection, GPU guardrails, real Playwright wall/fullscreen playback test, runtime lifecycle validation, and final reviewer.

Fix only defects demonstrated by evidence and add regressions. Require exact 260x260 wall caps, native fullscreen, a new URL namespace, and `See live branch` dashboard access.

Update `.agentic/live-branch/final-report.md`. Print exact tested frontend, API, and WHEP URLs. Do not push or merge.
