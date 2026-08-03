---
description: Implement an isolated on-demand NVMM/NVENC video wall and fullscreen branch with multi-agent review
---

You are the primary orchestrator. Read and obey:

- `AGENTS.md`
- `.opencode/instructions/on-demand-gpu-broadcast.md`
- `docs/on-demand-gpu-broadcast-design.md`
- `docs/on-demand-gpu-broadcast-implementation-plan.md`

Use the specialized agents in `.opencode/agents/`.

## Required workflow

1. Create or switch to a dedicated branch/worktree named `feature/on-demand-gpu-broadcast`. Never implement on the user's main branch.
2. Capture baseline evidence:
   - `git status --short`
   - relevant current pipeline files and route registration
   - baseline targeted tests for video ingestion, preview, broadcast, dashboard, and static-video lifecycle
   - current `/health` response when runtime is available
3. Initialize `.agentic/broadcast/state.json` and `.agentic/broadcast/task-board.md` from the templates in this package.
4. Dispatch in parallel, read-only:
   - `pipeline-analyst`
   - `broadcast-architect`
   - `broadcast-test-designer`
5. Reconcile their reports into one exact integration map. Stop if they disagree about the decoded-NVMM attachment point.
6. Implement in dependency order with disjoint ownership:
   - `session-api-implementer`
   - `gpu-pipeline-implementer`
   - `frontend-stream-implementer`
7. After each task, run its targeted tests and commit independently.
8. Run `isolation-reviewer`. Fix every critical/high finding before continuing.
9. Run `runtime-validator`. If GPU/DeepStream is unavailable, mark hardware checks BLOCKED rather than fabricating evidence.
10. Run `& "C:/Users/jafari.h/Desktop/ai_project/.venv/Scripts/python.exe" scripts/validate_gpu_broadcast_guardrails.py` and all targeted regression tests with the same Python interpreter.
11. Produce `.agentic/broadcast/final-report.md` containing:
    - exact files created and minimally modified;
    - baseline versus final test results;
    - proof the feature is disabled by default;
    - proof no-viewer demand leaves no active encoder branch;
    - wall resolution and fullscreen native-resolution evidence;
    - source/native FPS evidence;
    - NVMM/NVENC and no-CPU-map evidence;
    - known limitations and rollback command.
12. Do not merge. Present the commits and evidence for user review.

## Completion gate

You may say COMPLETE only when:

- baseline behavior passes with `GPU_BROADCAST_ENABLED=false`;
- no forbidden CPU frame operations exist in `app/broadcast_gpu/`;
- demand/ref-count/heartbeat lifecycle tests pass;
- wall and fullscreen frontend transitions pass;
- isolation review has no critical/high findings;
- runtime evidence is passed or explicitly marked BLOCKED due to unavailable NVIDIA runtime.
