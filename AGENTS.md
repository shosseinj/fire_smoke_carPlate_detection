# AGENTS.md — Unified Video AI Task Router

This file is the shared operating contract for Codex, OpenCode, and any sub-agents working in this repository.

## Project identity

- Name: Unified Video AI Task Router
- Version: 2.0.0
- Stack: FastAPI, PyTorch, TensorRT, ONNX Runtime, DeepStream, OpenCV
- Python: 3.10+
- Production runtime: NVIDIA GPU inside the DeepStream Docker image
- Main entrypoint: `python3 run.py`
- API documentation: `http://127.0.0.1:8000/docs`
- Dashboard: `http://127.0.0.1:8000/dashboard`
- Health and runtime evidence: `http://127.0.0.1:8000/health`

## Product priorities

Use this priority order unless the user explicitly overrides it for a task:

1. Correctness, safety, and stable service operation.
2. Bounded end-to-end latency and real-time behavior.
3. Sustainable throughput and fair camera scheduling.
4. Processing every historical frame.

Real-time behavior is more important than accumulating stale work. Under overload, prefer the newest useful frame per source, bounded queues, fair scheduling, and visible drop/backpressure metrics. Never allow an unbounded queue. If a task explicitly requires lossless processing, apply backpressure and explain that latency may grow and RTSP sources may still lose frames upstream when processing capacity is insufficient.

For performance-sensitive changes, report evidence rather than assumptions:

- Input FPS and processed FPS per task.
- Batch size and batch latency.
- Queue depth, capacity, frame age, blocked submissions, replacements, and drops.
- Active model artifact and runtime format: TensorRT, ONNX, or PT.
- GPU memory and utilization when available.
- `/health` before and after the change.

The normal detector input is fixed at `3×640×640`. TensorRT detector engines should use a dynamic batch dimension only, normally batch 1–8. Use `--dynamic-batch-only`; do not make height and width dynamic unless the task explicitly requires it.

## Architecture

- `app/main.py`: FastAPI application and lifespan.
- `app/runtime.py`: component construction, startup, shutdown, and model selection logs.
- `app/core/router.py`: routes frames to task-specific workers.
- `app/core/worker.py`: micro-batched task execution.
- `app/core/latest_buffer.py`: task queue policy and queue telemetry.
- `app/core/deepstream_ingestor.py`: production GPU decode and frame delivery.
- `app/core/video_ingestor.py`: OpenCV development fallback.
- `app/core/model_management.py`: persistent model catalog, selection, conversion jobs, and fallback resolution.
- `app/processors/fire_smoke.py`: fire/smoke detection, tracking, severity, and incidents.
- `app/processors/plate.py`: vehicle detection, plate detection, and Iranian plate OCR.
- `app/processors/face_recognition.py`: human detection, tracking, face detection, quality checks, ArcFace, and vector search.

The three independent inference workers are:

- `fire_smoke`
- `plate_recognition`
- `face_recognition`

Model fallback priority is `.engine → .onnx → .pt` when fallbacks are enabled. Persistent selections in the model database override configuration defaults. Confirm the active artifact through `/api/v1/models/settings` and runtime logs instead of trusting filenames alone.

## Multi-agent workflow

Use multiple agents when the task contains independent investigation, implementation, testing, or review work. One primary agent must act as coordinator.

### Coordinator responsibilities

The coordinator must:

1. Read this file and inspect the current worktree before delegation.
2. Restate the requested outcome and identify the real-time acceptance criteria.
3. Split work into bounded, non-overlapping responsibilities.
4. Assign explicit file or module ownership to each implementation agent.
5. Tell every agent that other agents share the worktree and that user changes must not be reverted.
6. Keep at most one agent editing a given file at a time.
7. Integrate the results, review the combined diff, and resolve conflicts.
8. Own the final testing phase and retry policy.
9. Report measured results, remaining risks, and any manual follow-up.

### Recommended roles

- Explorer: read-only code tracing, runtime diagnosis, and evidence collection.
- Backend worker: APIs, runtime wiring, databases, and model management.
- Video/GPU worker: DeepStream, OpenCV, TensorRT, batching, and performance-sensitive paths.
- Test worker: independent regression tests, live smoke tests, and failure reproduction.
- Reviewer: correctness, concurrency, shutdown behavior, bounded memory, and real-time impact.

Combine roles when the change is small. If the platform cannot create sub-agents, the primary agent must execute the same roles sequentially.

### Delegation rules

- Do not delegate vague tasks such as “fix the project.”
- Give each agent a concrete question, outcome, and owned files.
- Agents must not reset, clean, overwrite, or revert unrelated worktree changes.
- Agents must inspect existing code before editing and preserve established architecture.
- Agents must communicate discovered constraints before making cross-module changes.
- Parallel agents must not edit the same files unless the coordinator explicitly serializes their work.
- The coordinator must verify delegated claims in the actual repository or runtime.
- Use read-only explorers in parallel where possible; serialize overlapping implementation work.

## Development workflow

For every requested change:

1. Inspect `git status`, relevant source, configuration, tests, and live state when applicable.
2. Establish a baseline for behavior or reproduce the failure.
3. Define a minimal implementation plan and real-time acceptance criteria.
4. Delegate independent work with explicit ownership when multi-agent work is useful.
5. Implement the smallest coherent change.
6. Add or update tests for the changed behavior.
7. Review the combined diff for scope, secrets, concurrency hazards, and performance regressions.
8. Run the mandatory testing phase.
9. If tests fail, follow the three-retry repair policy.
10. Finish only after tests pass, or record the unresolved failure for manual work.

Do not install or upgrade packages unless the user explicitly authorizes it. Use the repository’s existing host environment, container, models, and tools. Never solve a version mismatch by silently changing TensorRT, CUDA, PyTorch, or DeepStream packages.

## Mandatory testing phase

Every development task ends with testing. Documentation-only changes require at least formatting, link/path, and consistency checks. Code changes require tests proportional to risk.

Run these stages in order when applicable:

### Stage 1: static validation

```bash
python -m py_compile <changed-python-files>
git diff --check
docker compose config --quiet
```

Only run commands relevant to the changed files. Do not let unrelated pre-existing worktree whitespace hide validation of the files owned by the task.

### Stage 2: focused tests

Run the smallest tests that directly cover the change, for example:

```bash
python -m pytest tests/test_latest_buffer.py tests/test_router.py -q
python -m pytest tests/test_model_management.py -q
python -m pytest tests/test_face_recognition.py -q
```

### Stage 3: regression suite

```bash
python -m pytest -q
```

Compare failures with the baseline. A pre-existing failure must be reported clearly and must not be claimed as caused or fixed without evidence.

### Stage 4: runtime validation

Runtime, ingestion, model, API, WebSocket, concurrency, or performance changes require live validation in the actual target container when available:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/api/v1/models/settings
docker logs --tail 200 merged-video-ai-router
```

For TensorRT changes, validate deserialization, execution-context creation, input profiles, and at least one real inference at minimum and maximum supported batch sizes. Engines must be built on the device and TensorRT runtime that will execute them.

For real-time pipeline changes, sample runtime counters long enough to show frame progression, queue behavior, failure counts, batch latency, and frame age. Backend startup alone is not sufficient proof.

## Three-retry repair policy

The initial testing phase is attempt zero. If any required test fails, Codex or OpenCode must diagnose and try to repair the issue automatically. A maximum of three repair retries is allowed.

For each retry:

1. Capture the exact failing command and the relevant error.
2. Identify a concrete root-cause hypothesis.
3. Apply a scoped fix to production code or valid test expectations.
4. Rerun the failed test first.
5. Rerun the focused regression tests after the failed test passes.
6. Rerun broader validation if the fix affects shared behavior.

Retry definitions:

- Retry 1: first diagnosis, fix, and retest.
- Retry 2: revised diagnosis, fix, and retest.
- Retry 3: final diagnosis, fix, and retest.

Do not repeat the same command three times without changing the diagnosis or implementation. Do not weaken, delete, skip, or mark tests as expected failures merely to obtain a green result. Update a test only when the intended contract genuinely changed, and state that contract change.

If all required tests pass during any retry, continue to the remaining testing stages and finish normally.

## Unresolved failure log

If the third repair retry still fails, stop automatic repair and append an entry to the repository-root text file:

```text
agent_test_failures.log
```

The file is append-only. Never erase or rewrite earlier entries. Create it only when an unresolved failure occurs. Do not include secrets, camera credentials, API keys, tokens, private URLs, or full sensitive payloads.

Each entry must use this format:

```text
================================================================================
Timestamp UTC: <ISO-8601 timestamp>
Agent: <Codex or OpenCode and model if known>
Task: <short requested outcome>
Status: FAILED_AFTER_3_RETRIES
Changed files: <comma-separated paths>
Baseline: <baseline result or not available>
Failed command: <exact command>
Final error: <concise error excerpt>
Retry 1: <hypothesis, change, result>
Retry 2: <hypothesis, change, result>
Retry 3: <hypothesis, change, result>
Current diagnosis: <best-supported root cause>
Manual next step: <specific action for the user>
================================================================================
```

After logging, the agent must:

- Tell the user that automatic repair stopped after three retries.
- Link or name `agent_test_failures.log`.
- Summarize what works and what remains broken.
- Leave diagnostic changes only when they are safe and useful.
- Never claim the development task is complete.

## Definition of done

A development task is complete only when:

- The requested behavior is implemented in the correct repository and runtime boundary.
- Relevant tests were added or updated.
- Static checks and focused tests pass.
- The regression suite passes, or pre-existing failures are proven and reported.
- Required live validation passes in the target environment.
- Real-time impact is measured for hot-path changes.
- No secrets or private camera URLs were exposed.
- The combined diff was reviewed and unrelated user changes were preserved.
- No unresolved required test remains. If one remains after three retries, it is logged and the task is handed back as incomplete.

## Coding conventions

- Add `from __future__ import annotations` to every Python file.
- Type-hint every function signature.
- Use `snake_case` for functions, variables, and modules.
- Use `PascalCase` for classes.
- Prefix private methods with `_`.
- Use `slots=True` for dataclasses and `frozen=True` for settings and value objects.
- Keep imports grouped as standard library, third-party, and local.
- Avoid comments in code unless they are necessary to explain a non-obvious invariant.
- Prefer minimum coherent diffs over unrelated refactors.
- Preserve Persian text, RTL behavior, and UTF-8 when editing user-facing Persian surfaces.

API routers use this pattern:

```python
router = APIRouter(prefix="/api/v1/<resource>", tags=["<tag-group>"])


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime
```

Use `Annotated` for FastAPI `UploadFile`, `Form`, `File`, and `Depends` parameters. Use Pydantic models for request and response contracts.

## Runtime and TensorRT constraints

- TensorRT engines are version-, OS-, GPU-, and platform-specific.
- `trt107` artifacts must be built and executed with the Python TensorRT 10.7 runtime.
- Do not assume system `trtexec` and Python `tensorrt` have the same version; verify both.
- Build engines inside the same container and on the same device class used for inference.
- Detector profiles should normally be `(-1, 3, 640, 640)` with batch 1–8 and fixed spatial dimensions.
- ArcFace profiles should normally be `(-1, 3, 112, 112)` with batch 1–64 and fixed spatial dimensions.
- Use `scripts/build_all_engines.py` to rebuild all device-specific engines.
- Persistent selections under `/api/v1/models/settings` can override `.env` and `app/config.py` defaults.
- Do not install `tensorrt-cu12` into the DeepStream image to repair an engine mismatch.

## Common commands

Development and tests:

```bash
python3 run.py
PROCESSOR_MODE=mock python3 run.py
python -m pytest -q
python3 scripts/smoke_test.py
PROCESSOR_MODE=mock python3 scripts/mock_load_test.py
```

Production container:

```bash
docker compose build
docker compose up -d --no-build
docker compose logs -f video-ai-router
```

Build all TensorRT engines inside the target container:

```bash
python3 scripts/build_all_engines.py --batch 8 --workspace 4 --device 0
```

Build a detector with dynamic batch and fixed spatial dimensions:

```bash
python3 -m app.model_export_worker \
  --source <input.pt> \
  --target <output.engine> \
  --format engine \
  --imgsz 640 \
  --batch 8 \
  --workspace 4 \
  --device 0 \
  --half \
  --dynamic-batch-only
```

Inspect live state:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/api/v1/models/settings
docker logs --tail 200 merged-video-ai-router
```

## Security and worktree safety

- Never expose RTSP credentials, API keys, model-service tokens, or private URLs.
- Redact secrets from logs, test output, examples, summaries, and failure entries.
- Inspect `git status` before editing.
- Treat existing modifications and untracked files as user-owned unless proven otherwise.
- Never use destructive Git or filesystem commands without explicit authorization.
- Do not rewrite unrelated code to make tests pass.
- Keep model weights, TensorRT engines, runtime databases, generated media, and local secrets out of commits unless explicitly requested.
