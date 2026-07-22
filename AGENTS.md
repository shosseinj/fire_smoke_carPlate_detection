# AGENTS.md — Unified Video AI Task Router

This file is the shared operating contract for Codex, OpenCode, and any sub-agents working in this repository.

## Project identity

- Name: Unified Video AI Task Router
- Version: 2.0.0
- Stack: FastAPI, PyTorch, TensorRT, ONNX Runtime, DeepStream, OpenCV
- Python: 3.10+
- Production runtime: NVIDIA GPU inside the DeepStream Docker image
- Main entrypoint: `python3 run.py`
- API documentation: `http://127.0.0.1:9999/docs`
- Dashboard: `http://127.0.0.1:9999/dashboard`
- Health and runtime evidence: `http://127.0.0.1:9999/health`

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
curl -fsS http://127.0.0.1:9999/health
curl -fsS http://127.0.0.1:9999/api/v1/models/settings
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

## Smoke-test / processor-tests convention

Every new app or feature module MUST add a smoke-test endpoint in `app/api/processor_tests.py` to
validate that the module is wired correctly end-to-end. This is mandatory regardless of whether
the module uses a real model pipeline or only a store layer.

- For store-only modules (e.g. Personnel), add a `POST /api/v1/tests/<module>/smoke` endpoint that
  exercises CRUD, search, image/asset operations, and import/export. The test must work in both
  `PROCESSOR_MODE=mock` and `PROCESSOR_MODE=real`.
- For pipeline modules (e.g. FaceRecognition, Plate, FireSmoke), add both `GET /api/v1/tests/<module>/models`
  (model file existence check) and `POST /api/v1/tests/<module>/full-pipeline` (upload-image pipeline test).
- Register the new test endpoint in the `GET /api/v1/tests/all` all-in-one status check.
- Add corresponding unit/integration tests under `tests/` that call the smoke-test endpoint and
  verify every step passes.

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
curl -fsS http://127.0.0.1:9999/health
curl -fsS http://127.0.0.1:9999/api/v1/models/settings
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

<!-- OPENCODE-REALTIME-WORKFLOW:START -->
# Real-Time Project Agent Rules

## Core interpretation rule

Treat a user request as desired behavior, not as a complete file list. Infer and implement every necessary change across configuration, schemas, endpoints, WebSockets, UI, services, workers, queues, processing pipelines, model invocation, persistence, event payloads, logging, metrics, deployment, documentation, and tests.

Never leave a feature locally implemented but unreachable. When an option changes externally controlled behavior, determine and update the correct external interface even when the user did not explicitly mention an endpoint or UI control. When no external interface is required, document the verified reason.

## Maintenance-phase behavior

This project is in maintenance and fine-tuning. Assume existing behavior should remain stable unless the user explicitly changes its contract. Treat each request as a behavioral outcome and inspect the complete connected lifecycle before editing code.

Apply the following proportionally to the risk and reach of each maintenance request. A small local fix needs a targeted audit, not a project-wide inventory:

1. Trace the affected entity or value from every input through validation, persistence, processing, output, and cleanup.
2. Build a concise impact map covering API contracts, UI/client calls, stores, database constraints, files, model or vector systems, caches, background workers, events, logs, tests, and deployment state that actually participate in the behavior.
3. Identify resources owned by the affected entity. Create, update, and delete operations must keep those resources consistent and must not leave orphaned state.
4. Search targeted names, types, routes, or helpers for plausible analogous locations. When analogues exist, classify them in the impact map as `AFFECTED`, `NOT_AFFECTED`, or `DEFERRED_WITH_REASON`.
5. Apply a shared fix to analogous locations only when they enforce the same verified invariant and the change is safe. Do not perform speculative project-wide refactors.
6. Prefer one established validator, service, cleanup helper, or contract definition over copied implementations. Remove duplication only when the replacement is behaviorally equivalent and covered by tests.
7. Verify direct behavior, connected side effects, compatibility, and at least one negative or failure path. A successful local function call is not enough when downstream state exists.
8. Preserve unrelated working behavior and user-owned worktree changes. Maintenance work is not permission to redesign a stable section.

### Entity lifecycle and ownership

Before changing CRUD behavior, identify all state owned by or referencing the entity. Deletion is complete only when the intended database records and all owned external artifacts are removed or deliberately retained according to a verified rule.

For example, deleting Personnel may require coordinated handling of:

- the personnel database row and dependent image rows;
- Qdrant face vectors and identity aliases;
- original files under `personnel_snapshots`;
- aligned files under `personnel_cropped_faces`;
- cached results or background work that can recreate stale state;
- foreign-key references, detection history, attendance history, and audit records that must be preserved, nulled, or deleted according to the existing contract;
- API, WebSocket, and UI state that must stop exposing the deleted entity.

Never infer that all related history should be deleted. Distinguish owned artifacts from historical or audit records, verify existing retention behavior, and test both cleanup and preservation.

### Contract and file-type consistency

External contracts must describe and validate the real payload, not merely accept a loosely typed value.

- Image endpoints use multipart `UploadFile` fields with binary image OpenAPI metadata and centralized runtime validation selected from MIME, extension, magic-byte, size, and decode checks according to the endpoint's established contract. Do not require a filename extension when that contract permits extensionless uploads.
- Excel import endpoints advertise and validate Excel workbook types and reject unrelated or malformed files.
- ZIP import endpoints advertise and validate ZIP archives, inspect archive safety, and reject unrelated or malformed files.
- File metadata in OpenAPI is a client hint; server-side validation remains mandatory.
- When one upload contract is corrected, audit analogous upload endpoints for the same mismatch. Change them only when the same defect is confirmed, and add focused contract tests for each affected endpoint.

Apply the same reasoning to identifiers, enums, timestamps, optional foreign keys, pagination shapes, response models, and permission checks. Do not report every database integrity failure as the same business error; map verified constraints to accurate, actionable messages.

### Connectivity and regression checks

For a change in one section, test the connected boundaries that can be affected. Examples include create/list/search consistency, model enrollment followed by deletion, file creation followed by rollback and cleanup, database mutation followed by API/WebSocket visibility, and configuration changes followed by actual container startup.

If a shared change affects multiple modules, add one focused test per distinct contract plus an integration test for the connected flow. Record pre-existing failures separately and do not weaken tests to hide them.

### Phased TODO execution

Every maintenance task must be managed through a visible phased TODO list. Create it after the initial repository and worktree inspection and before application-code edits. Keep exactly one phase `IN_PROGRESS` at a time and update the list whenever a phase finishes, fails, or materially changes.

Use these statuses:

- `TODO`: not started;
- `IN_PROGRESS`: active work;
- `DONE`: acceptance evidence collected;
- `BLOCKED`: cannot continue safely without user input or an external dependency.

Use this default phase structure, combining low-risk phases only when the task is genuinely small:

1. `Baseline and reproduction`: confirm the actual failure, current contract, worktree state, and target runtime.
2. `Impact and ownership map`: trace connected inputs, stores, external resources, outputs, analogous locations, risks, and acceptance tests.
3. `Direct implementation`: implement the smallest coherent fix for the requested behavior.
4. `Connected consistency`: repair verified downstream effects and affected analogous contracts without speculative expansion.
5. `Validation`: run static, focused, integration, regression, runtime, and deployment checks in the required order as applicable.
6. `Knowledge and handoff`: update durable project knowledge when necessary, review the final diff, and report evidence, remaining risks, rollback, and any deferred items.

A phase is not `DONE` because code was written. Mark it `DONE` only after its stated evidence or acceptance check passes. If new evidence invalidates an earlier phase, reopen that phase and update the TODO list instead of silently continuing.

The TODO list is an execution control, not a ceremonial plan. Keep it concise, reference concrete modules or contracts, and do not create redundant subtasks that repeat the same investigation or validation.

### Reuse and redundancy control

Do not solve the same invariant repeatedly in route handlers. Before adding logic, search for an existing service, validator, serializer, cleanup helper, or test fixture. Extend the authoritative implementation when safe, then migrate affected callers deliberately. Avoid parallel legacy/current implementations unless compatibility is verified and explicitly required.

When similar code cannot be unified safely, keep the implementations separate and record the reason. Three similar lines are preferable to a risky abstraction; repeated business rules that can drift require a shared owner.

### Maintenance knowledge updates

At each investigation, implementation, and validation stage, consider whether a newly verified fact is reusable project knowledge. Follow the `Knowledge maintenance` rules below and update `AGENTS.md` or the appropriate `.agentic` file only when the fact is stable and useful for future maintenance, such as an ownership relationship, integration constraint, required validation, authoritative helper, runtime command, or known failure mode.

Do not update knowledge files merely to narrate the current task. Avoid duplicate rules, consolidate overlapping guidance, and never record guesses, transient logs, credentials, private URLs, or one-off debugging details.

## Required workflow before code

1. Inspect the current repository and find the actual execution path.
2. Identify the most similar existing behavior and trace its complete lifecycle.
3. Produce a concise impact map: definitions, inputs, transformations, consumers, outputs, tests, deployment, and risks.
4. Define acceptance tests and affected evaluation sections.
5. Implement the smallest coherent end-to-end change.
6. Run focused, integration, regression, runtime, and deployment checks as applicable.
7. Update verified project knowledge and machine-readable state.
8. Report evidence, assumptions, failures, untested areas, risks, and rollback steps.

## Excel-catalog migration

Previous-project options are provided through an Excel catalog, not through a previous source repository. Migration is prompt-scoped and incremental.

Only the apps, routes, options, or behaviors selected in the current user prompt may be implemented. Required dependency and cross-layer changes for those selected items are allowed and mandatory. Every unselected catalog item must remain `PENDING`. Do not automatically continue to another group.

The Excel catalog is a partial specification. For each selected item, use evidence in this order:

1. current user prompt;
2. selected Excel fields;
3. verified current-project patterns;
4. explicitly recorded safe assumptions.

Never invent exact previous-project behavior or claim behavioral parity when previous implementation evidence was not supplied.

For each selected catalog item:

- resolve the exact app, route, option, or feature;
- extract supplied purpose, method, input/output, default, validation, permissions, runtime effect, and notes;
- inspect the current project for the correct architectural destination and similar implementations;
- classify specification coverage as `CONFIRMED_FROM_INPUT`, `IMPLEMENTED_BY_TARGET_CONVENTION`, `PARTIALLY_SPECIFIED`, `BLOCKED_NEEDS_DETAILS`, or `NOT_EVALUATED`;
- propagate the selected behavior through every required layer;
- test omitted, default, valid, boundary, invalid, permission, runtime, downstream, compatibility, and performance behavior.

When missing details can be handled safely through a strong existing target-project convention, record the assumption and proceed. When a security-sensitive or correctness-critical requirement cannot be inferred safely, mark only that item `BLOCKED_NEEDS_DETAILS`.

## Project-wide evaluation

Maintain `.agentic/evaluation/section-registry.json` for all important sections that actually exist. Each critical section must have suitable static, unit, component, integration, regression, end-to-end, runtime, performance, or deployment checks.

Use these statuses honestly:

- `PASS`
- `PASS_WITH_WARNINGS`
- `FAIL`
- `BLOCKED`
- `NOT_CONFIGURED`
- `NOT_TESTED`
- `NOT_APPLICABLE`
- `NOT_RUNTIME_VALIDATED`

Do not invent thresholds. Derive them from explicit requirements, verified current-project baselines, existing tests, production evidence supplied by the user, or explicit user direction.

## Real-time completion gate

A build, container start, or successful unit test is not enough. When the project supports a representative runtime, validate the actual flow from input stream through processing and detection/OCR to event generation, persistence, API/WebSocket output, and downstream consumer.

Measure available indicators such as input FPS, processed FPS, dropped frames, latency, queue depth, CPU, GPU, memory, errors, reconnects, and delivery time.

Do not claim `PASS` for a critical real-time feature that was not runtime validated. Use `NOT_RUNTIME_VALIDATED` and explain why.

## Knowledge maintenance

Update `AGENTS.md` or `.agentic` knowledge files only with reusable, important, verified facts: commands, architecture relationships, required services, integration rules, evaluation requirements, deployment constraints, known failure conditions, and validated performance limits.

Do not store guesses, temporary debugging notes, credentials, machine-specific secrets, or unverified assumptions as project facts. Task-specific assumptions belong in migration state and reports.

## Context efficiency

Use specialized subagents and on-demand skills. Keep the main context focused on decisions, interfaces, evidence, and unresolved risks. Do not send the entire repository to every subagent. Work one coherent feature or option group at a time.
<!-- OPENCODE-REALTIME-WORKFLOW:END -->
