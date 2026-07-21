<!-- OPENCODE-REALTIME-WORKFLOW:START -->
# Real-Time Project Agent Rules

## Core interpretation rule

Treat a user request as desired behavior, not as a complete file list. Infer and implement every necessary change across configuration, schemas, endpoints, WebSockets, UI, services, workers, queues, processing pipelines, model invocation, persistence, event payloads, logging, metrics, deployment, documentation, and tests.

Never leave a feature locally implemented but unreachable. When an option changes externally controlled behavior, determine and update the correct external interface even when the user did not explicitly mention an endpoint or UI control. When no external interface is required, document the verified reason.

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
