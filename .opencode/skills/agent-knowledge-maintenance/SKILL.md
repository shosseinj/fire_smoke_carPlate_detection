---
name: agent-knowledge-maintenance
description: Persist only reusable verified repository knowledge in AGENTS.md and .agentic state while excluding guesses, secrets, and temporary debugging details
compatibility: opencode
metadata:
  domain: agent-operations
  workflow: knowledge-maintenance
---
# Agent knowledge maintenance

## Store

Store verified build, lint, test, evaluation, deployment, rollback, and runtime commands; architecture relationships not obvious from filenames; required services; interface contracts; integration rules; recurring failure conditions; important conventions; and validated performance thresholds.

## Do not store

Do not store guesses, temporary task details, raw logs, transient errors, credentials, tokens, secrets, private URLs, personal machine paths, redundant rules, or facts supported only by an unverified assumption.

## Location

- `AGENTS.md`: concise durable project rules needed in most sessions.
- `.agentic/architecture.md`: verified architecture.
- `.agentic/integration-map.md`: section and interface relationships.
- `.agentic/feature-catalog.md`: implemented capabilities and evidence.
- `.agentic/state.json`: machine-readable current state and commands.
- `.agentic/decisions/`: durable architectural decisions.

Review meaningful updates independently before finalizing them.
