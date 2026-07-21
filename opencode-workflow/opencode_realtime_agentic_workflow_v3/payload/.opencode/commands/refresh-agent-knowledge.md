---
description: Update agent rules and project state only with reusable verified discoveries
agent: realtime-orchestrator
---
Review and update agent knowledge for: $ARGUMENTS

Load the `agent-knowledge-maintenance` skill. Inspect repository and runtime evidence. Update only reusable, important, verified facts in `AGENTS.md` and `.agentic` files. Avoid guesses, temporary debugging details, credentials, personal paths, and duplication.

Use `knowledge-reviewer` for independent review before finalizing meaningful rule changes.
