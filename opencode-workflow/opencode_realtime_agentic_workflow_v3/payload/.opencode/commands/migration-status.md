---
description: Show incremental Excel-catalog migration progress without implementing new options
agent: realtime-orchestrator
---
Report incremental migration status for: $ARGUMENTS

Read `.agentic/migration/legacy-endpoint-catalog.json`, `.agentic/migration/legacy-option-map.json`, `.agentic/migration/current-scope.json`, and `.agentic/migration/migration-history.json`.

Show selected and completed items, pending items, blocked or deferred items, specification-coverage and runtime-validation status, assumptions, tests, and evidence. Suggest only the smallest sensible next groups. Do not edit application code and do not automatically select the next group.
