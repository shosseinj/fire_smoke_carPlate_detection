---
description: Inspect this repository and configure its architecture, integration, Excel-catalog migration, and section-evaluation state
agent: realtime-orchestrator
---
Bootstrap the installed agentic workflow for this repository.

Do not implement a catalog option yet.

Inspect the current repository and update only the workflow state needed for future work:

1. Detect the actual language, frameworks, services, entry points, test tools, build commands, deployment method, real-time input paths, processing pipeline, endpoints, WebSockets, UI, workers, queues, storage, and observability.
2. Update `.agentic/architecture.md` and `.agentic/integration-map.md` with verified facts and exact paths.
3. Populate `.agentic/evaluation/section-registry.json` with every important section that actually exists and real commands that can evaluate it.
4. Populate regression relationships in `.agentic/evaluation/regression-matrix.json`.
5. Validate the Excel-derived `.agentic/migration/legacy-endpoint-catalog.json`, but keep all items `PENDING`.
6. Record baseline commands and unresolved setup requirements in `.agentic/state.json`.
7. Run `python scripts/validate_agentic_workflow.py` and any safe baseline checks.
8. Produce a concise bootstrap report with configured sections, catalog summary, missing fixtures, blocked checks, and next actions.

There is no previous source repository. Do not invent previous behavior, thresholds, or runtime validation.
