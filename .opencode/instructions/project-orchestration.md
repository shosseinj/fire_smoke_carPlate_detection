# Project orchestration instruction

For implementation and Excel-catalog migration work, use the `realtime-orchestrator` workflow.

Use `todowrite` to manage work in visible phases. Keep one phase in progress at a time, update it as work advances, and require evidence before completion. Follow the phased TODO structure defined in `AGENTS.md`.

Before application-code edits, provide:

- current-repository facts relevant to the request;
- confirmed prompt and Excel specification;
- missing or inferred details;
- an end-to-end impact map;
- affected evaluation sections;
- acceptance tests;
- compatibility and rollback considerations.

A user's request is behavioral intent. Infer endpoint, schema, UI, configuration, runtime-consumer, downstream-output, deployment, documentation, and test changes when the current architecture requires them.

Use OpenCode skills on demand:

- `change-propagation`
- `catalog-option-migration`
- `project-evaluation`
- `real-time-validation`
- `agent-knowledge-maintenance`

Use specialized read-only subagents for exploration and independent review. Keep implementation ownership in the primary orchestrator so responsibilities and evidence remain coherent.

Never convert unknown requirements into invented facts or thresholds. Mark assumptions, missing runtime validation, and blocked evidence explicitly.

## Incremental Excel-catalog scope

When a user supplies previous-project options step by step, resolve only the current prompt scope. Use `.agentic/migration/legacy-endpoint-catalog.json` to match app names, source rows, routes, and option names. There is no previous source repository.

Use prompt details first, Excel fields second, verified current-project conventions third, and explicit safe assumptions last. Keep all unmentioned items `PENDING`. Do not continue to another catalog group without a later user prompt. Do not claim exact previous behavior or parity unless the user supplied evidence.
