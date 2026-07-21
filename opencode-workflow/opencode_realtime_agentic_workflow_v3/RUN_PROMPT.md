# OpenCode initial setup prompt for Excel-catalog migration

Use the `realtime-orchestrator` agent and installed skills.

The current working directory is the new real-time project. Previous-project options are supplied only through `.agentic/migration/source/previous-project-options.xlsx` and the normalized `.agentic/migration/legacy-endpoint-catalog.json`. There is no previous source repository.

The Excel catalog is a partial specification. It may list only an application and route. Do not invent exact old behavior or claim behavioral parity.

This run is setup and inspection only. Do not implement all previous options.

1. Inspect the current repository architecture, startup paths, services, endpoints, configuration sources, UI, workers, queues, persistence, deployment, tests, and real-time processing paths.
2. Update `.agentic/architecture.md` and `.agentic/integration-map.md` using verified target-repository evidence.
3. Populate `.agentic/evaluation/section-registry.json` with important sections that actually exist and real evaluation commands where supported.
4. Validate the Excel catalog structure and keep every catalog item `PENDING`.
5. Explain how later prompts will enrich selected catalog rows with behavior details.
6. Establish an evidence policy:
   - current prompt details first;
   - Excel fields second;
   - verified target-project patterns third;
   - clearly recorded safe assumptions last.
7. For missing details, use `NEEDS_DETAILS_OR_TARGET_INFERENCE`; do not describe guessed behavior as previous-project behavior.
8. Prepare migration state files but create no application-code changes for catalog options.
9. Run baseline project checks and record honest statuses, including `NOT_RUNTIME_VALIDATED` where appropriate.
10. Update reusable verified project knowledge only.

Finish with:

- repository inspection summary;
- project section registry summary;
- baseline evaluation result;
- Excel catalog summary;
- suggested small migration groups only.

Do not select or implement a migration group until I provide it in a later prompt using `/migrate-selected-options` or `STEP_BY_STEP_PROMPT.md`.
