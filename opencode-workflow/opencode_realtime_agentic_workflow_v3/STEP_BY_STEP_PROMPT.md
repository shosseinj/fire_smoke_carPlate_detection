# Prompt for one incremental Excel-catalog migration step

Use the `realtime-orchestrator` agent.

Implement only the following selected previous-project catalog scope in this run:

- App or subsystem: `<APP NAME OR NONE>`
- Catalog routes: `<ONE OR MORE URLS OR NONE>`
- Option or feature: `<DESCRIBE THE OPTION/FEATURE>`
- Required behavior: `<DESCRIBE INPUTS, OUTPUTS, DEFAULTS, PERMISSIONS, ERRORS, AND RUNTIME EFFECTS YOU KNOW>`
- Explicitly out of scope: `<OPTIONAL EXCLUSIONS>`

Use the Excel catalog and this prompt as the specification. Inspect only the current repository for architecture and implementation patterns. There is no previous source repository.

Rules:

1. Resolve this prompt into exact selected work items using `.agentic/migration/legacy-endpoint-catalog.json`.
2. Do not implement any unmentioned catalog item. Keep it `PENDING`.
3. Merge evidence in this order:
   - current prompt details;
   - selected Excel row fields;
   - verified current-project conventions;
   - explicitly recorded safe assumptions.
4. Never claim exact previous behavior or behavioral parity without evidence supplied in the prompt or Excel.
5. You may inspect neighboring current-project code for context, but implement it only when it is a proven dependency of the selected behavior.
6. Before application code, show:
   - resolved scope;
   - confirmed specification;
   - missing or inferred details;
   - target-project destination;
   - end-to-end impact map;
   - required endpoint/schema/UI/config/service/pipeline/storage/event/deployment changes;
   - affected evaluation sections;
   - tests and acceptance criteria;
   - compatibility risks and rollback plan.
7. When behavior is underspecified but can be implemented safely using an established target-project pattern, record the assumption and continue. When a security-sensitive or correctness-critical detail cannot be safely inferred, mark only that item `BLOCKED_NEEDS_DETAILS`.
8. Apply the selected behavior through every necessary layer. I do not need to mention every required file or endpoint layer.
9. Preserve current project behavior when the selected option is omitted unless this prompt explicitly requires otherwise.
10. Test default, omitted, valid, invalid, boundary, permission, propagation, downstream output, integration, regression, deployment, and representative real-time behavior where applicable.
11. Evaluate specification coverage, not unsupported legacy parity. Use statuses such as `CONFIRMED_FROM_INPUT`, `IMPLEMENTED_BY_TARGET_CONVENTION`, `PARTIALLY_SPECIFIED`, `BLOCKED_NEEDS_DETAILS`, and `NOT_EVALUATED`.
12. Update current scope, catalog progress, option map, migration history, evaluation state, and reusable verified agent knowledge.
13. Stop after this selected scope. Do not start another pending item automatically.

Begin with scope resolution and the pre-code impact map, then implement only the selected scope.
