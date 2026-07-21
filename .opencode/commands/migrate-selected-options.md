---
description: Implement only the Excel-catalog apps, routes, options, or behaviors named in the current prompt, with end-to-end propagation and evaluation
agent: realtime-orchestrator
---
Implement only this selected previous-option catalog scope: $ARGUMENTS

Load `catalog-option-migration`, `change-propagation`, `project-evaluation`, and `real-time-validation`.

First ask the `scope-controller` to resolve the exact prompt scope against `.agentic/migration/legacy-endpoint-catalog.json`. Write the resolved scope to `.agentic/migration/current-scope.json` and create a migration run entry. All unselected catalog items must remain `PENDING`.

For selected items only:

1. Merge current prompt details with the selected Excel row fields.
2. Inspect the current project for the correct architecture and similar existing patterns.
3. Separate confirmed facts, target-convention inferences, safe assumptions, and missing critical details.
4. Show an impact map, acceptance tests, compatibility behavior, risks, and rollback method before application-code changes.
5. Implement every required cross-layer change needed to make the selected behavior reachable and functional.
6. Do not implement neighboring catalog items merely because they are in the same app. Shared refactors are allowed only when necessary for selected behavior and must preserve other behavior.
7. Run focused, integration, regression, deployment, and representative real-time checks as applicable.
8. Evaluate against the supplied specification and target-project conventions. Do not claim unsupported previous-project parity.
9. Update catalog, option-map, current-scope, history, evaluation, and verified knowledge state.

Finish the selected scope before proposing another group. Report any item as `BLOCKED_NEEDS_DETAILS`, `NOT_TESTED`, or `NOT_RUNTIME_VALIDATED` when evidence is incomplete.
