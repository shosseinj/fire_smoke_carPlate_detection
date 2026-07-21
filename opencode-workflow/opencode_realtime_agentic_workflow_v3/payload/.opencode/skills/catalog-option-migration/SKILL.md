---
name: catalog-option-migration
description: Implement previous-project options incrementally from only the Excel rows, routes, options, and prompt details selected in the current request
compatibility: opencode
metadata:
  domain: software-migration
  workflow: prompt-scoped-excel-catalog
---
# Incremental Excel-catalog option migration

## Scope is controlled by the current prompt

Migration is `PROMPT_SCOPED_INCREMENTAL_CATALOG_ONLY`.

Only apps, routes, options, or behaviors explicitly selected by the current prompt may be implemented. Required cross-layer changes supporting those selected items are in scope. All other catalog items remain `PENDING`.

Do not automatically implement an entire application because one route was named. Do not continue to the next group after the selected group is complete.

## Catalog versus complete behavior

`.agentic/migration/legacy-endpoint-catalog.json` is generated from Excel. It may contain only an app and URL. It is not previous source code and may not define methods, schemas, permissions, defaults, validation, business rules, UI, storage, or runtime effects.

Use evidence in this order:

1. current prompt details;
2. Excel fields for selected rows;
3. verified patterns in the current project;
4. clearly recorded safe assumptions.

Never claim exact previous behavior or behavioral parity without implementation evidence supplied by the user.

## Scope state

Before code:

1. Resolve the prompt into exact selected apps, routes, options, or behaviors.
2. Record it in `.agentic/migration/current-scope.json`.
3. Add a run entry to `.agentic/migration/migration-history.json`.
4. Mark only selected catalog items as `SELECTED`.
5. Keep every unselected item `PENDING`.
6. Record prompt enrichment and missing details.

## Specification fields

For each selected item record, when known:

- app, route, method, option name, and source Excel row;
- purpose and user-visible effect;
- request and response schemas;
- permissions and authentication behavior;
- type, values, default, boundaries, and validation;
- API, WebSocket, UI, CLI, environment, or config exposure;
- runtime consumer and downstream effects;
- persistence, event, log, and metric effects;
- deployment requirements;
- tests and fixtures;
- evidence source for every important decision;
- safe assumptions and unresolved details.

Store mappings in `.agentic/migration/legacy-option-map.json`.

## Specification coverage

Use:

- `CONFIRMED_FROM_INPUT`
- `IMPLEMENTED_BY_TARGET_CONVENTION`
- `PARTIALLY_SPECIFIED`
- `BLOCKED_NEEDS_DETAILS`
- `NOT_EVALUATED`

If a missing detail is security-sensitive, destructive, irreversible, or central to correctness and cannot be safely derived from an established current-project pattern, block only that item rather than guessing.

## Complete propagation for selected items

The prompt controls which behavior is selected, not which files can change. For selected behavior, update every necessary config, schema, endpoint, WebSocket, UI, service, worker, queue, pipeline, model call, storage, event, deployment, documentation, and test layer.

## Evaluation

Evaluate implementation against the supplied specification and target-project conventions for omitted/default, custom, disabled, boundary, invalid, permissions, runtime output, API/event payload, persistence, error handling, connectivity, regression, deployment, and performance.

## Completion

A selected item is not complete until it is specified sufficiently, mapped, propagated, consumed at runtime, tested, documented, and included in regression evaluation. Completing one scope does not authorize work on pending items.
