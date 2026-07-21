# Step-by-Step Excel-Catalog Migration

The migration mode is `PROMPT_SCOPED_INCREMENTAL_CATALOG_ONLY`.

## Core rule

OpenCode implements only the app, endpoint, option, feature, or behavior included in the current prompt. It still makes every required cross-layer change for that selected behavior. All unmentioned catalog items remain `PENDING`.

The included Excel catalog is a partial specification, not previous source code. A route list alone does not define complete behavior.

## Evidence order

1. Current prompt details
2. Selected Excel row fields
3. Verified current-project implementation patterns
4. Explicitly recorded safe assumptions

The workflow must not claim old-project behavioral parity when no old implementation was supplied.

## Recommended sequence

1. Install the workflow with the bundled Excel or `--options-excel`.
2. Run `/bootstrap-agentic` in the new repository.
3. Give one small selected scope using `/migrate-selected-options`.
4. Include behavior details you know, such as inputs, outputs, defaults, permissions, errors, and runtime effects.
5. Review the impact map, confirmed facts, and assumptions before code.
6. Let OpenCode finish implementation and evaluation of only that scope.
7. Run `/migration-status` to review progress.
8. Supply the next selected scope in a new prompt.
9. Periodically run `/evaluate-project` and `/validate-deployment`.

## Examples

```text
/migrate-selected-options App: Authentication. Routes: /api/v1/auth/login and /api/v1/auth/me. Login accepts username and password, returns access and refresh tokens, and /me returns the authenticated user profile.
```

```text
/migrate-selected-options App: Cameras. Route: /api/v1/cameras/{camera_id}/effective-settings. Return project defaults merged with camera-specific overrides. Do not implement unrelated camera CRUD routes.
```

```text
/migrate-selected-options Option: smoke confidence threshold. It is a float from 0 to 1, defaults to 0.5, can be configured globally and per camera, and must affect runtime smoke detections.
```

## Required state

- `.agentic/migration/source/previous-project-options.xlsx`: installed Excel source
- `.agentic/migration/legacy-endpoint-catalog.json`: normalized catalog and progress
- `.agentic/migration/current-scope.json`: only the current prompt scope
- `.agentic/migration/legacy-option-map.json`: specification and target mappings
- `.agentic/migration/migration-history.json`: completed and attempted prompt scopes

## Underspecified items

When the prompt and Excel do not fully define behavior:

- proceed only when an established current-project pattern provides a safe implementation;
- record the assumption and evidence;
- use `IMPLEMENTED_BY_TARGET_CONVENTION` or `PARTIALLY_SPECIFIED`;
- use `BLOCKED_NEEDS_DETAILS` for security-sensitive, destructive, irreversible, or correctness-critical unknowns.

## Completion boundary

After finishing the selected scope, stop. Do not automatically select or implement the next pending catalog item.
