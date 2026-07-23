---
description: Test every project app's CRUD/API surface sequentially with validation, response, permission, persistence, cleanup, and coverage reporting.
agent: build
---

Follow the `crud-testing` skill for this request.

First perform a mandatory test-freshness audit before implementing or running
new tests. Compare the current routes, OpenAPI schemas, stores, migrations,
response contracts, permissions, and lifecycle behavior with
`tests/test_crud_api.py` and module-specific tests. Classify every app as
`UPDATED`, `STALE`, `MISSING`, `PARTIAL`, or `BLOCKED`.

If any tests are stale, missing, or partial, stop and tell the user exactly
which test files and app behaviors need updating, then ask whether to update
the tests. Do not modify or implement tests until the user explicitly approves.
If the user declines, report the gaps and stop. If approved, update only the
necessary test files, rerun collection and the freshness audit, and then
continue. Never change production code just to satisfy stale tests.

After the freshness gate passes, test the project's CRUD/API applications one
by one. Use only a disposable PostgreSQL database and mock processors; never
expose credentials, private URLs, tokens, or sensitive payloads.

For every applicable app, validate create/list/get/update/delete or lifecycle
operations, success status codes, response JSON shape and fields, required and
optional validation, malformed and boundary inputs, duplicate/not-found and
foreign-key errors, authentication and role permissions, persistence,
side-effects, downstream visibility, and cleanup. Classify non-CRUD routes
instead of silently skipping them.

Run the focused app-by-app tests, then `tests/test_crud_api.py`, relevant
module-specific tests, and the full regression suite when appropriate. Review
`.agentic/reports/crud-analysis-latest.json` and `.agentic/reports/crud-analysis-latest.md`.
Report exact failures, skipped or blocked coverage, pre-existing failures,
runtime limitations, and recommended fixes. Preserve unrelated worktree
changes and do not modify application code unless the user explicitly asks
for implementation fixes.

Additional scope or test arguments:

$ARGUMENTS
