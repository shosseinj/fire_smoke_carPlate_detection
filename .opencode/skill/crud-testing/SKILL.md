---
name: crud-testing
description: Use when testing the project's CRUD/API surface app by app, including validation, permissions, response contracts, error handling, persistence, cleanup, and coverage reports.
compatibility: Windows PowerShell 5.1, pytest, Docker Compose, FastAPI TestClient
---

# Comprehensive CRUD Testing

Test the project's API applications one at a time. This skill is for finding
missing routes, incorrect validation, response-shape regressions, authorization
mistakes, persistence problems, incomplete cleanup, and untested error paths.

## Scope

Start with the authoritative CRUD suite and shared helpers:

- `tests/test_crud_api.py`
- `app/testsupport.py`
- `pytest.ini`

Use route registration, Pydantic schemas, stores/services, migrations, and
module-specific tests to expand or verify coverage. Do not assume that an
endpoint is CRUD merely because it is under `/api/v1`; classify it as CRUD,
read-only/operational, pipeline, WebSocket, or not applicable.

The current app-by-app CRUD/API order is:

- `auth`
- `cameras`
- `sources`
- `personnel`
- `locations` (`buildings`, `sections`, `rooms`)
- `shifts`
- `holidays`
- `requests`
- `broadcast`
- `diagnostics`
- `faces`
- `fire_smoke`
- `general_settings`
- `humans`
- `models`
- `plate_logs`
- `plate_settings`
- `results`
- `attendance`
- `frames`
- `health`
- Any additional registered app or legacy compatibility route discovered during inventory

If the route inventory differs from this list, report the difference and test
the discovered application instead of silently omitting it.

## Safety

- Run against a disposable PostgreSQL database only.
- Never use production `DATABASE_URL`, camera credentials, private RTSP URLs,
  model-service tokens, or real personnel data.
- Use `PROCESSOR_MODE=mock` and disable video ingestion for CRUD tests unless a
  specific pipeline smoke test requires otherwise.
- Use unique test identifiers and clean up created rows, files, vectors, and
  temporary artifacts where the application owns them.
- Preserve existing user changes. Do not reset, clean, overwrite, or delete
  unrelated worktree files.
- Redact credentials, tokens, private URLs, and full sensitive payloads from
  reports and final output.

## Required Workflow

### 1. Establish the baseline

Inspect the worktree and runtime configuration before testing. Confirm the
disposable database target without printing its password. Check route
registration and collect the CRUD tests:

```powershell
python -m pytest --collect-only -q tests/test_crud_api.py
```

If the disposable PostgreSQL service is not running and Docker is available,
use the repository's test profile:

```powershell
docker compose --profile test up -d postgres-test
```

Set `TEST_DATABASE_URL` only in the process environment. Do not write it to a
tracked file or report it with credentials.

### 2. Audit test freshness before implementation

Before writing, extending, or repairing any test, compare the current tests
with the current application contract. This is a mandatory approval gate, not
an optional report.

For every registered app, compare:

- OpenAPI and route methods/paths against collected test methods and paths
- Request schemas, required fields, optional fields, enums, ranges, and defaults
  against test payloads and validation assertions
- Success status codes and response schemas against test expectations
- Authentication and role requirements against permission tests
- Store models, migrations, side effects, and cleanup behavior against lifecycle tests
- Current module-specific tests and `tests/test_crud_api.py` against the current implementation

Classify each app as `UPDATED`, `STALE`, `MISSING`, `PARTIAL`, or `BLOCKED`.
Treat a test as stale when it asserts removed routes, old fields/status codes,
obsolete response shapes, or behavior no longer supported by the current
implementation. Treat an app as partial when the happy path exists but
validation, permissions, errors, persistence, cleanup, or downstream checks
are absent.

Write a concise freshness report containing the app, affected test files,
specific stale or missing cases, current contract evidence, and the proposed
test changes. Do not implement or modify tests while any app is `STALE`,
`MISSING`, or `PARTIAL`.

Stop and ask the user explicitly whether the test files should be updated.
The question must include the affected files and scope. If the user declines,
do not add or modify tests; report the coverage gaps and stop. If the user
approves, update only the necessary test files, preserve the application's
current contract, and rerun collection and the freshness audit before running
the CRUD suite. Never change production code solely to make stale tests pass.

If all applicable apps are `UPDATED`, tell the user that the freshness gate
passed and continue to the inventory and execution stages.

### 3. Build the app inventory

For each registered router, record:

- Module/app name and route prefix
- Methods and paths
- Request body, query, path, and multipart fields
- Required and optional fields, enum/range/format constraints, and defaults
- Expected success status codes and response schema/shape
- Authentication and role requirements
- Database, file, vector, cache, event, or worker side effects
- Cleanup and retention behavior
- Existing focused tests and missing cases

Use the OpenAPI schema and implementation as the contract. Do not invent
business rules where the project has no verified convention; mark them as
`UNKNOWN` or `NOT_APPLICABLE` in the report.

### 4. Test one app at a time

Run each app's test class or focused test group separately before running the
full suite. Keep the order deterministic and capture the exact command and
result. At minimum, validate the following where the app supports it:

- Create with a valid minimal payload
- Create with a complete payload and optional fields
- List response status, content type, top-level shape, item shape, and pagination/filter behavior
- Get an existing resource and verify identity and persisted values
- Update every supported update method and verify changed fields
- Delete or disable the resource and verify the documented lifecycle behavior
- Read-after-write consistency through the API and the authoritative store
- Invalid JSON, missing required fields, wrong types, malformed identifiers, invalid enum values, and boundary values
- Duplicate/conflict behavior and accurate error status/detail
- Missing-resource behavior, including 404/410 semantics where applicable
- Unauthenticated, viewer/operator, and admin authorization behavior
- Foreign-key and dependency failures
- File/image/ZIP upload validation and rollback when applicable
- External artifact cleanup or deliberate retention after deletion
- Event/WebSocket/API visibility when mutations publish downstream state

Do not treat any status below 500 as success. Assert the exact expected status
for each intentional success and failure case, then validate the response body
instead of only checking that it is JSON.

### 5. Run the project CRUD suite

Run the focused suite with output enabled so report paths are visible:

```powershell
$env:PROCESSOR_MODE = "mock"
if (-not $env:TEST_DATABASE_URL) { throw "Set TEST_DATABASE_URL to a disposable PostgreSQL database before running CRUD tests." }
python -m pytest tests/test_crud_api.py -q -s
```

When isolating a module, use its class name, for example:

```powershell
python -m pytest tests/test_crud_api.py -q -s -k "TestCamerasCrud"
```

Run module-specific tests after the app-by-app pass, then run the full
regression suite when the focused tests pass:

```powershell
python -m pytest -q
```

Use the project's three-retry repair policy for failures. Do not weaken,
delete, skip, or relabel a test merely to obtain a green result. Distinguish
pre-existing failures from failures introduced by the current run.

### 6. Review reports and evidence

The CRUD suite writes reports under `.agentic/reports/`:

- `crud-analysis-latest.json`
- `crud-analysis-latest.md`

Review every module, not only the aggregate pass rate. A module is complete
only when its tested endpoints have expected status codes, response contracts,
validation/error paths, permission paths, and side effects verified. Report
`NOT_TESTED`, `SKIPPED`, `BLOCKED`, and `UNKNOWN` explicitly; never convert
them to passes.

The final report must include:

- Commands and environment mode used, with secrets redacted
- Module-by-module pass/fail/skip counts
- Endpoint and method for every failure
- Expected versus actual status and response shape
- Validation, permission, persistence, cleanup, and downstream evidence
- Pre-existing failures and unresolved failures separately
- Runtime/deployment checks that were not possible
- Recommended next action for every failure or gap

## Completion Criteria

The CRUD testing task is complete only when:

- All discovered apps were classified and processed sequentially.
- Every applicable CRUD operation has a direct test or an explicit documented gap.
- Success, validation, authorization, not-found, conflict, and dependency paths
  were exercised where applicable.
- Response status, content type, schema/shape, and persisted values were checked.
- Owned resources were cleaned up or their retention was verified.
- `.agentic/reports/crud-analysis-latest.json` and `.md` were reviewed.
- Focused tests pass, or failures are recorded with exact reproduction details.
- The regression result and any missing runtime validation are reported honestly.
