# Query Optimization Implementation Report

## Scope

Implemented safe PostgreSQL query optimizations across the high-priority apps while preserving existing API contracts and attendance calculation logic.

## Implemented optimizations

### Database core
- Replaced Python-loop `executemany()` with one SQLAlchemy/DBAPI bulk execution call.
- Added bounded multi-row insert support for detection imports.
- Replaced several write-then-select flows with PostgreSQL `RETURNING`.

### Locations and room matching
- Added bounded in-process caches for parsed room/camera polygons.
- Invalidates polygon caches when rooms, polygons, activity, or camera assignments change.
- Reuses loaded polygons and batches multi-zone match inserts into one statement.
- Added bulk personnel-room access resolution.
- Fixed direct database connection lifecycle in location APIs.

### Human logs
- Replaced separate insert/update branches and final select with PostgreSQL upsert using `ON CONFLICT ... DO UPDATE ... RETURNING`.
- Preserved existing snapshot, identity, quality, video, and attendance behavior.

### Detection logs and attendance
- Replaced the attendance identity `OR` scan with two index-friendly queries and ID-based deduplication.
- Preserved daily, monthly, and yearly attendance calculation code.
- Added bulk detection-log creation and batched duplicate lookup for Excel imports.
- Detection import now preloads personnel, access rules, and duplicate candidates.
- Delete-all media processing now reads media keys in bounded ID batches instead of loading complete log records.
- Fixed direct database connection lifecycle in detection-log enrichment.

### Plate recognition
- Added a generated `normalized_plate` column and an active-plate index.
- Resolves all OCR candidates from one indexed query per result instead of one query per candidate.
- Plate-log insertion now uses the shared true bulk execution path.
- Plate create/update uses `RETURNING`.

### Personnel and images
- Removed the personnel-images N+1 query.
- Bulk-loaded department, shift, and audit-user display data for personnel lists.
- Personnel Excel import preloads all existing national codes.
- Personnel create/update uses `RETURNING`.

### Personnel requests
- Removed per-request personnel lookups from list endpoints.
- Skips unnecessary count queries for list-only responses.
- Bulk request creation uses one multi-row `INSERT ... RETURNING` statement.

### Administrative locations
- Removed N+1 section queries from building lists.
- Bulk-loaded building names and audit users for section/room lists.

## Alembic migration

Added:

`alembic/versions/20260802_0048_query_optimizations.py`

The migration adds the normalized plate column and query-oriented indexes for:

- detection attendance lookups;
- detection deduplication;
- room/camera time filtering;
- human attendance lookups;
- room-match history;
- personnel-image ordering;
- personnel-request filtering;
- active normalized plate lookup.

The declared application Alembic head is now `20260802_0048`.

## Tests

Focused validation command:

```bash
PYTHONPATH=. pytest -q \
  tests/test_database_models.py \
  tests/test_database_postgresql.py \
  tests/test_query_optimizations.py \
  tests/test_locations.py::TestPointInPolygon \
  tests/test_locations.py::TestParsePolygon
```

Result: **30 passed**.

Also completed:

- Python compilation for `app`, migrations, and tests;
- migration-chain validation: one head, `20260802_0048`;
- PostgreSQL DDL compilation for the new indexes and generated column.

## Validation limitations

The complete test suite could not be collected in this execution environment because these runtime dependencies are unavailable:

- `bcrypt`;
- `jdatetime`;
- `psycopg2`;
- a running PostgreSQL test database.

Some tests also activated their fallback Jalali stub, which does not implement `togregorian()`.

Therefore, real `EXPLAIN (ANALYZE, BUFFERS)` timings and full PostgreSQL integration results were not fabricated. Run the complete suite and migration against the project's normal PostgreSQL environment before production deployment.

## Deployment checklist

```bash
pip install -r requirements.txt
pip install -r requirements-postgres.txt
alembic upgrade head
pytest -q
```

After loading production-like data, inspect the main attendance, detection-list, polygon-match, and plate-lookup queries with `EXPLAIN (ANALYZE, BUFFERS)` and `pg_stat_statements`.
