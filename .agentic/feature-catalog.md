# Feature Catalog

Record completed features and migrated options only after evidence is available.

For each entry include purpose, input interfaces, runtime consumers, downstream outputs, configuration/defaults, tests, runtime validation status, deployment notes, and rollback path.

## Authentication (JWT Login + Profile)

- **Purpose**: Token-based user authentication and profile retrieval. Provides JWT access tokens for API security and role-based access control.
- **Input interfaces**: `POST /api/v1/auth/login` (username+password JSON body), `GET /api/v1/auth/me` (Bearer token in Authorization header).
- **Runtime consumers**: API clients, frontend dashboard, downstream protected endpoints through `get_current_user()` FastAPI dependency.
- **Downstream outputs**: JWT access token (HS256, 24h default expiry), user profile JSON (id, username, role, is_active, created_at_utc).
- **Configuration/defaults**: `JWT_SECRET_KEY` (env), `JWT_ALGORITHM=HS256`, `JWT_EXPIRY_MINUTES=1440`, `AUTH_DB_PATH=data/auth.sqlite3`, `AUTH_DEFAULT_ADMIN_USERNAME=admin`.
- **Tests**: 14 tests in `tests/test_auth.py` covering valid/invalid login, token structure, /me auth scenarios. `python -m pytest tests/test_auth.py -q` passes.
- **Runtime validation**: Mock-mode tested with `TestClient`. Endpoints registered in OpenAPI spec. Real deployment requires setting a strong JWT_SECRET_KEY.
- **Deployment notes**: Default admin user (admin/admin123) auto-seeded on first startup. Change default password immediately in production. JWT secret must be a strong random value in production.
- **Rollback path**: Remove `app.include_router(auth_router)` from `app/main.py`, delete `app/api/auth.py`, `app/api/auth_schemas.py`, `app/core/auth.py`, `app/core/auth_store.py`, revert `app/config.py` JWT fields, and `tests/test_auth.py`. Git commands: `git checkout -- <files>` or `git revert <commit>`.
- **Excluded neighboring items**: `/api/v1/auth/logout`, `/api/v1/auth/refresh`, `/api/v1/auth/create-admin`, `/api/v1/auth/create-user`, `/api/v1/auth/me/password`, `/api/v1/auth/token`, `/api/v1/auth/users`, `/api/v1/auth/users/{user_id}`, `/api/v1/auth/users/{user_id}/role` — all remain PENDING.
