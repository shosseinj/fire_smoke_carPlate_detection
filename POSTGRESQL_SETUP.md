# PostgreSQL-only database setup

This project now uses PostgreSQL for all built-in persistent application data.
The application does not create or upgrade tables at startup; Alembic is the
single source of truth for schema management.

## 1. Install database dependencies

```bash
pip install -r requirements-postgres.txt
```

## 2. Configure the connection

The current development default is:

```text
postgresql+psycopg2://postgres:Asd12345@host.docker.internal:5432/ai_database
```

Override it through `DATABASE_URL` when needed:

```bash
export DATABASE_URL='postgresql+psycopg2://postgres:password@host:5432/ai_database'
```

PowerShell:

```powershell
$env:DATABASE_URL = 'postgresql+psycopg2://postgres:password@host.docker.internal:5432/ai_database'
```

When the API runs directly on the PostgreSQL server, use `localhost` instead of
`host.docker.internal`. When both services are in Docker Compose, use the
PostgreSQL service name as the host.

## 3. Create the empty database

Run this once in PostgreSQL if `ai_database` does not exist:

```sql
CREATE DATABASE ai_database;
```

## 4. Create the schema

```bash
alembic upgrade head
```

Equivalent helper:

```bash
python scripts/init_postgres.py
```

## 5. Verify

```bash
python scripts/verify_postgres.py
```

## Development schema changes

After changing `app/database.py`, create and review a new revision:

```bash
alembic revision --autogenerate -m "describe change"
alembic upgrade head
```

Downgrade one revision:

```bash
alembic downgrade -1
```

Reset an expendable development database:

```bash
alembic downgrade base
alembic upgrade head
```

## Face embeddings

Face embeddings use the `face_embeddings` PostgreSQL table by default. A remote
Qdrant server is still optional: set `FACE_QDRANT_URL` to use it. No local
relational database fallback is used.

## Security

The development connection string contains a password. For deployment, rotate
that password and inject the replacement using `DATABASE_URL` rather than
committing it to source control.
