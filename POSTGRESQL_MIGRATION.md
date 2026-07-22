# PostgreSQL migration

The application database now uses one shared PostgreSQL database through SQLAlchemy and `psycopg2`. Application timestamps are stored as timezone-aware PostgreSQL timestamps, with the database session forced to UTC. Date-only business fields use `DATE`; shift clock values use `TIME`.

## 1. Install the driver

```bash
pip install -r requirements-postgres.txt
```

## 2. Configure the server

The project default is:

```text
postgresql+psycopg2://postgres:Reza1995@host.docker.internal:5432/ai_database
```

Prefer setting it explicitly in the container:

```bash
-e DATABASE_URL="postgresql+psycopg2://postgres:YOUR_PASSWORD@host.docker.internal:5432/ai_database"
```

On Linux Docker, `host.docker.internal` may require:

```bash
--add-host=host.docker.internal:host-gateway
```

Create the database before starting the API:

```sql
CREATE DATABASE ai_database;
```

The application creates missing tables and indexes on startup.

## 3. Verify connectivity

```bash
python scripts/verify_postgres.py
```

## 4. Copy legacy SQLite data (optional)

Stop the API first and back up both databases. Then run:

```bash
python scripts/migrate_sqlite_to_postgres.py /path/to/ai_database \
  --database-url "postgresql+psycopg2://postgres:YOUR_PASSWORD@host.docker.internal:5432/ai_database" \
  --truncate
```

Multiple legacy SQLite files can be supplied in one command. The script copies only recognized application tables and resets PostgreSQL sequences afterward.

## Scope

The application’s users, cameras, personnel, locations, shifts, holidays, requests, settings, model metadata, and event logs use PostgreSQL. The local SQLite fallback inside face-vector storage remains intentionally separate because it is a specialized vector fallback, not the application relational database. Qdrant remains the preferred face-vector store.
