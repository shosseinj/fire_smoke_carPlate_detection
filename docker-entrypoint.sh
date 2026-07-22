#!/bin/sh
set -eu

cd /workspace
python3 -m alembic upgrade head
exec python3 -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 9999 \
  --reload \
  --reload-dir /workspace/app
