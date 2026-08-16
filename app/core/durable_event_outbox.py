from __future__ import annotations

import json
import os
import threading
import hashlib
import uuid
from pathlib import Path
from typing import Any

from app.core.detection_event_schemas import HumanDetectionEvent
from app.database import Database, ensure_database


class DurableHumanEventOutbox:
    """Fsync-first human-event publisher with PostgreSQL/Redis recovery."""
    def __init__(self, database: Database | str, redis_client: Any, stream: str, spool: Path,
                 *, poll_seconds: float = 0.25, batch_size: int = 50, max_spool_files: int = 10_000,
                 after_fsync: Any | None = None) -> None:
        self.database, self.redis, self.stream = ensure_database(database), redis_client, stream
        self.spool, self.poll_seconds, self.batch_size = spool, max(.05, poll_seconds), max(1, min(batch_size, 500))
        self.spool.mkdir(parents=True, exist_ok=True)
        if max_spool_files <= 0:
            raise ValueError("outbox spool bound must be positive")
        self.max_spool_files = max_spool_files
        self._after_fsync = after_fsync
        self.quarantine = self.spool / "quarantine"; self.quarantine.mkdir(exist_ok=True)
        self._recover_orphans()
        self._spool_count = sum(1 for _ in self.spool.glob("*.json"))
        self._stop = threading.Event(); self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._metrics = {key: 0 for key in ("spooled", "spool_failed", "db_pending", "published", "publish_failed")}
        self._error: str | None = None

    def publish(self, event: HumanDetectionEvent) -> bool:
        payload = event.model_dump_json()
        digest = hashlib.sha256(event.event_id.encode("utf-8")).hexdigest()
        target = self.spool / f"{digest}.json"
        temporary = self.spool / f"{digest}.{uuid.uuid4().hex}.tmp"
        try:
            with self._lock:
                if target.exists():
                    if target.read_text("utf-8") != payload:
                        raise ValueError("event id conflicts with durable spool payload")
                    return True
                if self._spool_count >= self.max_spool_files:
                    raise RuntimeError("durable event spool is full")
                with temporary.open("x", encoding="utf-8") as output:
                    output.write(payload); output.flush(); os.fsync(output.fileno())
                if self._after_fsync is not None:
                    self._after_fsync(temporary)
                temporary.replace(target)
                self._spool_count += 1
                if os.name != "nt":
                    directory_fd = os.open(str(self.spool), os.O_RDONLY)
                    try: os.fsync(directory_fd)
                    finally: os.close(directory_fd)
            self._metrics["spooled"] += 1
            return True
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            self._metrics["spool_failed"] += 1; self._error = type(exc).__name__
            return False

    def _recover_orphans(self) -> None:
        for path in self.spool.glob("*.tmp"):
            try:
                payload = path.read_text("utf-8")
                event = HumanDetectionEvent.model_validate_json(payload)
                digest = hashlib.sha256(event.event_id.encode("utf-8")).hexdigest()
                if not path.name.startswith(digest + "."):
                    raise ValueError("orphan filename does not match event id")
                target = self.spool / f"{digest}.json"
                if target.exists():
                    if target.read_text("utf-8") != payload:
                        raise ValueError("orphan conflicts with promoted event")
                    path.unlink()
                else:
                    path.replace(target)
                    if os.name != "nt":
                        directory_fd = os.open(str(self.spool), os.O_RDONLY)
                        try: os.fsync(directory_fd)
                        finally: os.close(directory_fd)
            except Exception:
                destination = self.quarantine / f"{path.name}.{uuid.uuid4().hex}.bad"
                path.replace(destination)

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="durable-human-outbox", daemon=True)
            self._thread.start()

    def _import_spool(self) -> None:
        for path in list(self.spool.glob("*.json"))[:self.batch_size]:
            payload = path.read_text("utf-8")
            event = HumanDetectionEvent.model_validate_json(payload)
            with self.database.connection() as connection:
                connection.execute(
                    "INSERT INTO detection_event_outbox (event_id,event_type,stream,payload) VALUES (?,'human',?,?) ON CONFLICT DO NOTHING",
                    (event.event_id, self.stream, payload),
                )
                row = connection.execute("SELECT stream,payload FROM detection_event_outbox WHERE event_id=?", (event.event_id,)).fetchone()
                if row is None or (row["stream"], row["payload"]) != (self.stream, payload):
                    raise ValueError("event id conflicts with PostgreSQL outbox payload")
            path.unlink()
            with self._lock:
                self._spool_count = max(0, self._spool_count - 1)

    def dispatch_once(self) -> int:
        self._import_spool()
        with self.database.connection() as connection:
            rows = connection.execute(
                "SELECT event_id,stream,payload FROM detection_event_outbox WHERE published_at_utc IS NULL ORDER BY created_at_utc LIMIT ?",
                (self.batch_size,),
            ).fetchall()
        self._metrics["db_pending"] = len(rows)
        sent = 0
        for row in rows:
            try:
                self.redis.xadd(row["stream"], {"event": row["payload"]})
                with self.database.connection() as connection:
                    connection.execute("UPDATE detection_event_outbox SET published_at_utc=CURRENT_TIMESTAMP,last_error=NULL WHERE event_id=? AND published_at_utc IS NULL", (row["event_id"],))
                sent += 1; self._metrics["published"] += 1
            except Exception as exc:
                self._metrics["publish_failed"] += 1; self._error = type(exc).__name__
                with self.database.connection() as connection:
                    connection.execute("UPDATE detection_event_outbox SET attempt_count=attempt_count+1,last_error=? WHERE event_id=?", (type(exc).__name__, row["event_id"]))
                break
        return sent

    def _run(self) -> None:
        while not self._stop.is_set():
            try: self.dispatch_once(); self._error = None
            except Exception as exc: self._error = type(exc).__name__
            self._stop.wait(self.poll_seconds)

    def close(self, timeout: float = 5.0) -> bool:
        self._stop.set()
        if self._thread is not None: self._thread.join(timeout)
        return self._thread is None or not self._thread.is_alive()

    def status(self) -> dict[str, Any]:
        # Windows has no portable directory-fsync API. File fsync and atomic
        # replace are still used; Unix additionally fsyncs the spool directory.
        return {**self._metrics, "running": bool(self._thread and self._thread.is_alive()),
                "error": self._error, "directory_fsync_supported": os.name != "nt"}
