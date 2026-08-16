from __future__ import annotations

import json
import hashlib
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.core.detection_event_schemas import HumanDetectionEvent


def _safe_component(value: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip("._") or "unknown"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{readable[:100]}-{digest}"


class HumanEventAuditStore:
    """Retained, atomic event and media-processing audit files."""

    def __init__(self, root: Path | str,
                 camera_id_resolver: Callable[[str], int | str | None] | None = None) -> None:
        self.root = Path(root)
        self.camera_id_resolver = camera_id_resolver
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def paths(self, event: HumanDetectionEvent) -> tuple[Path, Path]:
        camera_id = self.storage_camera_id(event)
        if self.camera_id_resolver is not None:
            directory = self.root / camera_id
        else:
            directory = self.root / _safe_component(camera_id)
        stem = _safe_component(event.event_id)
        return directory / f"{stem}.event.json", directory / f"{stem}.status.json"

    def storage_camera_id(self, event: HumanDetectionEvent) -> str:
        if self.camera_id_resolver is None:
            return event.camera_id
        resolved = self.camera_id_resolver(event.camera_id)
        if resolved is None or not str(resolved).isdigit() or int(resolved) <= 0:
            raise ValueError("camera source does not resolve to a positive numeric id")
        return str(resolved)

    def persist_event(self, event: HumanDetectionEvent) -> Path:
        event_path, status_path = self.paths(event)
        payload = event.model_dump_json()
        with self._lock:
            event_path.parent.mkdir(parents=True, exist_ok=True)
            if event_path.exists():
                if event_path.read_text("utf-8") != payload:
                    raise ValueError("event id conflicts with retained audit payload")
            else:
                self._atomic_write(event_path, payload)
            if not status_path.exists():
                self._atomic_write(status_path, json.dumps({
                    "schema_version": 1,
                    "event_id": event.event_id,
                    "camera_id": event.camera_id,
                    "storage_camera_id": self.storage_camera_id(event),
                    "state": "event_saved",
                    "event_saved": True,
                    "redis_published": False,
                    "redis_publish_pending": False,
                    "segments_found": 0,
                    "segments_used": [],
                    "concatenation_required": False,
                    "concatenation_succeeded": False,
                    "clip_created": False,
                    "clip_frames": 0,
                    "minio_uploaded": False,
                    "database_finalized": False,
                    "local_archived": False,
                    "acknowledged": False,
                    "ack_pending": False,
                    "dead_lettered": False,
                    "dead_letter_pending": False,
                    "completed": False,
                    "last_error": None,
                    "updated_at_utc": self._now(),
                }, sort_keys=True))
        return event_path

    def update(self, event: HumanDetectionEvent, *, state: str | None = None,
               **values: Any) -> Path:
        self.persist_event(event)
        _, status_path = self.paths(event)
        with self._lock:
            status = json.loads(status_path.read_text("utf-8"))
            status.update(values)
            if state is not None:
                status["state"] = state
            status["updated_at_utc"] = self._now()
            self._atomic_write(status_path, json.dumps(status, sort_keys=True))
        return status_path

    def remove(self, event: HumanDetectionEvent) -> None:
        event_path, status_path = self.paths(event)
        with self._lock:
            status_path.unlink(missing_ok=True)
            event_path.unlink(missing_ok=True)
            try:
                event_path.parent.rmdir()
            except OSError:
                pass

    def move_to_failed(self, event: HumanDetectionEvent) -> tuple[Path, Path]:
        event_path, status_path = self.paths(event)
        failed_directory = self.root / "failed" / self.storage_camera_id(event)
        failed_event = failed_directory / event_path.name
        failed_status = failed_directory / status_path.name
        with self._lock:
            failed_directory.mkdir(parents=True, exist_ok=True)
            if event_path.is_file():
                event_path.replace(failed_event)
            if status_path.is_file():
                status_path.replace(failed_status)
            try:
                event_path.parent.rmdir()
            except OSError:
                pass
        return failed_event, failed_status

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _atomic_write(target: Path, payload: str) -> None:
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(target)
            if os.name != "nt":
                directory_fd = os.open(str(target.parent), os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
