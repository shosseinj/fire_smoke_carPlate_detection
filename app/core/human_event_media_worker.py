from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Callable

from app.core.detection_event_schemas import HumanDetectionEvent
from app.core.human_event_extractor import extract_human_media
from app.core.recording_segment_store import IncompleteCoverageError, RecordingSegmentStore
from app.core.recording_storage import (
    ObjectConflictError, RecordingStorageService, sha256_file,
)


LOGGER = logging.getLogger(__name__)


class HumanEventMediaWorker:
    """Bounded embedded consumer. ACKs only durable success or durable dead-letter."""
    def __init__(self, redis_client: Any, stream: str, segment_store: RecordingSegmentStore,
                 storage: RecordingStorageService, finalizer: Callable[[Any, str, str], Any], *,
                 group: str = "detection-media-v1", consumer: str = "embedded-worker",
                 dead_letter_stream: str = "detection:human:dead:v1", block_ms: int = 1000,
                  claim_idle_ms: int = 30000, max_attempts: int = 5, max_segments: int = 16,
                  max_duration_seconds: float = 120.0, max_temp_bytes: int = 2 * 1024**3,
                  temp_root: Path | str = "saved_media/temporary_minIO/human_track",
                  local_root: Path | str = "saved_media/human_track",
                  write_enabled: bool = True) -> None:
        if min(block_ms, claim_idle_ms, max_attempts, max_segments, max_temp_bytes) <= 0 or max_duration_seconds <= 0:
            raise ValueError("human media worker bounds must be positive")
        self.redis, self.stream, self.segment_store, self.storage = redis_client, stream, segment_store, storage
        self.finalizer, self.group, self.consumer, self.dead_letter_stream = finalizer, group, consumer, dead_letter_stream
        self.block_ms, self.claim_idle_ms, self.max_attempts = max(1, block_ms), max(1, claim_idle_ms), max(1, max_attempts)
        self.max_segments, self.max_duration_seconds, self.max_temp_bytes = max_segments, max_duration_seconds, max_temp_bytes
        self.temp_root = Path(temp_root)
        self.local_root = Path(local_root)
        self.write_enabled = write_enabled
        self._stop = threading.Event(); self._thread: threading.Thread | None = None
        self._metrics = {key: 0 for key in ("processed", "retried", "dead_lettered", "errors")}

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            self.redis.xgroup_create(self.stream, self.group, id="$", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        self._thread = threading.Thread(target=self._run, name="human-event-media", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                claimed = self.redis.xautoclaim(self.stream, self.group, self.consumer,
                                                self.claim_idle_ms, "0-0", count=1)
                entries = claimed[1] if claimed and len(claimed) > 1 else []
                if not entries:
                    response = self.redis.xreadgroup(self.group, self.consumer, {self.stream: ">"},
                                                     count=1, block=self.block_ms)
                    entries = response[0][1] if response else []
                for message_id, fields in entries:
                    self._handle(message_id, fields)
            except Exception:
                self._metrics["errors"] += 1
                self._stop.wait(min(1.0, self.block_ms / 1000))

    @staticmethod
    def _value(fields: dict[Any, Any], key: str) -> str:
        value = fields.get(key, fields.get(key.encode()))
        return value.decode() if isinstance(value, bytes) else str(value)

    def _delivery_count(self, message_id: Any) -> int:
        try:
            rows = self.redis.xpending_range(self.stream, self.group, message_id, message_id, 1)
            if rows:
                return int(rows[0].get("times_delivered", rows[0].get(b"times_delivered", 1)))
        except Exception:
            pass
        return 1

    def _handle(self, message_id: Any, fields: dict[Any, Any]) -> None:
        payload = self._value(fields, "event")
        try:
            event = HumanDetectionEvent.model_validate_json(payload)
            segments = self.segment_store.match(event.camera_id, event.clip_start_at_utc,
                                                event.clip_end_at_utc, max_segments=self.max_segments)
            if not self.write_enabled:
                self._metrics["retried"] += 1
                return
            self.temp_root.mkdir(parents=True, exist_ok=True)
            camera = re.sub(r"[^A-Za-z0-9_.-]", "_", event.camera_id).strip("._") or "unknown"
            event_id = re.sub(r"[^A-Za-z0-9_.-]", "_", event.event_id).strip("._") or "unknown"
            root = self.temp_root / camera
            root.mkdir(parents=True, exist_ok=True)
            downloads = [root / f".{event_id}.segment-{index}.mp4"
                         for index in range(len(segments))]
            clip, snapshot = root / f"{event_id}.mp4", root / f"{event_id}.jpg"
            total = 0
            for segment in segments:
                stat = self.storage.stat(segment.object_key)
                total += int(stat.size)
            # Reserve 25% for re-encoded output and JPEG before downloading.
            if total <= 0 or total + max(total // 4, 16 * 1024 * 1024) > self.max_temp_bytes:
                raise ValueError("temporary media bound exceeded")
            total = 0
            for index, segment in enumerate(segments):
                path = self.storage.download_object(segment.object_key, downloads[index],
                                                    expected_sha256=segment.sha256)
                total += path.stat().st_size
                if total > self.max_temp_bytes:
                    raise ValueError("temporary media bound exceeded")
            extract_human_media(downloads, [s.started_at_utc for s in segments],
                                event.clip_start_at_utc, event.clip_end_at_utc,
                                event.best_frame_at_utc, clip, snapshot,
                                max_segments=self.max_segments,
                                max_duration_seconds=self.max_duration_seconds)
            prefix = f"human_track/{event.camera_id}/{event.best_frame_at_utc:%Y/%m/%d}/{event.event_id}"
            clip_key, snapshot_key = f"{prefix}/clip.mp4", f"{prefix}/snapshot.jpg"
            self.storage.upload_object(clip_key, clip, "video/mp4")
            self.storage.upload_object(snapshot_key, snapshot, "image/jpeg")
            bucket = self.storage.settings.bucket_name
            self.finalizer(event, f"minio://{bucket}/{clip_key}", f"minio://{bucket}/{snapshot_key}")
            durable_root = self.local_root / Path(prefix).relative_to("human_track")
            self._archive_file(clip, durable_root / "clip.mp4")
            self._archive_file(snapshot, durable_root / "snapshot.jpg")
            for path in downloads:
                path.unlink(missing_ok=True)
            try:
                root.rmdir()
            except OSError:
                pass
            self.redis.xack(self.stream, self.group, message_id)
            self._metrics["processed"] += 1
        except Exception as exc:
            LOGGER.warning(
                "Human event media processing failed: event_id=%s error=%s detail=%s",
                getattr(locals().get("event"), "event_id", "unknown"),
                type(exc).__name__, exc,
            )
            terminal = isinstance(exc, (ValueError, ObjectConflictError)) and not isinstance(exc, IncompleteCoverageError)
            exhausted = self._delivery_count(message_id) >= self.max_attempts
            if terminal or exhausted:
                try:
                    self.redis.xadd(self.dead_letter_stream, {"source_stream": self.stream,
                        "source_id": message_id, "event": payload, "error": type(exc).__name__})
                    self.redis.xack(self.stream, self.group, message_id)
                    self._metrics["dead_lettered"] += 1
                except Exception:
                    self._metrics["errors"] += 1
            else:
                self._metrics["retried"] += 1

    @staticmethod
    def _archive_file(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            if sha256_file(source) != sha256_file(target):
                raise ObjectConflictError(f"local human media conflicts with durable copy: {target}")
            source.unlink()
            return
        source.replace(target)

    def close(self, timeout: float = 5.0) -> bool:
        self._stop.set()
        # This client is dedicated to the worker. Disconnecting its pool actively
        # interrupts a blocking XREADGROUP; runtime closes the client object later.
        pool = getattr(self.redis, "connection_pool", None)
        if pool is not None:
            try: pool.disconnect()
            except Exception: pass
        if self._thread is not None:
            self._thread.join(timeout)
        return self._thread is None or not self._thread.is_alive()

    def status(self) -> dict[str, Any]:
        return {"enabled": True, "running": bool(self._thread and self._thread.is_alive()), **self._metrics}
