"""Thread-safe demand tracking for on-demand video and AI streams."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Any, Callable, Literal


LOGGER = logging.getLogger(__name__)

StreamKind = Literal["video", "ai"]
DemandSnapshot = dict[str, Any]
DemandListener = Callable[[DemandSnapshot], None]


@dataclass
class StreamDemandLease:
    """One idempotently releasable subscriber registration."""

    _controller: "StreamDemandController"
    _kind: StreamKind
    _source_id: str | None = None
    _released: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def release(self) -> bool:
        """Release this subscriber once; return whether the count changed."""

        with self._lock:
            if self._released:
                return False
            self._released = True
        self._controller.release(self._kind, source_id=self._source_id)
        return True

    def __enter__(self) -> "StreamDemandLease":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class StreamDemandController:
    """Track active consumers and notify pipelines when demand changes.

    Listener callbacks run outside the controller lock. A faulty listener is
    logged and isolated so subscriber accounting remains reliable.
    """

    def __init__(self, *, video_release_grace_seconds: float = 0.0) -> None:
        self._lock = threading.RLock()
        self._global_video_subscribers = 0
        self._video_subscribers_by_source: dict[str, int] = {}
        self._video_release_grace_seconds = max(
            0.0, float(video_release_grace_seconds)
        )
        self._global_video_grace_until = 0.0
        self._video_grace_until_by_source: dict[str, float] = {}
        self._video_grace_generation: dict[str | None, int] = {}
        self._ai_subscribers = 0
        self._listeners: list[DemandListener] = []

    def acquire(
        self, kind: StreamKind, *, source_id: str | None = None
    ) -> StreamDemandLease:
        if kind == "ai" and source_id is not None:
            raise ValueError("Per-source demand is supported only for video streams")
        listeners, snapshot = self._change(kind, 1, source_id=source_id)
        self._notify(listeners, snapshot)
        return StreamDemandLease(self, kind, source_id)

    def acquire_video(self, source_id: str | None = None) -> StreamDemandLease:
        """Acquire global wall demand or demand for one fullscreen source."""

        normalized = self._normalize_source_id(source_id)
        return self.acquire("video", source_id=normalized)

    def acquire_ai(self) -> StreamDemandLease:
        return self.acquire("ai")

    def release(self, kind: StreamKind, *, source_id: str | None = None) -> bool:
        """Remove a subscriber without ever allowing a negative count."""

        normalized = self._normalize_source_id(source_id)
        if kind == "ai" and normalized is not None:
            raise ValueError("Per-source demand is supported only for video streams")
        with self._lock:
            current = self._count_for(kind, source_id=normalized)
            if current == 0:
                return False
            if kind == "video":
                if normalized is None:
                    self._global_video_subscribers -= 1
                else:
                    remaining = current - 1
                    if remaining:
                        self._video_subscribers_by_source[normalized] = remaining
                    else:
                        self._video_subscribers_by_source.pop(normalized, None)
                self._begin_video_grace_locked(normalized)
            else:
                self._ai_subscribers -= 1
            listeners = tuple(self._listeners)
            snapshot = self._snapshot_locked()
        self._notify(listeners, snapshot)
        return True

    def release_video(self, source_id: str | None = None) -> bool:
        return self.release(
            "video", source_id=self._normalize_source_id(source_id)
        )

    def release_ai(self) -> bool:
        return self.release("ai")

    def video_required(self, source_id: str | None = None) -> bool:
        """Return aggregate demand, or demand applicable to one source.

        Global wall subscribers require every source. A source-specific
        subscriber requires only that source. With no argument, this reports
        whether any video branch is required.
        """

        normalized = self._normalize_source_id(source_id)
        with self._lock:
            self._prune_expired_grace_locked()
            global_required = (
                self._global_video_subscribers > 0
                or self._global_video_grace_until > time.monotonic()
            )
            if normalized is None:
                return global_required or bool(self._video_subscribers_by_source) or bool(
                    self._video_grace_until_by_source
                )
            return global_required or (
                self._video_subscribers_by_source.get(normalized, 0) > 0
                or self._video_grace_until_by_source.get(normalized, 0.0)
                > time.monotonic()
            )

    def ai_required(self) -> bool:
        with self._lock:
            return self._ai_subscribers > 0

    def snapshot(self) -> DemandSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def status(self) -> DemandSnapshot:
        return self.snapshot()

    def add_listener(
        self, listener: DemandListener, *, notify_immediately: bool = False
    ) -> Callable[[], None]:
        """Register a demand-change listener and return an unsubscribe callback."""

        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)
            snapshot = self._snapshot_locked()
        if notify_immediately:
            self._notify((listener,), snapshot)

        def unsubscribe() -> None:
            self.remove_listener(listener)

        return unsubscribe

    def remove_listener(self, listener: DemandListener) -> None:
        with self._lock:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

    def _change(
        self,
        kind: StreamKind,
        delta: int,
        *,
        source_id: str | None = None,
    ) -> tuple[tuple[DemandListener, ...], DemandSnapshot]:
        with self._lock:
            if kind == "video":
                self._cancel_video_grace_locked(source_id)
                if source_id is None:
                    self._global_video_subscribers = max(
                        0, self._global_video_subscribers + delta
                    )
                else:
                    count = max(
                        0,
                        self._video_subscribers_by_source.get(source_id, 0) + delta,
                    )
                    if count:
                        self._video_subscribers_by_source[source_id] = count
                    else:
                        self._video_subscribers_by_source.pop(source_id, None)
            elif kind == "ai":
                self._ai_subscribers = max(0, self._ai_subscribers + delta)
            else:
                raise ValueError(f"Unsupported stream kind: {kind!r}")
            return tuple(self._listeners), self._snapshot_locked()

    def _count_for(self, kind: StreamKind, *, source_id: str | None = None) -> int:
        if kind == "video":
            if source_id is None:
                return self._global_video_subscribers
            return self._video_subscribers_by_source.get(source_id, 0)
        if kind == "ai":
            return self._ai_subscribers
        raise ValueError(f"Unsupported stream kind: {kind!r}")

    def _snapshot_locked(self) -> DemandSnapshot:
        self._prune_expired_grace_locked()
        now = time.monotonic()
        subscribers_by_source = dict(self._video_subscribers_by_source)
        grace_by_source = {
            source_id: round(max(0.0, deadline - now), 3)
            for source_id, deadline in self._video_grace_until_by_source.items()
            if deadline > now
        }
        total_video_subscribers = (
            self._global_video_subscribers + sum(subscribers_by_source.values())
        )
        global_grace_remaining = max(0.0, self._global_video_grace_until - now)
        video_required = (
            total_video_subscribers > 0
            or global_grace_remaining > 0
            or bool(grace_by_source)
        )
        return {
            "video_subscribers": total_video_subscribers,
            "global_video_subscribers": self._global_video_subscribers,
            "video_subscribers_by_source": subscribers_by_source,
            "ai_subscribers": self._ai_subscribers,
            "video_required": video_required,
            "ai_required": self._ai_subscribers > 0,
            "video_release_grace_seconds": self._video_release_grace_seconds,
            "global_video_grace_remaining_seconds": round(
                global_grace_remaining, 3
            ),
            "video_grace_remaining_by_source": grace_by_source,
        }

    @staticmethod
    def _normalize_source_id(source_id: str | None) -> str | None:
        if source_id is None:
            return None
        normalized = source_id.strip()
        return normalized or None

    def _cancel_video_grace_locked(self, source_id: str | None) -> None:
        self._video_grace_generation[source_id] = (
            self._video_grace_generation.get(source_id, 0) + 1
        )
        if source_id is None:
            self._global_video_grace_until = 0.0
        else:
            self._video_grace_until_by_source.pop(source_id, None)

    def _begin_video_grace_locked(self, source_id: str | None) -> None:
        if self._count_for("video", source_id=source_id) > 0:
            return
        grace = self._video_release_grace_seconds
        self._video_grace_generation[source_id] = (
            self._video_grace_generation.get(source_id, 0) + 1
        )
        generation = self._video_grace_generation[source_id]
        if grace <= 0:
            self._cancel_video_grace_locked(source_id)
            return
        deadline = time.monotonic() + grace
        if source_id is None:
            self._global_video_grace_until = deadline
        else:
            self._video_grace_until_by_source[source_id] = deadline
        timer = threading.Timer(
            grace, self._expire_video_grace, args=(source_id, generation)
        )
        timer.daemon = True
        timer.start()

    def _expire_video_grace(
        self, source_id: str | None, generation: int
    ) -> None:
        with self._lock:
            if self._video_grace_generation.get(source_id) != generation:
                return
            if self._count_for("video", source_id=source_id) > 0:
                return
            if source_id is None:
                self._global_video_grace_until = 0.0
            else:
                self._video_grace_until_by_source.pop(source_id, None)
            listeners = tuple(self._listeners)
            snapshot = self._snapshot_locked()
        self._notify(listeners, snapshot)

    def _prune_expired_grace_locked(self) -> None:
        now = time.monotonic()
        if self._global_video_grace_until <= now:
            self._global_video_grace_until = 0.0
        expired = [
            source_id
            for source_id, deadline in self._video_grace_until_by_source.items()
            if deadline <= now
        ]
        for source_id in expired:
            self._video_grace_until_by_source.pop(source_id, None)

    @staticmethod
    def _notify(
        listeners: tuple[DemandListener, ...], snapshot: DemandSnapshot
    ) -> None:
        for listener in listeners:
            try:
                listener(dict(snapshot))
            except Exception:
                LOGGER.exception("Stream demand listener failed")
