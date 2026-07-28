"""Thread-safe demand tracking for on-demand video and AI streams."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
from typing import Callable, Literal


LOGGER = logging.getLogger(__name__)

StreamKind = Literal["video", "ai"]
DemandSnapshot = dict[str, int | bool]
DemandListener = Callable[[DemandSnapshot], None]


@dataclass
class StreamDemandLease:
    """One idempotently releasable subscriber registration."""

    _controller: "StreamDemandController"
    _kind: StreamKind
    _released: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def release(self) -> bool:
        """Release this subscriber once; return whether the count changed."""

        with self._lock:
            if self._released:
                return False
            self._released = True
        self._controller.release(self._kind)
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

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._video_subscribers = 0
        self._ai_subscribers = 0
        self._listeners: list[DemandListener] = []

    def acquire(self, kind: StreamKind) -> StreamDemandLease:
        listeners, snapshot = self._change(kind, 1)
        self._notify(listeners, snapshot)
        return StreamDemandLease(self, kind)

    def acquire_video(self) -> StreamDemandLease:
        return self.acquire("video")

    def acquire_ai(self) -> StreamDemandLease:
        return self.acquire("ai")

    def release(self, kind: StreamKind) -> bool:
        """Remove a subscriber without ever allowing a negative count."""

        with self._lock:
            current = self._count_for(kind)
            if current == 0:
                return False
            if kind == "video":
                self._video_subscribers -= 1
            else:
                self._ai_subscribers -= 1
            listeners = tuple(self._listeners)
            snapshot = self._snapshot_locked()
        self._notify(listeners, snapshot)
        return True

    def release_video(self) -> bool:
        return self.release("video")

    def release_ai(self) -> bool:
        return self.release("ai")

    def video_required(self) -> bool:
        with self._lock:
            return self._video_subscribers > 0

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
        self, kind: StreamKind, delta: int
    ) -> tuple[tuple[DemandListener, ...], DemandSnapshot]:
        with self._lock:
            if kind == "video":
                self._video_subscribers = max(0, self._video_subscribers + delta)
            elif kind == "ai":
                self._ai_subscribers = max(0, self._ai_subscribers + delta)
            else:
                raise ValueError(f"Unsupported stream kind: {kind!r}")
            return tuple(self._listeners), self._snapshot_locked()

    def _count_for(self, kind: StreamKind) -> int:
        if kind == "video":
            return self._video_subscribers
        if kind == "ai":
            return self._ai_subscribers
        raise ValueError(f"Unsupported stream kind: {kind!r}")

    def _snapshot_locked(self) -> DemandSnapshot:
        return {
            "video_subscribers": self._video_subscribers,
            "ai_subscribers": self._ai_subscribers,
            "video_required": self._video_subscribers > 0,
            "ai_required": self._ai_subscribers > 0,
        }

    @staticmethod
    def _notify(
        listeners: tuple[DemandListener, ...], snapshot: DemandSnapshot
    ) -> None:
        for listener in listeners:
            try:
                listener(dict(snapshot))
            except Exception:
                LOGGER.exception("Stream demand listener failed")
