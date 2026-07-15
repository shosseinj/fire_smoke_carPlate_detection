from __future__ import annotations

import queue
import threading
import uuid
from collections import deque
from typing import Any

from app.core.types import TaskName, TaskResult


class ResultStore:
    def __init__(self, maximum_recent: int = 1000) -> None:
        self._lock = threading.RLock()
        self._recent: deque[TaskResult] = deque(maxlen=max(1, maximum_recent))
        self._subscribers: dict[str, queue.Queue[dict[str, Any]]] = {}

    def publish(self, result: TaskResult) -> None:
        payload = result.to_dict()
        with self._lock:
            self._recent.append(result)
            stale: list[str] = []
            for subscriber_id, target in self._subscribers.items():
                try:
                    target.put_nowait(payload)
                except queue.Full:
                    try:
                        target.get_nowait()
                        target.task_done()
                        target.put_nowait(payload)
                    except (queue.Empty, queue.Full):
                        stale.append(subscriber_id)
            for subscriber_id in stale:
                self._subscribers.pop(subscriber_id, None)

    def recent(
        self,
        *,
        source_id: str | None = None,
        task: TaskName | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 1000))
        with self._lock:
            values = list(self._recent)
        filtered = [
            item
            for item in reversed(values)
            if (source_id is None or item.source_id == source_id)
            and (task is None or item.task == task)
        ]
        return [item.to_dict() for item in filtered[:limit]]

    def subscribe(self, maximum_queue: int = 200) -> tuple[str, queue.Queue[dict[str, Any]]]:
        subscriber_id = uuid.uuid4().hex
        target: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max(1, maximum_queue))
        with self._lock:
            self._subscribers[subscriber_id] = target
        return subscriber_id, target

    def unsubscribe(self, subscriber_id: str) -> None:
        with self._lock:
            self._subscribers.pop(subscriber_id, None)
