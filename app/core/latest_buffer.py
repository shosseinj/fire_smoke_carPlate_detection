from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

from app.core.types import FramePacket


@dataclass(frozen=True, slots=True)
class BufferStats:
    pending_sources: int
    accepted: int
    stale_replaced: int
    rejected_after_close: int
    accepted_by_source: dict[str, int]
    stale_replaced_by_source: dict[str, int]


class LatestPerSourceBuffer:
    """A bounded-latency, fair queue.

    Only the newest pending frame for each source is retained. A source ID is
    present once in the order deque, so fast cameras cannot starve slow ones.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: dict[str, FramePacket] = {}
        self._order: deque[str] = deque()
        self._closed = False
        self._accepted = 0
        self._stale_replaced = 0
        self._rejected_after_close = 0
        self._accepted_by_source: dict[str, int] = {}
        self._stale_replaced_by_source: dict[str, int] = {}

    def put(self, packet: FramePacket) -> bool:
        with self._condition:
            if self._closed:
                self._rejected_after_close += 1
                return False
            if packet.source_id in self._latest:
                self._stale_replaced += 1
                self._stale_replaced_by_source[packet.source_id] = (
                    self._stale_replaced_by_source.get(packet.source_id, 0) + 1
                )
            else:
                self._order.append(packet.source_id)
            self._latest[packet.source_id] = packet
            self._accepted += 1
            self._accepted_by_source[packet.source_id] = (
                self._accepted_by_source.get(packet.source_id, 0) + 1
            )
            self._condition.notify()
            return True

    def take_batch(self, maximum: int, max_wait_seconds: float) -> list[FramePacket]:
        maximum = max(1, maximum)
        max_wait_seconds = max(0.0, max_wait_seconds)
        with self._condition:
            while not self._order and not self._closed:
                self._condition.wait()
            if not self._order and self._closed:
                return []

            deadline = time.monotonic() + max_wait_seconds
            while len(self._order) < maximum and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)

            packets: list[FramePacket] = []
            while self._order and len(packets) < maximum:
                source_id = self._order.popleft()
                packet = self._latest.pop(source_id, None)
                if packet is not None:
                    packets.append(packet)
            return packets

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def stats(self) -> BufferStats:
        with self._condition:
            return BufferStats(
                pending_sources=len(self._latest),
                accepted=self._accepted,
                stale_replaced=self._stale_replaced,
                rejected_after_close=self._rejected_after_close,
                accepted_by_source=dict(self._accepted_by_source),
                stale_replaced_by_source=dict(self._stale_replaced_by_source),
            )
