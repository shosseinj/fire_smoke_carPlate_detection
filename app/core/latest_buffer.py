from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

from app.core.types import FramePacket


@dataclass(frozen=True, slots=True)
class BufferStats:
    policy: str
    pending_sources: int
    queue_depth: int
    queue_capacity: int | None
    accepted: int
    stale_replaced: int
    blocked_submissions: int
    full_rejections: int
    rejected_after_close: int
    accepted_by_source: dict[str, int]
    stale_replaced_by_source: dict[str, int]


class LatestPerSourceBuffer:
    """A bounded-latency, fair queue.

    Only the newest pending frame for each source is retained. A source ID is
    present once in the order deque, so fast cameras cannot starve slow ones.
    """

    def __init__(
        self,
        *,
        policy: str = "latest_per_source",
        capacity: int = 256,
        block_timeout_seconds: float = 1.0,
    ) -> None:
        if policy not in {"latest_per_source", "lossless_fifo"}:
            raise ValueError("policy must be latest_per_source or lossless_fifo")
        self._condition = threading.Condition()
        self._policy = policy
        self._capacity = max(1, int(capacity))
        timeout = float(block_timeout_seconds)
        self._block_timeout_seconds = timeout if timeout > 0 else None
        self._latest: dict[str, FramePacket] = {}
        self._order: deque[str] = deque()
        self._fifo: deque[FramePacket] = deque()
        self._closed = False
        self._accepted = 0
        self._stale_replaced = 0
        self._blocked_submissions = 0
        self._full_rejections = 0
        self._rejected_after_close = 0
        self._accepted_by_source: dict[str, int] = {}
        self._stale_replaced_by_source: dict[str, int] = {}

    def put(self, packet: FramePacket) -> bool:
        with self._condition:
            if self._closed:
                self._rejected_after_close += 1
                return False
            if self._policy == "lossless_fifo":
                deadline = (
                    time.monotonic() + self._block_timeout_seconds
                    if self._block_timeout_seconds is not None
                    else None
                )
                while len(self._fifo) >= self._capacity and not self._closed:
                    self._blocked_submissions += 1
                    remaining = (
                        deadline - time.monotonic() if deadline is not None else None
                    )
                    if remaining is not None and remaining <= 0:
                        self._full_rejections += 1
                        return False
                    self._condition.wait(timeout=remaining)
                if self._closed:
                    self._rejected_after_close += 1
                    return False
                self._fifo.append(packet)
                self._record_accepted(packet)
                self._condition.notify_all()
                return True
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
            # Wake all worker threads so they can pull available frames
            # concurrently instead of serialising on one thread.
            self._condition.notify_all()
            return True

    def _record_accepted(self, packet: FramePacket) -> None:
        self._accepted += 1
        self._accepted_by_source[packet.source_id] = (
            self._accepted_by_source.get(packet.source_id, 0) + 1
        )

    def take_batch(self, maximum: int, max_wait_seconds: float) -> list[FramePacket]:
        maximum = max(1, maximum)
        max_wait_seconds = max(0.0, max_wait_seconds)
        with self._condition:
            pending = self._fifo if self._policy == "lossless_fifo" else self._order
            while not pending and not self._closed:
                self._condition.wait()
            if not pending and self._closed:
                return []

            deadline = time.monotonic() + max_wait_seconds
            while len(pending) < maximum and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)

            packets: list[FramePacket] = []
            if self._policy == "lossless_fifo":
                while self._fifo and len(packets) < maximum:
                    packets.append(self._fifo.popleft())
                self._condition.notify_all()
            else:
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

    def clear(self) -> int:
        with self._condition:
            discarded = (
                len(self._fifo)
                if self._policy == "lossless_fifo"
                else len(self._latest)
            )
            self._fifo.clear()
            self._latest.clear()
            self._order.clear()
            self._condition.notify_all()
            return discarded

    def stats(self) -> BufferStats:
        with self._condition:
            return BufferStats(
                policy=self._policy,
                pending_sources=(
                    len({packet.source_id for packet in self._fifo})
                    if self._policy == "lossless_fifo"
                    else len(self._latest)
                ),
                queue_depth=(len(self._fifo) if self._policy == "lossless_fifo" else len(self._latest)),
                queue_capacity=(self._capacity if self._policy == "lossless_fifo" else None),
                accepted=self._accepted,
                stale_replaced=self._stale_replaced,
                blocked_submissions=self._blocked_submissions,
                full_rejections=self._full_rejections,
                rejected_after_close=self._rejected_after_close,
                accepted_by_source=dict(self._accepted_by_source),
                stale_replaced_by_source=dict(self._stale_replaced_by_source),
            )
