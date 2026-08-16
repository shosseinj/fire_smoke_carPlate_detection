from __future__ import annotations

from collections import deque, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
import threading
import time
import uuid
from typing import Callable

from app.core.detection_event_schemas import HumanDetectionEvent
from app.core.types import FramePacket, TaskResult


@dataclass(slots=True)
class _Track:
    first: datetime
    last: datetime
    best_at: datetime
    best_index: int
    bbox: tuple[float, float, float, float]
    width: int
    height: int
    quality: float
    face: bool
    room_id: int | None = None
    attendance: bool = False
    name: str = "Unknown"
    ref_img_id: str | None = None
    score: float = 0.0
    recognized: bool = False
    observations: int = 1


class HumanDetectionEventObserver:
    """Metadata-only accumulator for completed human tracks."""

    def __init__(self, publisher: Callable[[], object | None], *, completed_capacity: int = 4096,
                  active_capacity: int = 4096, pending_capacity: int = 4096,
                  clip_padding_seconds: float = 5.0, completion_capacity: int = 4096,
                  retry_seconds: float = .1, min_observations: int = 1) -> None:
        if min(completed_capacity, active_capacity, pending_capacity) <= 0:
            raise ValueError("observer capacities must be positive")
        self._publisher = publisher
        self._capacity = completed_capacity
        self._active_capacity = active_capacity
        self._pending_capacity = pending_capacity
        self._clip_padding = timedelta(seconds=max(0.0, float(clip_padding_seconds)))
        if completion_capacity <= 0 or retry_seconds <= 0:
            raise ValueError("completion retry bounds must be positive")
        if min_observations <= 0:
            raise ValueError("minimum human track observations must be positive")
        self._min_observations = int(min_observations)
        self._completion_capacity = completion_capacity; self._retry_seconds = retry_seconds
        self._waiting_capacity = completion_capacity
        self._admission_capacity = min(active_capacity, completion_capacity + self._waiting_capacity)
        self._completions: OrderedDict[tuple[str, str, int], HumanDetectionEvent] = OrderedDict()
        # Waiting completions are bounded by admitted active tracks. They retain
        # the fully materialized event and are promoted without another frame.
        self._waiting_completions: OrderedDict[tuple[str, str, int], HumanDetectionEvent] = OrderedDict()
        self._rejected: set[tuple[str, str, int]] = set()
        self._rejected_order: deque[tuple[str, str, int]] = deque()
        self._wake = threading.Event(); self._stop = threading.Event()
        self._active: OrderedDict[tuple[str, str, int], _Track] = OrderedDict()
        self._pending: OrderedDict[tuple[str, str, int], None] = OrderedDict()
        self._completed: set[tuple[str, str, int]] = set()
        self._completed_order: deque[tuple[str, str, int]] = deque()
        self._counts = {k: 0 for k in ("observed", "published", "rejected", "malformed", "duplicate",
                                                   "unresolved_recognized", "active_evicted", "pending_evicted",
                                                   "handoff_full", "retry_attempts", "undurable")}
        self._counts["admission_dropped"] = 0
        self._counts["short_tracks_rejected"] = 0
        self._lock = threading.Lock()
        self._retry_thread = threading.Thread(target=self._retry_completions, name="human-event-completion", daemon=True)
        self._retry_thread.start()

    @staticmethod
    def _time(value: str) -> datetime:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("capture timestamp is not aware")
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _bounded(value: object) -> float:
        return max(0.0, min(1.0, float(value or 0.0)))

    def observe(self, packet: FramePacket, result: TaskResult, *, room_ids_by_track: dict[int, int], counts_for_attendance: bool) -> None:
        # Avoid retaining tracks forever when event publication is disabled/unavailable.
        publisher = self._publisher()
        if publisher is None:
            return
        session, camera = str(result.data.get("tracking_session_id") or ""), str(result.source_id or "")
        try:
            captured = self._time(packet.captured_at_utc)
            size = result.data["source_frame_size"]
            width, height = int(size["width"]), int(size["height"])
            if not session or not camera or width <= 0 or height <= 0:
                raise ValueError
        except Exception:
            with self._lock: self._counts["malformed"] += 1
            return

        with self._lock:
            for human in result.data.get("humans", ()):
                try:
                    track_id = int(human["track_id"]); key = (session, camera, track_id)
                    if key in self._completed or key in self._rejected:
                        continue
                    raw = human["source_bbox"]
                    x1, y1, x2, y2 = (float(raw[i]) for i in range(4))
                    bbox = (max(0.0, min(float(width), x1)), max(0.0, min(float(height), y1)),
                            max(0.0, min(float(width), x2)), max(0.0, min(float(height), y2)))
                    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3] or track_id < 0:
                        raise ValueError
                    face = bool(human.get("face_visible"))
                    quality = self._bounded(human.get("best_face_quality") if face else human.get("confidence"))
                    state = self._active.get(key)
                    if state is None:
                        # Explicit drop-new policy: capacity is reserved before any
                        # partial track state is created. Completed handoffs remain
                        # represented in _active until durable and are never evicted.
                        if len(self._active) >= self._admission_capacity:
                            self._remember_rejected(key)
                            self._counts["admission_dropped"] += 1
                            continue
                        state = self._active[key] = _Track(captured, captured, captured, int(result.frame_index), bbox, width, height, quality, face)
                    else:
                        state.observations += 1
                        state.first = min(state.first, captured)
                        state.last = max(state.last, captured)
                        if (face, quality, -int(result.frame_index)) > (state.face, state.quality, -state.best_index):
                            state.best_at, state.best_index, state.bbox = captured, int(result.frame_index), bbox
                            state.width, state.height, state.quality, state.face = width, height, quality, face
                    self._active.move_to_end(key)
                    if track_id in room_ids_by_track:
                        state.room_id = int(room_ids_by_track[track_id])
                    state.attendance = bool(counts_for_attendance)
                    if human.get("identity_stable"):
                        state.recognized = True
                        state.name = str(human.get("person") or "")
                        state.ref_img_id = str(human.get("ref_img_id") or "")
                        state.score = self._bounded(human.get("recognition_score"))
                    self._counts["observed"] += 1
                    if key in self._pending:
                        self._finalize(key, publisher)
                except Exception:
                    self._counts["malformed"] += 1

            for human in result.data.get("disappeared_humans", ()):
                try:
                    key = (session, camera, int(human["track_id"]))
                    if key in self._completed or key in self._rejected:
                        self._counts["duplicate"] += 1; continue
                    self._pending[key] = None
                    self._pending.move_to_end(key)
                    while len(self._pending) > self._pending_capacity:
                        evicted = next((candidate for candidate in self._pending
                                        if candidate not in self._completions
                                        and candidate not in self._waiting_completions), None)
                        if evicted is None:
                            self._counts["handoff_full"] += 1; break
                        self._pending.pop(evicted, None)
                        self._counts["pending_evicted"] += 1
                    self._finalize(key, publisher)
                except Exception:
                    self._counts["malformed"] += 1

    def _finalize(self, key: tuple[str, str, int], publisher: object) -> None:
        state = self._active.get(key)
        if state is None:
            return
        if state.observations < self._min_observations:
            self._counts["short_tracks_rejected"] += 1
            self._terminal(key)
            return
        match = re.fullmatch(r"personnel_([1-9][0-9]*)", state.ref_img_id or "")
        if state.recognized and (match is None or not state.name.strip()):
            self._counts["unresolved_recognized"] += 1
            self._terminal(key)
            return
        recognized = state.recognized
        now = max(datetime.now(timezone.utc), state.last)
        try:
            event = HumanDetectionEvent(
                event_id=uuid.uuid5(uuid.NAMESPACE_URL, "human:" + ":".join(map(str, key))).hex,
                camera_id=key[1], room_id=state.room_id, tracking_session_id=key[0], track_id=key[2],
                personnel_id=int(match.group(1)) if recognized and match else None,
                ref_img_id=state.ref_img_id if recognized else None, name=state.name if recognized else "Unknown",
                recognition_status="recognized" if recognized else "unknown", recognition_confidence=state.score if recognized else 0.0,
                first_seen_at_utc=state.first, last_seen_at_utc=state.last, best_frame_at_utc=state.best_at,
                best_frame_index=state.best_index, bounding_box=state.bbox, frame_width=state.width, frame_height=state.height,
                snapshot_quality=state.quality, clip_start_at_utc=state.first - self._clip_padding,
                clip_end_at_utc=state.last + self._clip_padding,
                counts_for_attendance=state.attendance, created_at_utc=now)
        except Exception:
            self._counts["malformed"] += 1
            return
        if key not in self._completions and key not in self._waiting_completions:
            if len(self._completions) >= self._completion_capacity:
                # Every completed admitted track has room here because active
                # admission is independently bounded by active_capacity.
                if len(self._waiting_completions) >= self._waiting_capacity:
                    self._counts["handoff_full"] += 1
                    self._counts["undurable"] = len(self._completions) + len(self._waiting_completions) + 1
                    return
                self._waiting_completions[key] = event
                self._counts["handoff_full"] += 1
                self._counts["undurable"] = len(self._completions) + len(self._waiting_completions)
                self._wake.set()
                return
            self._completions[key] = event
            self._counts["undurable"] = len(self._completions) + len(self._waiting_completions)
        self._wake.set()

    def _promote_waiting_locked(self) -> None:
        while len(self._completions) < self._completion_capacity and self._waiting_completions:
            key, event = self._waiting_completions.popitem(last=False)
            self._completions[key] = event
        self._counts["undurable"] = len(self._completions) + len(self._waiting_completions)

    def _retry_completions(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self._retry_seconds); self._wake.clear()
            with self._lock:
                self._promote_waiting_locked()
                item = next(iter(self._completions.items()), None)
            if item is None:
                if self._stop.is_set(): return
                continue
            key, event = item; publisher = self._publisher()
            accepted = False
            if publisher is not None:
                try: accepted = bool(publisher.publish(event))  # type: ignore[attr-defined]
                except Exception: accepted = False
            with self._lock:
                self._counts["retry_attempts"] += 1
                if accepted and self._completions.get(key) is event:
                    self._completions.pop(key, None); self._counts["published"] += 1
                    self._terminal(key); self._promote_waiting_locked()
                elif not accepted:
                    self._counts["rejected"] += 1
            if not accepted:
                self._stop.wait(self._retry_seconds)

    def close(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        self._wake.set()
        while time.monotonic() < deadline:
            with self._lock:
                if not self._completions and not self._waiting_completions: break
            self._wake.set(); time.sleep(min(.01, max(0.0, deadline - time.monotonic())))
        self._stop.set(); self._wake.set(); self._retry_thread.join(max(0.0, deadline - time.monotonic()))
        with self._lock: remaining = len(self._completions) + len(self._waiting_completions)
        return not self._retry_thread.is_alive() and remaining == 0

    def _terminal(self, key: tuple[str, str, int]) -> None:
        self._active.pop(key, None)
        self._pending.pop(key, None)
        self._remember(key)

    def _remember(self, key: tuple[str, str, int]) -> None:
        self._completed.add(key); self._completed_order.append(key)
        while len(self._completed_order) > self._capacity:
            self._completed.discard(self._completed_order.popleft())

    def _remember_rejected(self, key: tuple[str, str, int]) -> None:
        if key in self._rejected:
            return
        self._rejected.add(key); self._rejected_order.append(key)
        while len(self._rejected_order) > self._capacity:
            self._rejected.discard(self._rejected_order.popleft())

    def status(self) -> dict[str, int]:
        with self._lock:
            return {**self._counts, "active": len(self._active), "completed": len(self._completed),
                    "pending": len(self._pending), "completion_primary": len(self._completions),
                    "completion_waiting": len(self._waiting_completions),
                    "rejected_cached": len(self._rejected),
                    "min_observations": self._min_observations}
