from __future__ import annotations

from collections import deque, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import re
import threading
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


class HumanDetectionEventObserver:
    """Metadata-only accumulator for completed human tracks."""

    def __init__(self, publisher: Callable[[], object | None], *, completed_capacity: int = 4096,
                 active_capacity: int = 4096, pending_capacity: int = 4096) -> None:
        if min(completed_capacity, active_capacity, pending_capacity) <= 0:
            raise ValueError("observer capacities must be positive")
        self._publisher = publisher
        self._capacity = completed_capacity
        self._active_capacity = active_capacity
        self._pending_capacity = pending_capacity
        self._active: OrderedDict[tuple[str, str, int], _Track] = OrderedDict()
        self._pending: OrderedDict[tuple[str, str, int], None] = OrderedDict()
        self._completed: set[tuple[str, str, int]] = set()
        self._completed_order: deque[tuple[str, str, int]] = deque()
        self._counts = {k: 0 for k in ("observed", "published", "rejected", "malformed", "duplicate",
                                                  "unresolved_recognized", "active_evicted", "pending_evicted")}
        self._lock = threading.Lock()

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
                    if key in self._completed:
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
                        state = self._active[key] = _Track(captured, captured, captured, int(result.frame_index), bbox, width, height, quality, face)
                    else:
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
                    while len(self._active) > self._active_capacity:
                        evicted, _ = self._active.popitem(last=False)
                        self._pending.pop(evicted, None)
                        self._counts["active_evicted"] += 1
                except Exception:
                    self._counts["malformed"] += 1

            for human in result.data.get("disappeared_humans", ()):
                try:
                    key = (session, camera, int(human["track_id"]))
                    if key in self._completed:
                        self._counts["duplicate"] += 1; continue
                    self._pending[key] = None
                    self._pending.move_to_end(key)
                    while len(self._pending) > self._pending_capacity:
                        self._pending.popitem(last=False)
                        self._counts["pending_evicted"] += 1
                    self._finalize(key, publisher)
                except Exception:
                    self._counts["malformed"] += 1

    def _finalize(self, key: tuple[str, str, int], publisher: object) -> None:
        state = self._active.get(key)
        if state is None:
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
                snapshot_quality=state.quality, clip_start_at_utc=state.first, clip_end_at_utc=state.last,
                counts_for_attendance=state.attendance, created_at_utc=now)
        except Exception:
            self._counts["malformed"] += 1
            return
        try:
            accepted = publisher.publish(event)  # type: ignore[attr-defined]
        except Exception:
            accepted = False
        self._counts["published" if accepted else "rejected"] += 1
        self._terminal(key)

    def _terminal(self, key: tuple[str, str, int]) -> None:
        self._active.pop(key, None)
        self._pending.pop(key, None)
        self._remember(key)

    def _remember(self, key: tuple[str, str, int]) -> None:
        self._completed.add(key); self._completed_order.append(key)
        while len(self._completed_order) > self._capacity:
            self._completed.discard(self._completed_order.popleft())

    def status(self) -> dict[str, int]:
        with self._lock:
            return {**self._counts, "active": len(self._active), "completed": len(self._completed),
                    "pending": len(self._pending)}
