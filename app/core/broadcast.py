from __future__ import annotations

import queue
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.core.types import FramePacket, TaskName, TaskResult
from app.core.source_registry import SourceChange


TASK_LABELS = {
    TaskName.FIRE_SMOKE: "FIRE / SMOKE",
    TaskName.PLATE_RECOGNITION: "PLATE",
}


@dataclass(slots=True)
class PendingAnnotatedFrame:
    frame: np.ndarray
    expected_tasks: set[TaskName]
    results: dict[TaskName, TaskResult] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EncodedBroadcastFrame:
    version: int
    source_id: str
    frame_index: int
    jpeg: bytes
    tasks: tuple[str, ...]
    updated_monotonic: float


@dataclass(frozen=True, slots=True)
class BroadcastControlEvent:
    payload: dict[str, Any]


class AnnotatedBroadcastHub:
    """Composes exact-frame AI results and exposes latest annotated JPEGs."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        jpeg_quality: int = 82,
        pending_frames_per_source: int = 12,
    ) -> None:
        self._enabled = enabled
        self.jpeg_quality = max(40, min(jpeg_quality, 100))
        self.pending_frames_per_source = max(2, pending_frames_per_source)
        self._condition = threading.Condition(threading.RLock())
        self._pending: dict[
            str, OrderedDict[int, PendingAnnotatedFrame]
        ] = defaultdict(OrderedDict)
        self._latest: dict[str, EncodedBroadcastFrame] = {}
        self._subscribers: dict[
            str, queue.Queue[EncodedBroadcastFrame | BroadcastControlEvent | None]
        ] = {}
        self._version = 0
        self._rendered_frames = 0
        self._encode_failures = 0

    @property
    def enabled(self) -> bool:
        with self._condition:
            return self._enabled

    def set_enabled(self, enabled: bool) -> bool:
        with self._condition:
            self._enabled = bool(enabled)
            if not self._enabled:
                self._pending.clear()
                self._latest.clear()
                for target in self._subscribers.values():
                    try:
                        target.put_nowait(None)
                    except queue.Full:
                        try:
                            target.get_nowait()
                            target.task_done()
                            target.put_nowait(None)
                        except (queue.Empty, queue.Full):
                            pass
            self._condition.notify_all()
            return self._enabled

    def publish_source_change(self, change: SourceChange) -> None:
        record = change.record
        camera = None
        if record is not None:
            camera = {
                "camera_id": record.source_id,
                "name": record.name,
                "enabled": record.enabled,
                "tasks": sorted(task.value for task in record.tasks),
                "updated_at_utc": record.updated_at_utc,
            }
        event = BroadcastControlEvent(
            payload={
                "type": "camera_changed",
                "action": change.action,
                "camera_id": change.source_id,
                "revision": change.revision,
                "camera": camera,
            }
        )
        with self._condition:
            self._pending.pop(change.source_id, None)
            self._latest.pop(change.source_id, None)
            if self._enabled:
                for target in self._subscribers.values():
                    try:
                        target.put_nowait(event)
                    except queue.Full:
                        try:
                            target.get_nowait()
                            target.task_done()
                            target.put_nowait(event)
                        except (queue.Empty, queue.Full):
                            pass
            self._condition.notify_all()

    @staticmethod
    def _expected_tasks(packet: FramePacket, result: TaskResult) -> set[TaskName]:
        configured = packet.metadata.get("assigned_tasks", [])
        tasks: set[TaskName] = set()
        for value in configured:
            try:
                tasks.add(TaskName(value))
            except ValueError:
                continue
        return tasks or {result.task}

    @staticmethod
    def _bounded_box(box: Any, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            return None
        height, width = frame.shape[:2]
        try:
            x1, y1, x2, y2 = (int(round(float(value))) for value in box[:4])
        except (TypeError, ValueError):
            return None
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        return x1, y1, x2, y2

    @staticmethod
    def _text(
        frame: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
        *,
        scale: float = 0.55,
        thickness: int = 1,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = origin
        x = max(0, min(x, frame.shape[1] - width - 2))
        y = max(height + 4, min(y, frame.shape[0] - baseline - 2))
        cv2.rectangle(
            frame,
            (x, y - height - 5),
            (x + width + 5, y + baseline + 2),
            (12, 16, 24),
            -1,
        )
        cv2.putText(frame, text, (x + 2, y), font, scale, color, thickness, cv2.LINE_AA)

    def _draw_fire_smoke(self, frame: np.ndarray, result: TaskResult) -> str:
        if result.error:
            return "F/S: ERROR"
        tracks = result.data.get("tracks", [])
        for track in tracks:
            box = self._bounded_box(track.get("bbox"), frame)
            if box is None:
                continue
            label = str(track.get("label", "hazard")).lower()
            confidence = float(track.get("confidence", 0.0) or 0.0)
            color = (40, 70, 255) if label == "fire" else (0, 185, 255)
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            self._text(frame, f"{label.upper()} {confidence:.0%}", (x1, y1), color)
        severity = str(result.data.get("severity", "none")).upper()
        return f"F/S: {len(tracks)} BOXES | {severity}"

    def _draw_plates(self, frame: np.ndarray, result: TaskResult) -> str:
        if result.error:
            return "PLATE: ERROR"
        plates = result.data.get("plates", [])
        for plate in plates:
            box = self._bounded_box(plate.get("bbox"), frame)
            if box is None:
                continue
            value = str(plate.get("plate") or "UNREADABLE")
            confidence = float(plate.get("detector_confidence", 0.0) or 0.0)
            color = (55, 220, 95)
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            self._text(frame, f"PLATE {value} {confidence:.0%}", (x1, y1), color)
        return f"PLATE: {len(plates)} BOXES"

    def _render(
        self,
        source_id: str,
        frame_index: int,
        pending: PendingAnnotatedFrame,
    ) -> bytes | None:
        frame = pending.frame.copy()
        height, width = frame.shape[:2]
        statuses: list[str] = []
        for task, result in sorted(pending.results.items(), key=lambda item: item[0].value):
            if task == TaskName.FIRE_SMOKE:
                statuses.append(self._draw_fire_smoke(frame, result))
            elif task == TaskName.PLATE_RECOGNITION:
                statuses.append(self._draw_plates(frame, result))

        header_height = min(76, max(64, height // 8))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, header_height), (9, 13, 22), -1)
        cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)
        task_text = " + ".join(
            TASK_LABELS[task]
            for task in sorted(pending.expected_tasks, key=lambda item: item.value)
        )

        def fitted_scale(text: str, preferred: float, available: int) -> float:
            (text_width, _), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, preferred, 1
            )
            if text_width <= available:
                return preferred
            return max(0.3, preferred * available / max(text_width, 1))

        title = f"{source_id.upper()} | FRAME {frame_index}"
        tasks_line = f"AI TASKS: {task_text}"
        status_line = " | ".join(statuses)
        cv2.putText(
            frame,
            title,
            (9, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            fitted_scale(title, 0.52, width - 18),
            (245, 248, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            tasks_line,
            (9, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            fitted_scale(tasks_line, 0.5, width - 18),
            (130, 224, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            status_line,
            (9, min(header_height - 7, 61)),
            cv2.FONT_HERSHEY_SIMPLEX,
            fitted_scale(status_line, 0.43, width - 18),
            (155, 210, 255),
            1,
            cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            self._encode_failures += 1
            return None
        return encoded.tobytes()

    def publish_result(self, packet: FramePacket, result: TaskResult) -> None:
        with self._condition:
            if not self._enabled:
                return
            expected = self._expected_tasks(packet, result)
            source_pending = self._pending[packet.source_id]
            pending = source_pending.get(packet.frame_index)
            if pending is None:
                pending = PendingAnnotatedFrame(
                    frame=packet.frame.copy(),
                    expected_tasks=expected,
                )
                source_pending[packet.frame_index] = pending
            else:
                pending.expected_tasks.update(expected)
            pending.results[result.task] = result

            while len(source_pending) > self.pending_frames_per_source:
                source_pending.popitem(last=False)

            if not pending.expected_tasks.issubset(pending.results):
                return
            jpeg = self._render(packet.source_id, packet.frame_index, pending)
            source_pending.pop(packet.frame_index, None)
            if jpeg is None:
                return
            latest = self._latest.get(packet.source_id)
            if latest is not None and packet.frame_index < latest.frame_index:
                return
            self._version += 1
            self._rendered_frames += 1
            encoded_frame = EncodedBroadcastFrame(
                version=self._version,
                source_id=packet.source_id,
                frame_index=packet.frame_index,
                jpeg=jpeg,
                tasks=tuple(sorted(task.value for task in pending.expected_tasks)),
                updated_monotonic=time.monotonic(),
            )
            self._latest[packet.source_id] = encoded_frame
            for target in self._subscribers.values():
                try:
                    target.put_nowait(encoded_frame)
                except queue.Full:
                    try:
                        target.get_nowait()
                        target.task_done()
                        target.put_nowait(encoded_frame)
                    except (queue.Empty, queue.Full):
                        pass
            self._condition.notify_all()

    def subscribe(
        self,
        maximum_queue: int = 64,
    ) -> tuple[
        str,
        queue.Queue[EncodedBroadcastFrame | BroadcastControlEvent | None],
    ]:
        subscriber_id = uuid.uuid4().hex
        target: queue.Queue[
            EncodedBroadcastFrame | BroadcastControlEvent | None
        ] = queue.Queue(
            maxsize=max(8, maximum_queue)
        )
        with self._condition:
            self._subscribers[subscriber_id] = target
            for frame in sorted(self._latest.values(), key=lambda item: item.version):
                try:
                    target.put_nowait(frame)
                except queue.Full:
                    break
        return subscriber_id, target

    def unsubscribe(self, subscriber_id: str) -> None:
        with self._condition:
            self._subscribers.pop(subscriber_id, None)

    def latest(self, source_id: str) -> EncodedBroadcastFrame | None:
        with self._condition:
            return self._latest.get(source_id)

    def wait_next(
        self,
        source_id: str,
        after_version: int,
        timeout: float = 10.0,
    ) -> EncodedBroadcastFrame | None:
        with self._condition:
            self._condition.wait_for(
                lambda: not self._enabled
                or (
                    source_id in self._latest
                    and self._latest[source_id].version > after_version
                ),
                timeout=timeout,
            )
            if not self._enabled:
                return None
            frame = self._latest.get(source_id)
            return frame if frame is not None and frame.version > after_version else None

    def status(self) -> dict[str, Any]:
        with self._condition:
            now = time.monotonic()
            return {
                "enabled": self._enabled,
                "rendered_frames": self._rendered_frames,
                "encode_failures": self._encode_failures,
                "websocket_subscribers": len(self._subscribers),
                "active_streams": {
                    source_id: {
                        "frame_index": frame.frame_index,
                        "tasks": list(frame.tasks),
                        "age_seconds": round(max(0.0, now - frame.updated_monotonic), 3),
                    }
                    for source_id, frame in self._latest.items()
                },
            }
