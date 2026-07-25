from __future__ import annotations

import queue
import logging
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

LOGGER = logging.getLogger(__name__)


TASK_LABELS = {
    TaskName.FIRE_SMOKE: "FIRE / SMOKE",
    TaskName.PLATE_RECOGNITION: "PLATE",
    TaskName.FACE_RECOGNITION: "FACE",
}


@dataclass(slots=True)
class PendingAnnotatedFrame:
    frame: np.ndarray
    expected_tasks: set[TaskName]
    captured_monotonic: float = 0.0
    results: dict[TaskName, TaskResult] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EncodedBroadcastFrame:
    version: int
    source_id: str
    frame_index: int
    jpeg: bytes
    wall_jpeg: bytes
    frame_width: int
    frame_height: int
    wall_width: int
    wall_height: int
    tasks: tuple[str, ...]
    updated_monotonic: float

    def rendition(self, *, full_resolution: bool) -> tuple[bytes, int, int, str]:
        if full_resolution:
            return self.jpeg, self.frame_width, self.frame_height, "full"
        return self.wall_jpeg, self.wall_width, self.wall_height, "wall"


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
        wall_jpeg_quality: int = 70,
        wall_max_width: int = 320,
        wall_max_height: int = 320,
        pending_frames_per_source: int = 12,
        face_overlay_ttl_ms: float = 250.0,
        async_render: bool = True,
        draw_zones: bool = True,
    ) -> None:
        self._enabled = enabled
        self.draw_zones = draw_zones
        self.jpeg_quality = max(40, min(jpeg_quality, 100))
        self.wall_jpeg_quality = max(40, min(wall_jpeg_quality, 100))
        self.wall_max_width = max(16, wall_max_width)
        self.wall_max_height = max(16, wall_max_height)
        self.pending_frames_per_source = max(2, pending_frames_per_source)
        self.face_overlay_ttl_seconds = max(0.0, float(face_overlay_ttl_ms) / 1000.0)
        self._async_render = async_render
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
        self._full_encoded_bytes = 0
        self._wall_encoded_bytes = 0
        self._face_overlay_cache_hits = 0
        self._latest_face_results: dict[str, tuple[float, TaskResult]] = {}
        self._source_zones: dict[str, list[list[list[float]]]] = {}
        self._render_queue: queue.Queue[
            tuple[str, int, PendingAnnotatedFrame]
        ] = queue.Queue(maxsize=64)
        self._render_thread: threading.Thread | None = None
        self._stop_render = threading.Event()
        if self._async_render:
            self._start_render_thread()

    def _start_render_thread(self) -> None:
        if self._render_thread is not None and self._render_thread.is_alive():
            return
        self._stop_render.clear()
        self._render_thread = threading.Thread(
            target=self._render_loop,
            name="broadcast-renderer",
            daemon=True,
        )
        self._render_thread.start()

    def _render_loop(self) -> None:
        while not self._stop_render.is_set():
            try:
                source_id, frame_index, pending = self._render_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._do_render(source_id, frame_index, pending)
            except Exception:
                LOGGER.exception(
                    "Broadcast render failed: source=%s frame=%s tasks=%s",
                    source_id,
                    frame_index,
                    sorted(task.value for task in pending.expected_tasks),
                )

    def _do_render(self, source_id: str, frame_index: int, pending: PendingAnnotatedFrame) -> None:
        rendered = self._render(source_id, frame_index, pending)
        if rendered is None:
            return
        jpeg, wall_jpeg, width, height, wall_width, wall_height = rendered
        with self._condition:
            latest = self._latest.get(source_id)
            if latest is not None and frame_index < latest.frame_index:
                return
            self._version += 1
            self._rendered_frames += 1
            encoded_frame = EncodedBroadcastFrame(
                version=self._version,
                source_id=source_id,
                frame_index=frame_index,
                jpeg=jpeg,
                wall_jpeg=wall_jpeg,
                frame_width=width,
                frame_height=height,
                wall_width=wall_width,
                wall_height=wall_height,
                tasks=tuple(sorted(task.value for task in pending.expected_tasks)),
                updated_monotonic=time.monotonic(),
            )
            self._latest[source_id] = encoded_frame
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

    def set_draw_zones(self, draw_zones: bool) -> None:
        self.draw_zones = bool(draw_zones)

    def set_source_zones(
        self, source_id: str, zones: list[list[list[float]]]
    ) -> None:
        """Set the zone polygons to draw for this source.

        Each entry in zones is a list of [x, y] points defining a closed polygon.
        Pass an empty list to clear zones for this source.
        """
        self._source_zones[source_id] = list(zones)

    def clear_source_zones(self, source_id: str) -> None:
        self._source_zones.pop(source_id, None)

    def close(self) -> None:
        """Stop the render thread and clean up resources."""
        self._stop_render.set()
        if self._render_thread is not None:
            self._render_thread.join(timeout=2.0)
        self.set_enabled(False)

    def publish_source_change(self, change: SourceChange) -> None:
        record = change.record
        camera = None
        if record is not None:
            camera = {
                "source_uri": record.source_uri,
                "name": record.name,
                "enabled": record.enabled,
                "tasks": sorted(task.value for task in record.tasks),
                "frame_width": record.frame_width,
                "frame_height": record.frame_height,
                "updated_at_utc": record.updated_at_utc,
            }
        event = BroadcastControlEvent(
            payload={
                "type": "camera_changed",
                "action": change.action,
                "source_uri": change.source_uri,
                "revision": change.revision,
                "camera": camera,
            }
        )
        with self._condition:
            self._pending.pop(change.source_uri, None)
            self._latest.pop(change.source_uri, None)
            self._latest_face_results.pop(change.source_uri, None)
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

    def publish_control_event(self, payload: dict[str, Any]) -> None:
        event = BroadcastControlEvent(payload=payload)
        with self._condition:
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
            if track.get("confirmed") is False and not track.get("alert_active", False):
                continue
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

    def _draw_faces(self, frame: np.ndarray, result: TaskResult) -> str:
        if result.error:
            return "HUMAN/FACE: ERROR"
        humans = result.data.get("humans", [])
        for human in humans:
            box = self._bounded_box(human.get("bbox"), frame)
            if box is None:
                continue
            person = str(human.get("person") or "Unknown")
            track_id = human.get("track_id")
            score = float(human.get("recognition_score", 0.0) or 0.0)
            known = person != "Unknown"
            color = (70, 230, 100) if known else (0, 190, 255)
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            suffix = f" {score:.0%}" if known else ""
            self._text(
                frame,
                f"HUMAN {person} #{track_id}{suffix}",
                (x1, y1),
                color,
            )
        faces = result.data.get("faces", [])
        for face in faces:
            box = self._bounded_box(face.get("bbox"), frame)
            if box is None:
                continue
            color = (160, 220, 255)
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        recognized = sum(
            1 for human in humans if human.get("person") not in {None, "Unknown"}
        )
        return f"HUMAN: {recognized}/{len(humans)} KNOWN | FACE: {len(faces)}"

    def _draw_zones(self, frame: np.ndarray, source_id: str) -> None:
        """Draw zone polygons on the frame if draw_zones is enabled and zones exist."""
        if not self.draw_zones:
            return
        zones = self._source_zones.get(source_id)
        if not zones:
            return
        height, width = frame.shape[:2]
        for i, polygon in enumerate(zones):
            pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            # Set a distinct color per polygon
            color_palette = [
                (255, 100, 100),   # red
                (100, 255, 100),   # green
                (100, 100, 255),   # blue
                (255, 255, 100),   # yellow
                (255, 100, 255),   # magenta
                (100, 255, 255),   # cyan
            ]
            color = color_palette[i % len(color_palette)]
            # Semi-transparent fill via overlay
            overlay = frame.copy()
            cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)
            cv2.fillPoly(overlay, [pts], color=color)
            # Blend with 25% opacity
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            # Draw thicker border again on top for visibility
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

    def _render(
        self,
        source_id: str,
        frame_index: int,
        pending: PendingAnnotatedFrame,
    ) -> tuple[bytes, bytes, int, int, int, int] | None:
        # Use in-place operations to reduce frame copies
        frame = pending.frame
        height, width = frame.shape[:2]
        statuses: list[str] = []
        results = dict(pending.results)
        if (
            TaskName.FACE_RECOGNITION in pending.expected_tasks
            and TaskName.FACE_RECOGNITION not in results
            and self.face_overlay_ttl_seconds > 0
        ):
            with self._condition:
                cached = self._latest_face_results.get(source_id)
            if cached is not None:
                cached_at, cached_result = cached
                if (
                    cached_result.frame_index <= frame_index
                    and time.monotonic() - cached_at <= self.face_overlay_ttl_seconds
                ):
                    results[TaskName.FACE_RECOGNITION] = cached_result
                    with self._condition:
                        self._face_overlay_cache_hits += 1
        for task, result in sorted(results.items(), key=lambda item: item[0].value):
            if task == TaskName.FIRE_SMOKE:
                statuses.append(self._draw_fire_smoke(frame, result))
            elif task == TaskName.PLATE_RECOGNITION:
                statuses.append(self._draw_plates(frame, result))
            elif task == TaskName.FACE_RECOGNITION:
                statuses.append(self._draw_faces(frame, result))

        # Draw zone polygons (if enabled and zones exist for this source)
        self._draw_zones(frame, source_id)

        # Draw header overlay in-place (no copy needed)
        header_height = min(76, max(64, height // 8))
        cv2.rectangle(frame, (0, 0), (width, header_height), (9, 13, 22), -1)
        
        task_text = (
            " + ".join(
                TASK_LABELS[task]
                for task in sorted(pending.expected_tasks, key=lambda item: item.value)
            )
            if pending.expected_tasks
            else "PLAY ONLY"
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
        status_line = " | ".join(statuses) if statuses else "LIVE VIEW - AI DISABLED"
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
        
        # JPEG encode (CPU)
        ok, full_jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        full_jpeg = full_jpeg.tobytes() if ok else b""
        if not ok:
            self._encode_failures += 1
            return None
        scale = min(
            1.0,
            self.wall_max_width / max(width, 1),
            self.wall_max_height / max(height, 1),
        )
        wall_width = max(1, min(width, int(round(width * scale))))
        wall_height = max(1, min(height, int(round(height * scale))))
        if wall_width == width and wall_height == height:
            wall_jpeg = full_jpeg
        else:
            # Use faster interpolation for wall frame
            wall_frame = cv2.resize(
                frame,
                (wall_width, wall_height),
                interpolation=cv2.INTER_LINEAR,
            )
            wall_ok, wall_jpeg = cv2.imencode(".jpg", wall_frame, [cv2.IMWRITE_JPEG_QUALITY, self.wall_jpeg_quality])
            wall_jpeg = wall_jpeg.tobytes() if wall_ok else b""
            if not wall_ok:
                self._encode_failures += 1
                return None
        self._full_encoded_bytes += len(full_jpeg)
        self._wall_encoded_bytes += len(wall_jpeg)
        return full_jpeg, wall_jpeg, width, height, wall_width, wall_height

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
                    captured_monotonic=packet.captured_monotonic,
                )
                source_pending[packet.frame_index] = pending
            else:
                pending.expected_tasks.update(expected)
            pending.results[result.task] = result
            if result.task == TaskName.FACE_RECOGNITION:
                self._latest_face_results[packet.source_id] = (time.monotonic(), result)

            while len(source_pending) > self.pending_frames_per_source:
                source_pending.popitem(last=False)

            complete = pending.expected_tasks.issubset(pending.results)
            # Fire/smoke is latency-sensitive. Publish its result immediately on
            # the exact source frame instead of waiting for the slower face and
            # plate workers to also finish that frame. If they do, the same frame
            # is rendered again below with the complete result set.
            eager_fire_smoke = result.task == TaskName.FIRE_SMOKE
            if not complete and not eager_fire_smoke:
                return
            
            # Use async rendering to avoid blocking the main pipeline
            if self._async_render:
                try:
                    self._render_queue.put_nowait(
                        (packet.source_id, packet.frame_index, pending)
                    )
                except queue.Full:
                    pass  # Drop frame if render queue is full
                if complete:
                    source_pending.pop(packet.frame_index, None)
                return
            
            # Synchronous rendering (fallback)
            rendered = self._render(packet.source_id, packet.frame_index, pending)
            if complete:
                source_pending.pop(packet.frame_index, None)
            if rendered is None:
                return
            jpeg, wall_jpeg, width, height, wall_width, wall_height = rendered
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
                wall_jpeg=wall_jpeg,
                frame_width=width,
                frame_height=height,
                wall_width=wall_width,
                wall_height=wall_height,
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

    def publish_passthrough(self, packet: FramePacket) -> None:
        """Broadcast an enabled source frame independently of assigned AI work."""
        with self._condition:
            if not self._enabled:
                return
            pending = PendingAnnotatedFrame(
                frame=packet.frame.copy(),
                expected_tasks=set(),
                captured_monotonic=packet.captured_monotonic,
            )
            
            # Use async rendering to avoid blocking the main pipeline
            if self._async_render:
                try:
                    self._render_queue.put_nowait(
                        (packet.source_id, packet.frame_index, pending)
                    )
                except queue.Full:
                    pass  # Drop frame if render queue is full
                return
            
            # Synchronous rendering (fallback)
            rendered = self._render(packet.source_id, packet.frame_index, pending)
            if rendered is None:
                return
            jpeg, wall_jpeg, width, height, wall_width, wall_height = rendered
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
                wall_jpeg=wall_jpeg,
                frame_width=width,
                frame_height=height,
                wall_width=wall_width,
                wall_height=wall_height,
                tasks=(),
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
                "jpeg_quality": self.jpeg_quality,
                "wall_jpeg_quality": self.wall_jpeg_quality,
                "wall_max_width": self.wall_max_width,
                "wall_max_height": self.wall_max_height,
                "full_encoded_bytes": self._full_encoded_bytes,
                "wall_encoded_bytes": self._wall_encoded_bytes,
                "face_overlay_ttl_ms": self.face_overlay_ttl_seconds * 1000.0,
                "face_overlay_cache_hits": self._face_overlay_cache_hits,
                "websocket_subscribers": len(self._subscribers),
                "active_streams": {
                    source_id: {
                        "frame_index": frame.frame_index,
                        "tasks": list(frame.tasks),
                        "full_resolution": [frame.frame_width, frame.frame_height],
                        "wall_resolution": [frame.wall_width, frame.wall_height],
                        "age_seconds": round(max(0.0, now - frame.updated_monotonic), 3),
                    }
                    for source_id, frame in self._latest.items()
                },
            }
