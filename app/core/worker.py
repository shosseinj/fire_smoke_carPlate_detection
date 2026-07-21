from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable

from app.core.latest_buffer import LatestPerSourceBuffer
from app.core.result_store import ResultStore
from app.core.types import FramePacket, TaskName, TaskResult
from app.processors.base import BatchProcessor

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class WorkerCounters:
    batches: int = 0
    frames: int = 0
    failed_batches: int = 0
    last_batch_size: int = 0
    last_batch_ms: float = 0.0
    last_error: str | None = None


class TaskWorker:
    def __init__(
        self,
        *,
        processor: BatchProcessor,
        result_store: ResultStore,
        batch_size: int,
        max_wait_ms: float,
        result_callback: Callable[[FramePacket, TaskResult], None] | None = None,
        result_observer: Callable[[FramePacket, TaskResult], None] | None = None,
        location_observer: Callable[[FramePacket, TaskResult], None] | None = None,
    ) -> None:
        self.processor = processor
        self.result_store = result_store
        self.batch_size = max(1, batch_size)
        self.max_wait_seconds = max(0.0, max_wait_ms / 1000.0)
        self.result_callback = result_callback
        self.result_observer = result_observer
        self.location_observer = location_observer
        self.buffer = LatestPerSourceBuffer()
        self.counters = WorkerCounters()
        self._thread: threading.Thread | None = None
        self._started = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"task-worker-{self.processor.task.value}",
            daemon=True,
        )
        self._thread.start()
        self._started.wait(timeout=2.0)

    def submit(self, packet: FramePacket) -> bool:
        return self.buffer.put(packet)

    def _run(self) -> None:
        self._started.set()
        while True:
            packets = self.buffer.take_batch(self.batch_size, self.max_wait_seconds)
            if not packets:
                return
            started = time.perf_counter()
            try:
                results = self.processor.process_batch(packets)
                if len(results) != len(packets):
                    raise RuntimeError(
                        f"Processor {self.processor.task.value} returned {len(results)} "
                        f"results for {len(packets)} packets"
                    )
                for result in results:
                    self.result_store.publish(result)
                for packet, result in zip(packets, results):
                    # self._print_positive_detection(result)
                    if self.result_observer is not None:
                        try:
                            self.result_observer(packet, result)
                        except Exception:
                            LOGGER.exception("Result observer failed")
                    if self.location_observer is not None:
                        try:
                            self.location_observer(packet, result)
                        except Exception:
                            LOGGER.exception("Location observer failed")
                    if self.result_callback is not None:
                        try:
                            self.result_callback(packet, result)
                        except Exception:
                            LOGGER.exception("Annotated broadcast callback failed")
                self.counters.batches += 1
                self.counters.frames += len(packets)
                self.counters.last_batch_size = len(packets)
                self.counters.last_batch_ms = (time.perf_counter() - started) * 1000.0
                self.counters.last_error = None
            except Exception as exc:  # keep worker alive after model/runtime errors
                elapsed = (time.perf_counter() - started) * 1000.0
                self.counters.failed_batches += 1
                self.counters.last_error = str(exc)
                LOGGER.exception("Task batch failed: %s", self.processor.task.value)
                for packet in packets:
                    failed = TaskResult.failed(
                        task=self.processor.task,
                        packet=packet,
                        processing_ms=elapsed,
                        error=exc,
                    )
                    self.result_store.publish(failed)
                    if self.result_callback is not None:
                        try:
                            self.result_callback(packet, failed)
                        except Exception:
                            LOGGER.exception("Annotated broadcast callback failed")

    @staticmethod
    def _print_positive_detection(result: TaskResult) -> None:
        if result.error:
            return
        if result.task == TaskName.FIRE_SMOKE:
            tracks = result.data.get("tracks", [])
            if not tracks:
                return
            counts = {
                "fire": sum(1 for track in tracks if track.get("label") == "fire"),
                "smoke": sum(1 for track in tracks if track.get("label") == "smoke"),
            }
            details = {
                "severity": result.data.get("severity", "none"),
                **counts,
            }
        elif result.task == TaskName.PLATE_RECOGNITION:
            plates = result.data.get("plates", [])
            if not plates:
                return
            details = {
                "plate_count": len(plates),
                "plates": [plate.get("plate") or "unreadable" for plate in plates],
            }
        elif result.task == TaskName.FACE_RECOGNITION:
            faces = result.data.get("faces", [])
            known = [face.get("person") for face in faces if face.get("person") not in {None, "Unknown"}]
            if not known:
                return
            details = {"recognized_count": len(known), "people": known}
        else:
            return
        print(
            f"[DETECTION] source={result.source_id} task={result.task.value} "
            f"frame={result.frame_index} data={json.dumps(details, ensure_ascii=True)}",
            flush=True,
        )

    def close(self) -> None:
        self.buffer.close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.processor.close()

    def status(self) -> dict:
        buffer_stats = self.buffer.stats()
        return {
            "task": self.processor.task.value,
            "batch_size": self.batch_size,
            "max_wait_ms": self.max_wait_seconds * 1000.0,
            "counters": asdict(self.counters),
            "buffer": asdict(buffer_stats),
            "processor": self.processor.status(),
        }
