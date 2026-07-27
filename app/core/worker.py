from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
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
    frames_by_source: dict[str, int] = field(default_factory=dict)
    failed_frames_by_source: dict[str, int] = field(default_factory=dict)


class TaskWorker:
    def __init__(
        self,
        *,
        processor: BatchProcessor,
        result_store: ResultStore,
        batch_size: int,
        max_wait_ms: float,
        num_threads: int = 1,
        queue_policy: str = "latest_per_source",
        queue_capacity: int = 512,
        queue_block_timeout_ms: float = 500.0,
        result_callback: Callable[[FramePacket, TaskResult], None] | None = None,
        result_observer: Callable[[FramePacket, TaskResult], None] | None = None,
        location_observer: Callable[[FramePacket, TaskResult], None] | None = None,
    ) -> None:
        self.processor = processor
        self.result_store = result_store
        self.batch_size = max(1, batch_size)
        self.max_wait_seconds = max(0.0, max_wait_ms / 1000.0)
        self.num_threads = max(1, num_threads)
        self.result_callback = result_callback
        self.result_observer = result_observer
        self.location_observer = location_observer
        self.buffer = LatestPerSourceBuffer(
            policy=queue_policy,
            capacity=queue_capacity,
            block_timeout_seconds=queue_block_timeout_ms / 1000.0,
        )
        self.counters = WorkerCounters()
        self._threads: list[threading.Thread] = []
        self._started = threading.Event()
        self._started_count = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._threads and any(t.is_alive() for t in self._threads):
                return
            self._threads = []
            self._started_count = 0
        for i in range(self.num_threads):
            thread = threading.Thread(
                target=self._run,
                name=f"task-worker-{self.processor.task.value}-{i}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        self._started.wait(timeout=2.0)

    def submit(self, packet: FramePacket) -> bool:
        return self.buffer.put(packet)

    def _run(self) -> None:
        with self._lock:
            self._started_count += 1
            if self._started_count >= self.num_threads:
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
                with self._lock:
                    self.counters.batches += 1
                    self.counters.frames += len(packets)
                    for packet in packets:
                        self.counters.frames_by_source[packet.source_id] = (
                            self.counters.frames_by_source.get(packet.source_id, 0) + 1
                        )
                    self.counters.last_batch_size = len(packets)
                    self.counters.last_batch_ms = (time.perf_counter() - started) * 1000.0
                    self.counters.last_error = None
            except Exception as exc:  # keep worker alive after model/runtime errors
                elapsed = (time.perf_counter() - started) * 1000.0
                with self._lock:
                    self.counters.failed_batches += 1
                    self.counters.last_error = str(exc)
                    for packet in packets:
                        self.counters.failed_frames_by_source[packet.source_id] = (
                            self.counters.failed_frames_by_source.get(packet.source_id, 0)
                            + 1
                        )
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
        for thread in self._threads:
            thread.join(timeout=5.0)
        self.processor.close()

    def status(self) -> dict:
        buffer_stats = self.buffer.stats()
        with self._lock:
            counters = asdict(self.counters)
        return {
            "task": self.processor.task.value,
            "batch_size": self.batch_size,
            "max_wait_ms": self.max_wait_seconds * 1000.0,
            "num_threads": self.num_threads,
            "active_threads": sum(1 for t in self._threads if t.is_alive()),
            "counters": counters,
            "buffer": asdict(buffer_stats),
            "processor": self.processor.status(),
        }
