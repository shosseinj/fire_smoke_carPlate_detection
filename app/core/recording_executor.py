from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

from app.core.live_branch import GpuLiveBranchManager


class RecordingProcess(Protocol):
    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def send_signal(self, sig: int) -> None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RecordingExecution:
    job_id: str
    source_id: str
    start_at_utc: datetime
    end_at_utc: datetime

    def __post_init__(self) -> None:
        if self.start_at_utc.tzinfo is None or self.end_at_utc.tzinfo is None:
            raise ValueError("recording timestamps must be timezone-aware UTC values")
        if self.start_at_utc.utcoffset() != timezone.utc.utcoffset(self.start_at_utc):
            raise ValueError("start_at_utc must be UTC")
        if self.end_at_utc.utcoffset() != timezone.utc.utcoffset(self.end_at_utc):
            raise ValueError("end_at_utc must be UTC")
        if self.end_at_utc <= self.start_at_utc:
            raise ValueError("end_at_utc must be after start_at_utc")


@dataclass(frozen=True, slots=True)
class RecordingResult:
    status: str
    local_path: Path | None
    upload_enqueued: bool


ProcessFactory = Callable[[list[str]], RecordingProcess]
PlayableProbe = Callable[[Path], bool]


def _probe_playable_mp4(path: Path) -> bool:
    try:
        result = subprocess.run(
            ["gst-discoverer-1.0", str(path)], stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False, timeout=10.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0
def _spawn_gstreamer(arguments: list[str]) -> RecordingProcess:
    kwargs: dict[str, object] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "shell": False,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(arguments, **kwargs)  # type: ignore[arg-type,return-value]


class ScheduledRecordingExecutor:
    """Isolated H264 remux execution; it never touches decoder or AI callbacks."""

    def __init__(
        self,
        *,
        live_branches: GpuLiveBranchManager,
        spool_path: Path,
        process_factory: ProcessFactory = _spawn_gstreamer,
        poll_seconds: float = 0.1,
        eos_grace_seconds: float = 10.0,
        now: Callable[[], datetime] | None = None,
        playable_probe: PlayableProbe = _probe_playable_mp4,
    ) -> None:
        self._live_branches = live_branches
        self._spool_path = spool_path
        self._process_factory = process_factory
        self._poll_seconds = max(0.01, poll_seconds)
        self._eos_grace_seconds = max(0.1, eos_grace_seconds)
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._playable_probe = playable_probe

    def execute(self, job: RecordingExecution, cancelled: threading.Event) -> RecordingResult:
        owner_id = f"recording:{hashlib.sha256(job.job_id.encode('utf-8')).hexdigest()}"
        final_path, partial_path = self._paths(job.job_id)
        if final_path.is_file():
            return RecordingResult("already_finalized", final_path, False)
        if self._wait_until(job.start_at_utc, cancelled):
            return RecordingResult("cancelled", None, False)

        acquired = False
        process: RecordingProcess | None = None
        try:
            contract = self._live_branches.acquire_durable(job.source_id, owner_id)
            if not contract.get("enabled"):
                raise RuntimeError("fullscreen live branch is disabled")
            acquired = True
            self._spool_path.mkdir(parents=True, exist_ok=True)
            partial_path.unlink(missing_ok=True)
            process = self._process_factory(self._pipeline(self._live_branches.recording_uri(job.source_id), partial_path))
            cancelled_during_recording, interrupted = self._monitor(process, job.end_at_utc, cancelled)
            self._finish_with_eos(process)
            if not partial_path.is_file():
                raise RuntimeError("recording subprocess produced no output")
            os.replace(partial_path, final_path)
            status = "cancelled_partial" if cancelled_during_recording else ("interrupted_partial" if interrupted else "finalized")
            return RecordingResult(status, final_path, False)
        finally:
            if process is not None and process.poll() is None:
                self._stop_process(process)
            if acquired:
                self._live_branches.release_durable(job.source_id, owner_id)

    def _paths(self, job_id: str) -> tuple[Path, Path]:
        stem = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
        final_path = self._spool_path / f"{stem}.mp4"
        return final_path, final_path.with_suffix(".mp4.partial")

    def finalized_path(self, job_id: str) -> Path:
        return self._paths(job_id)[0]

    def recoverable_path(self, job_id: str) -> Path | None:
        final_path, partial_path = self._paths(job_id)
        if final_path.is_file() and final_path.stat().st_size > 0:
            return final_path
        if partial_path.is_file() and partial_path.stat().st_size > 0 and self._playable_probe(partial_path):
            os.replace(partial_path, final_path)
            return final_path
        return None

    @staticmethod
    def _pipeline(uri: str, partial_path: Path) -> list[str]:
        return [
            "gst-launch-1.0", "-e", "rtspsrc", f"location={uri}", "protocols=tcp",
            "latency=100", "!", "rtph264depay", "!", "h264parse", "!",
            "queue", "max-size-buffers=120", "max-size-bytes=0", "max-size-time=0",
            "leaky=downstream", "!", "mp4mux", "faststart=true", "!", "filesink",
            f"location={partial_path}",
        ]

    def _wait_until(self, target: datetime, cancelled: threading.Event) -> bool:
        while True:
            remaining = (target - self._now()).total_seconds()
            if remaining <= 0:
                return cancelled.is_set()
            if cancelled.wait(min(self._poll_seconds, remaining)):
                return True

    def _monitor(self, process: RecordingProcess, end: datetime, cancelled: threading.Event) -> tuple[bool, bool]:
        while self._now() < end:
            if cancelled.wait(self._poll_seconds):
                return True, False
            code = process.poll()
            if code is not None:
                return False, True
        return cancelled.is_set(), False

    def _finish_with_eos(self, process: RecordingProcess) -> None:
        if process.poll() is not None:
            return
        eos_signal = signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT
        process.send_signal(eos_signal)
        try:
            code = process.wait(timeout=self._eos_grace_seconds)
        except subprocess.TimeoutExpired:
            self._stop_process(process)
            raise RuntimeError("recording subprocess did not finalize after EOS") from None
        if code != 0:
            raise RuntimeError(f"recording subprocess failed while finalizing (code {code})")

    def _stop_process(self, process: RecordingProcess) -> None:
        process.terminate()
        try:
            process.wait(timeout=self._eos_grace_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=self._eos_grace_seconds)
