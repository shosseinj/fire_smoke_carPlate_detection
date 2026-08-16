from __future__ import annotations

import subprocess
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.core.recording_executor import RecordingExecution, ScheduledRecordingExecutor


class _LiveBranches:
    def __init__(self) -> None:
        self.acquired: list[tuple[str, str]] = []
        self.released: list[tuple[str, str]] = []

    def acquire_durable(self, source_id: str, owner_id: str) -> dict[str, object]:
        self.acquired.append((source_id, owner_id))
        return {"enabled": True}

    def release_durable(self, source_id: str, owner_id: str) -> bool:
        self.released.append((source_id, owner_id))
        return True

    def recording_uri(self, _source_id: str) -> str:
        return "rtsp://user:secret@mediamtx:8554/live-branch/fullscreen/hash"


class _Process:
    def __init__(self, output: Path, code: int = 0) -> None:
        self.output = output
        self.code: int | None = None
        self.final_code = code
        self.signals: list[int] = []
        output.write_bytes(b"playable-mp4")

    def poll(self) -> int | None:
        return self.code

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)
        self.code = self.final_code

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.code is None:
            raise subprocess.TimeoutExpired("gst-launch-1.0", 1)
        return self.code

    def terminate(self) -> None:
        self.code = -1

    def kill(self) -> None:
        self.code = -9


class _CancelOnWait:
    def __init__(self) -> None:
        self.cancelled = False

    def is_set(self) -> bool:
        return self.cancelled

    def wait(self, _timeout: float) -> bool:
        self.cancelled = True
        return True


def _job(now: datetime) -> RecordingExecution:
    return RecordingExecution("job/with unsafe name", "camera-private-uri", now, now + timedelta(seconds=1))


def test_remuxes_existing_fullscreen_h264_and_atomically_finalizes(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    live = _LiveBranches()
    uploads: list[tuple[str, Path]] = []
    arguments: list[str] = []

    def spawn(args: list[str]) -> _Process:
        arguments.extend(args)
        output = Path(next(value.split("=", 1)[1] for value in args if value.startswith("location=") and value.endswith(".partial")))
        return _Process(output)

    executor = ScheduledRecordingExecutor(
        live_branches=live,  # type: ignore[arg-type]
        spool_path=tmp_path,
        process_factory=spawn,
        now=lambda: now + timedelta(seconds=2),
    )
    result = executor.execute(_job(now), threading.Event())

    assert result.status == "finalized"
    assert result.local_path is not None and result.local_path.read_bytes() == b"playable-mp4"
    assert not list(tmp_path.glob("*.partial"))
    assert "rtph264depay" in arguments and "h264parse" in arguments and "mp4mux" in arguments
    assert not {"decodebin", "nvv4l2decoder", "x264enc", "nvv4l2h264enc"}.intersection(arguments)
    assert "max-size-buffers=120" in arguments and "leaky=downstream" in arguments
    assert live.released == live.acquired
    assert uploads == []


def test_cancellation_before_start_opens_no_branch_or_process(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    live = _LiveBranches()
    cancelled = threading.Event()
    cancelled.set()
    executor = ScheduledRecordingExecutor(
        live_branches=live,  # type: ignore[arg-type]
        spool_path=tmp_path,
        process_factory=lambda _args: pytest.fail("process must not start"),
        now=lambda: now,
    )

    result = executor.execute(
        RecordingExecution("cancelled", "camera", now + timedelta(hours=1), now + timedelta(hours=2)),
        cancelled,
    )

    assert result.status == "cancelled"
    assert live.acquired == []


def test_cancellation_during_recording_eos_finalizes_playable_partial(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    live = _LiveBranches()
    processes: list[_Process] = []

    def spawn(args: list[str]) -> _Process:
        output = Path(next(value.split("=", 1)[1] for value in args if value.endswith(".partial")))
        process = _Process(output)
        processes.append(process)
        return process

    executor = ScheduledRecordingExecutor(
        live_branches=live,  # type: ignore[arg-type]
        spool_path=tmp_path,
        process_factory=spawn,
        now=lambda: now,
    )

    result = executor.execute(_job(now), _CancelOnWait())  # type: ignore[arg-type]

    assert result.status == "cancelled_partial"
    assert result.local_path is not None and result.local_path.read_bytes() == b"playable-mp4"
    assert processes[0].signals
    assert live.released == live.acquired


def test_retry_of_finalized_job_only_reenqueues_upload(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    live = _LiveBranches()
    uploads: list[tuple[str, Path]] = []
    executor = ScheduledRecordingExecutor(
        live_branches=live,  # type: ignore[arg-type]
        spool_path=tmp_path,
        process_factory=lambda _args: pytest.fail("finalized job must not record again"),
        now=lambda: now,
    )
    final_path, _ = executor._paths("job/with unsafe name")
    final_path.write_bytes(b"existing")

    result = executor.execute(_job(now), threading.Event())

    assert result.status == "already_finalized"
    assert uploads == []
    assert live.acquired == []


def test_owner_is_released_when_subprocess_start_fails(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    live = _LiveBranches()
    executor = ScheduledRecordingExecutor(
        live_branches=live,  # type: ignore[arg-type]
        spool_path=tmp_path,
        process_factory=lambda _args: (_ for _ in ()).throw(RuntimeError("redacted start failure")),
        now=lambda: now + timedelta(seconds=2),
    )

    with pytest.raises(RuntimeError, match="redacted start failure"):
        executor.execute(_job(now), threading.Event())

    assert live.released == live.acquired


def test_source_interruption_finalizes_existing_playable_partial(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    live = _LiveBranches()

    def spawn(args: list[str]) -> _Process:
        output = Path(next(value.split("=", 1)[1] for value in args if value.endswith(".partial")))
        process = _Process(output)
        process.code = 1
        return process

    executor = ScheduledRecordingExecutor(
        live_branches=live,  # type: ignore[arg-type]
        spool_path=tmp_path,
        process_factory=spawn,
        now=lambda: now,
    )
    result = executor.execute(_job(now), threading.Event())

    assert result.status == "interrupted_partial"
    assert result.local_path is not None and result.local_path.read_bytes() == b"playable-mp4"


def test_utc_and_interval_contract_is_strict() -> None:
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError, match="timezone-aware"):
        RecordingExecution("job", "camera", now.replace(tzinfo=None), now)
    with pytest.raises(ValueError, match="after"):
        RecordingExecution("job", "camera", now, now)


def test_restart_promotes_only_playable_partial_artifact(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    executor = ScheduledRecordingExecutor(
        live_branches=_LiveBranches(), spool_path=tmp_path, now=lambda: now,
        playable_probe=lambda path: path.read_bytes() == b"playable",
    )  # type: ignore[arg-type]
    final_path, partial_path = executor._paths("restart-job")
    partial_path.write_bytes(b"playable")

    assert executor.recoverable_path("restart-job") == final_path
    assert final_path.read_bytes() == b"playable" and not partial_path.exists()

pytestmark = pytest.mark.streaming
