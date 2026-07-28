from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.core.rtsp_process_supervisor import RtspProcessSupervisor
from app.core.source_registry import RTSP, SourceRecord


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeProcess:
    _next_pid = 1000

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.exitcode: int | None = None
        self.pid: int | None = None
        self.alive = False
        self.start_calls = 0
        self.join_calls = 0
        self.terminate_calls = 0
        self.kill_calls = 0

    def start(self) -> None:
        self.start_calls += 1
        self.pid = FakeProcess._next_pid
        FakeProcess._next_pid += 1
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        self.join_calls += 1

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.alive = False
        self.exitcode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self.alive = False
        self.exitcode = -9

    def exit(self, code: int) -> None:
        self.exitcode = code
        self.alive = False


class FakeProcessFactory:
    def __init__(self) -> None:
        self.created: list[FakeProcess] = []

    def __call__(self, **kwargs: Any) -> FakeProcess:
        process = FakeProcess(**kwargs)
        self.created.append(process)
        return process


class FakeRegistry:
    def __init__(self, *source_ids: str) -> None:
        self._records = [
            SourceRecord(
                source_uri=source_id,
                name=f"camera-{index}",
                source_type=RTSP,
            )
            for index, source_id in enumerate(source_ids, start=1)
        ]

    def list(self) -> list[SourceRecord]:
        return list(self._records)


class Sink:
    enabled = False

    def submit_frame(self, **_kwargs: Any) -> bool:
        return True

    def submit_round(self, **_kwargs: Any) -> dict[str, int]:
        return {"accepted_sources": 0}


def build_supervisor(
    tmp_path: Path,
    *,
    source_ids: tuple[str, ...] = ("rtsp://camera-1/live",),
    backoff_seconds: tuple[float, ...] = (2.0, 5.0, 10.0, 30.0),
    segfault_quarantine_threshold: int = 99,
    quarantine_seconds: float = 60.0,
) -> tuple[RtspProcessSupervisor, FakeClock, FakeProcessFactory]:
    clock = FakeClock()
    factory = FakeProcessFactory()
    sink = Sink()
    supervisor = RtspProcessSupervisor(
        registry=FakeRegistry(*source_ids),
        router=sink,
        project_root=tmp_path,
        frontend_frame_worker=sink,
        process_factory=factory,
        monotonic=clock,
        backoff_seconds=backoff_seconds,
        segfault_quarantine_threshold=segfault_quarantine_threshold,
        quarantine_seconds=quarantine_seconds,
        source_open_stagger_seconds=0.0,
    )
    return supervisor, clock, factory


def test_exit_139_restarts_only_failed_source_and_keeps_healthy_child(
    tmp_path: Path,
) -> None:
    supervisor, clock, factory = build_supervisor(
        tmp_path,
        source_ids=("rtsp://camera-1/live", "rtsp://camera-2/live"),
    )
    supervisor.poll_once()
    # Source opens are intentionally limited to one per scheduler poll.
    supervisor.poll_once()
    failed, healthy = factory.created
    failed.exit(139)

    supervisor.poll_once()
    state = supervisor.status()["sources"]
    assert state["rtsp://camera-1/live"]["state"] == "restarting"
    assert state["rtsp://camera-1/live"]["exit_code"] == 139
    assert state["rtsp://camera-2/live"]["running"] is True
    assert healthy.start_calls == 1

    clock.advance(2.0)
    supervisor.poll_once()
    assert len(factory.created) == 3
    assert factory.created[-1] is not healthy
    assert healthy.is_alive()


def test_restart_backoff_progresses_and_caps_at_last_delay(tmp_path: Path) -> None:
    supervisor, clock, factory = build_supervisor(tmp_path)
    supervisor.poll_once()

    for expected_delay in (2.0, 5.0, 10.0, 30.0, 30.0):
        factory.created[-1].exit(1)
        supervisor.poll_once()
        source = supervisor.status()["sources"]["rtsp://camera-1/live"]
        assert source["restart_in_seconds"] == expected_delay
        assert source["state"] == "restarting"

        clock.advance(expected_delay - 0.01)
        supervisor.poll_once()
        assert factory.created[-1].exitcode == 1
        clock.advance(0.01)
        supervisor.poll_once()
        assert factory.created[-1].exitcode is None


def test_repeated_exit_139_quarantines_source_then_allows_restart(
    tmp_path: Path,
) -> None:
    supervisor, clock, factory = build_supervisor(
        tmp_path,
        segfault_quarantine_threshold=3,
        quarantine_seconds=60.0,
    )
    supervisor.poll_once()

    for delay in (2.0, 5.0):
        factory.created[-1].exit(139)
        supervisor.poll_once()
        clock.advance(delay)
        supervisor.poll_once()

    factory.created[-1].exit(139)
    supervisor.poll_once()
    source = supervisor.status()["sources"]["rtsp://camera-1/live"]
    assert source["state"] == "quarantined"
    assert source["consecutive_exit_139"] == 3
    assert source["restart_in_seconds"] == 60.0

    clock.advance(59.99)
    supervisor.poll_once()
    assert len(factory.created) == 3
    clock.advance(0.01)
    supervisor.poll_once()
    assert len(factory.created) == 4


def test_negative_sigsegv_exit_is_quarantined_like_shell_139(
    tmp_path: Path,
) -> None:
    supervisor, clock, factory = build_supervisor(
        tmp_path,
        segfault_quarantine_threshold=2,
        quarantine_seconds=60.0,
    )
    supervisor.poll_once()
    factory.created[-1].exit(-11)
    supervisor.poll_once()
    clock.advance(2.0)
    supervisor.poll_once()
    factory.created[-1].exit(-11)
    supervisor.poll_once()

    source = supervisor.status()["sources"]["rtsp://camera-1/live"]
    assert source["state"] == "quarantined"
    assert source["exit_code"] == -11
    assert source["consecutive_exit_139"] == 2


def test_status_exposes_supervision_fields_without_rtsp_credentials(
    tmp_path: Path,
) -> None:
    source_id = "rtsp://admin:secret@camera-1:554/live"
    supervisor, _clock, factory = build_supervisor(
        tmp_path, source_ids=(source_id,)
    )
    supervisor.poll_once()
    factory.created[0].exit(139)
    supervisor.poll_once()

    status = supervisor.status()
    assert source_id not in status["sources"]
    source = status["sources"]["rtsp://***:***@camera-1:554/live"]
    assert {
        "pid",
        "state",
        "running",
        "failed",
        "exit_code",
        "restart_count",
        "last_error",
        "last_frame_time",
        "last_frame_age_seconds",
        "restart_in_seconds",
        "consecutive_exit_139",
    } <= source.keys()
    assert source["exit_code"] == 139
    assert source["restart_count"] == 1


def test_close_is_idempotent_and_stops_each_running_child(tmp_path: Path) -> None:
    supervisor, _clock, factory = build_supervisor(
        tmp_path,
        source_ids=("rtsp://camera-1/live", "rtsp://camera-2/live"),
    )
    supervisor.start()
    deadline = time.monotonic() + 1.0
    while len(factory.created) < 2 and time.monotonic() < deadline:
        time.sleep(0.005)

    assert len(factory.created) == 2
    supervisor.close()
    supervisor.close()

    assert all(not process.is_alive() for process in factory.created)
    assert all(process.terminate_calls == 1 for process in factory.created)
    assert all(process.kill_calls == 0 for process in factory.created)
    assert supervisor.status()["running"] is False
