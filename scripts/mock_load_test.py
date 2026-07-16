from __future__ import annotations

import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.runtime import build_runtime
from app.core.types import TaskName


def main(rounds: int = 100, sources: int = 50) -> None:
    with tempfile.TemporaryDirectory() as directory:
        runtime = build_runtime(
            replace(
                settings,
                processor_mode="mock",
                camera_db_path=Path(directory) / "cameras.sqlite3",
                source_registry_path=Path(directory) / "sources.json",
            )
        )
        for index in range(1, sources + 1):
            runtime.registry.update(
                f"camera-{index:02d}",
                enabled=True,
                tasks={TaskName.FIRE_SMOKE, TaskName.PLATE_RECOGNITION},
            )
        runtime.start()
        started = time.perf_counter()
        try:
            source_ids = [f"camera-{index:02d}" for index in range(1, sources + 1)]
            frames = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in source_ids]
            for sequence in range(1, rounds + 1):
                runtime.router.submit_round(
                    frames=frames,
                    source_ids=source_ids,
                    round_sequence=sequence,
                )
            time.sleep(0.5)
            elapsed = time.perf_counter() - started
            print(f"Submitted {rounds * sources} source frames in {elapsed:.3f}s")
            print(runtime.router.status())
        finally:
            runtime.close()


if __name__ == "__main__":
    main()
