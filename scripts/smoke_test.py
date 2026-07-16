from __future__ import annotations

import os
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["PROCESSOR_MODE"] = "mock"

from app.config import settings
from app.runtime import build_runtime


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        runtime = build_runtime(
            replace(
                settings,
                processor_mode="mock",
                camera_db_path=Path(directory) / "cameras.sqlite3",
                source_registry_path=Path(directory) / "sources.json",
                video_ingestion_enabled=False,
            )
        )
        runtime.start()
        try:
            frames = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(8)]
            source_ids = [f"camera-{index:02d}" for index in range(1, 9)]
            summary = runtime.router.submit_round(
                frames=frames,
                source_ids=source_ids,
                round_sequence=1,
            )
            deadline = time.time() + 2
            while time.time() < deadline and len(runtime.results.recent(limit=100)) < 10:
                time.sleep(0.02)
            results = runtime.results.recent(limit=100)
            assert summary["task_submissions"] == 10, summary
            assert len(results) == 10, len(results)
            print("SMOKE TEST PASSED")
            print(summary)
        finally:
            runtime.close()


if __name__ == "__main__":
    main()
