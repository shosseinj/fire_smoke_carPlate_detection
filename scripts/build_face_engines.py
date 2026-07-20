from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _build(source: Path, target: Path, *, imgsz: int, batch: int, workspace: float, device: str) -> None:
    command = [
        sys.executable,
        "-m",
        "app.model_export_worker",
        "--source",
        str(source),
        "--target",
        str(target),
        "--format",
        "engine",
        "--imgsz",
        str(imgsz),
        "--batch",
        str(batch),
        "--workspace",
        str(workspace),
        "--device",
        device,
        "--half",
        "--dynamic-batch-only",
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dynamic TensorRT engines for the face pipeline")
    parser.add_argument("--root", type=Path, default=Path("weights/face_recognition/linux_trt10"))
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workspace", type=float, default=4.0)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    root = args.root.resolve()
    jobs = (
        (
            root / "yolo26s-pose_batch8_linux.pt",
            root / f"yolo26s-pose_dynamic_b{args.batch}_trt107.engine",
            640,
        ),
        (
            root / "yolov8n-face_batch8_linux.pt",
            root / f"yolov8n-face_dynamic_b{args.batch}_trt107.engine",
            640,
        ),
    )
    for source, target, imgsz in jobs:
        if not source.is_file():
            raise FileNotFoundError(source)
        _build(
            source,
            target,
            imgsz=imgsz,
            batch=args.batch,
            workspace=args.workspace,
            device=args.device,
        )
        print(f"Created and validated {target}")


if __name__ == "__main__":
    main()
