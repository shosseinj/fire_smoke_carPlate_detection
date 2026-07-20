from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EngineBuild:
    name: str
    source: Path
    target: Path
    imgsz: int
    batch: int | None = None


def _build(
    job: EngineBuild,
    *,
    batch: int,
    workspace: float,
    device: str,
) -> None:
    if not job.source.is_file():
        raise FileNotFoundError(job.source)
    job.target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "app.model_export_worker",
        "--source",
        str(job.source),
        "--target",
        str(job.target),
        "--format",
        "engine",
        "--imgsz",
        str(job.imgsz),
        "--batch",
        str(job.batch or batch),
        "--workspace",
        str(workspace),
        "--device",
        device,
        "--half",
        "--dynamic-batch-only",
    ]
    print(f"Building {job.name}: {job.target}", flush=True)
    subprocess.run(command, check=True)
    print(f"Created and validated {job.name}: {job.target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build all dynamic-batch TensorRT engines on the current device"
    )
    parser.add_argument("--root", type=Path, default=Path("/workspace/weights"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workspace", type=float, default=4.0)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    import tensorrt as trt

    if not trt.__version__.startswith("10.7."):
        raise RuntimeError(
            f"Expected TensorRT 10.7.x for trt107 filenames, found {trt.__version__}"
        )
    if not 1 <= args.batch <= 8:
        raise ValueError("--batch must be between 1 and 8")

    root = args.root.resolve()
    suffix = f"dynamic_b{args.batch}_trt107.engine"
    jobs = (
        EngineBuild(
            name="fire_smoke",
            source=root / "fire_smoke/best_nano_111.pt",
            target=root / f"fire_smoke/linux_trt10/best_nano_111_{suffix}",
            imgsz=640,
        ),
        EngineBuild(
            name="vehicle_detector",
            source=root / "vehicle_detector/yolo11n.pt",
            target=root / f"vehicle_detector/yolo11n_{suffix}",
            imgsz=640,
        ),
        EngineBuild(
            name="plate_detector",
            source=root / "plate_detector/model.pt",
            target=root / f"plate_detector/model_{suffix}",
            imgsz=640,
        ),
        EngineBuild(
            name="face_human_pose",
            source=root / "face_recognition/linux_trt10/yolo26s-pose_batch8_linux.pt",
            target=root / f"face_recognition/linux_trt10/yolo26s-pose_{suffix}",
            imgsz=640,
        ),
        EngineBuild(
            name="face_detector",
            source=root / "face_recognition/linux_trt10/yolov8n-face_batch8_linux.pt",
            target=root / f"face_recognition/linux_trt10/yolov8n-face_{suffix}",
            imgsz=640,
        ),
        EngineBuild(
            name="face_embedding",
            source=root / "face_recognition/linux_trt10/arcface_fp16.onnx",
            target=root / "face_recognition/linux_trt10/arcface_dynamic_b64_trt107.engine",
            imgsz=112,
            batch=64,
        ),
    )
    for job in jobs:
        _build(
            job,
            batch=args.batch,
            workspace=args.workspace,
            device=args.device,
        )


if __name__ == "__main__":
    main()
