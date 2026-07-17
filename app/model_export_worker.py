from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from app.processors.ultralytics_loader import load_yolo_class


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated Ultralytics model export")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--format", choices=("engine", "onnx"), required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--workspace", type=float, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    target = Path(args.target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    YOLO = load_yolo_class()
    model = YOLO(str(source))
    kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "half": args.half,
        "dynamic": args.dynamic,
        "device": args.device,
    }
    if args.format == "engine":
        kwargs["workspace"] = args.workspace
    exported = Path(model.export(**kwargs)).resolve()
    if exported != target:
        if target.exists():
            target.unlink()
        shutil.move(str(exported), str(target))
    if not target.is_file():
        raise RuntimeError(f"Exporter did not create {target}")


if __name__ == "__main__":
    main()
