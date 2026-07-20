from __future__ import annotations

import argparse
import ast
import json
import shutil
from pathlib import Path
from typing import Any

from app.processors.ultralytics_loader import load_yolo_class


def _onnx_metadata(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    metadata: dict[str, Any] = {}
    for item in model.metadata_props:
        try:
            metadata[item.key] = ast.literal_eval(item.value)
        except (SyntaxError, ValueError):
            metadata[item.key] = item.value
    return metadata


def _write_ultralytics_engine(
    target: Path,
    serialized: bytes,
    metadata: dict[str, Any],
) -> None:
    metadata_bytes = json.dumps(metadata).encode("utf-8")
    target.write_bytes(
        len(metadata_bytes).to_bytes(4, byteorder="little")
        + metadata_bytes
        + serialized
    )


def _validate_engine(path: Path) -> None:
    import tensorrt as trt

    with path.open("rb") as engine_file:
        metadata_length = int.from_bytes(engine_file.read(4), byteorder="little")
        metadata_bytes = engine_file.read(metadata_length)
        try:
            json.loads(metadata_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            engine_file.seek(0)
        engine_bytes = engine_file.read()
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError(
            f"TensorRT {trt.__version__} could not deserialize exported engine: {path}"
        )
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(
            f"TensorRT {trt.__version__} could not create an execution context: {path}"
        )


def _build_dynamic_batch_engine(
    onnx_path: Path,
    target: Path,
    *,
    imgsz: int,
    max_batch: int,
    workspace: float,
    half: bool,
    ultralytics_metadata: bool = True,
) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parsing failed: " + " | ".join(errors))
    if network.num_inputs != 1:
        raise RuntimeError(f"Expected one model input, found {network.num_inputs}")
    model_input = network.get_input(0)
    model_input.shape = (-1, 3, imgsz, imgsz)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        model_input.name,
        (1, 3, imgsz, imgsz),
        (max_batch, 3, imgsz, imgsz),
        (max_batch, 3, imgsz, imgsz),
    )
    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(workspace * 1024**3),
    )
    if half:
        config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT dynamic-batch engine build failed")
    if ultralytics_metadata:
        _write_ultralytics_engine(
            target,
            bytes(serialized),
            _onnx_metadata(onnx_path),
        )
    else:
        target.write_bytes(bytes(serialized))
    _validate_engine(target)


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
    parser.add_argument("--dynamic-batch-only", action="store_true")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    target = Path(args.target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.dynamic_batch_only:
        if args.format != "engine":
            raise ValueError("--dynamic-batch-only requires --format engine")
        if source.suffix.lower() == ".onnx":
            onnx_path = source
            ultralytics_metadata = False
        else:
            YOLO = load_yolo_class()
            model = YOLO(str(source))
            onnx_path = Path(
                model.export(
                    format="onnx",
                    imgsz=args.imgsz,
                    batch=args.batch,
                    half=False,
                    dynamic=True,
                    device=args.device,
                )
            ).resolve()
            ultralytics_metadata = True
        _build_dynamic_batch_engine(
            onnx_path,
            target,
            imgsz=args.imgsz,
            max_batch=args.batch,
            workspace=args.workspace,
            half=args.half,
            ultralytics_metadata=ultralytics_metadata,
        )
        return
    YOLO = load_yolo_class()
    model = YOLO(str(source))
    kwargs: dict[str, Any] = {
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
    if args.format == "engine":
        _validate_engine(exported)
    if exported != target:
        if target.exists():
            target.unlink()
        shutil.move(str(exported), str(target))
    if not target.is_file():
        raise RuntimeError(f"Exporter did not create {target}")


if __name__ == "__main__":
    main()
