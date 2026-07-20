from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Sequence

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.core.types import FramePacket, TaskName, TaskResult
from app.processors.face_recognition import FaceRecognitionProcessor
from app.processors.fire_smoke import FireSmokeProcessor
from app.processors.plate import PlateRecognitionProcessor
from app.runtime import Runtime

router = APIRouter(
    prefix="/api/v1/tests",
    tags=["processor-tests"],
)


def get_runtime() -> Runtime:
    from app.main import runtime
    return runtime


def _decode_frames(files: list[UploadFile]) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for upload in files:
        content = upload.file.read()
        frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=415, detail=f"Could not decode {upload.filename}")
        frames.append(frame)
    return frames


def _build_packets(frames: list[np.ndarray], task: TaskName) -> list[FramePacket]:
    now = datetime.now(timezone.utc).isoformat()
    return [
        FramePacket(
            source_id=f"test-{index}",
            frame=frame,
            source_frame=frame.copy(),
            round_sequence=0,
            frame_index=index,
            captured_monotonic=time.monotonic(),
            captured_at_utc=now,
        )
        for index, frame in enumerate(frames)
    ]


def _face_processor(runtime: Runtime) -> FaceRecognitionProcessor:
    proc = runtime.face_processor
    if not isinstance(proc, FaceRecognitionProcessor):
        raise HTTPException(status_code=400, detail="Face processor is not available (mock mode?)")
    return proc


def _fire_processor(runtime: Runtime) -> FireSmokeProcessor:
    worker = runtime.router._workers.get(TaskName.FIRE_SMOKE)
    if worker is None:
        raise HTTPException(status_code=400, detail="Fire/smoke worker is not available")
    proc = worker.processor
    if not isinstance(proc, FireSmokeProcessor):
        raise HTTPException(status_code=400, detail="Fire/smoke processor is not available (mock mode?)")
    return proc


def _plate_processor(runtime: Runtime) -> PlateRecognitionProcessor:
    worker = runtime.router._workers.get(TaskName.PLATE_RECOGNITION)
    if worker is None:
        raise HTTPException(status_code=400, detail="Plate worker is not available")
    proc = worker.processor
    if not isinstance(proc, PlateRecognitionProcessor):
        raise HTTPException(status_code=400, detail="Plate processor is not available (mock mode?)")
    return proc


def _model_file_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "size_bytes": None}
    return {"path": str(path), "exists": True, "size_bytes": path.stat().st_size}


# ---------------------------------------------------------------------------
# Face Recognition
# ---------------------------------------------------------------------------

@router.get(
    "/face/models",
    summary="Check face recognition model files",
    description="Check that all face recognition model files exist and report TensorRT version.",
)
def face_models_status(runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    s = runtime.settings
    trt_version = "unknown"
    try:
        import tensorrt as trt
        trt_version = trt.__version__
    except ImportError:
        trt_version = "not_installed"
    return {
        "tensorrt_version": trt_version,
        "human_detector": _model_file_status(s.face_human_model_path),
        "face_detector": _model_file_status(s.face_detector_model_path),
        "embedding_model": _model_file_status(s.face_embedding_model_path),
        "settings": {
            "human_confidence": s.face_human_confidence,
            "face_confidence": s.face_detection_confidence,
            "recognition_threshold": s.face_recognition_threshold,
            "batch_size": s.face_batch_size,
            "human_imgsz": s.face_human_imgsz,
            "face_imgsz": s.face_detector_imgsz,
            "embedding_batch_size": s.face_embedding_batch_size,
            "device": s.face_device,
        },
    }


@router.post(
    "/face/human-detection",
    summary="Test human detection only",
    description="Upload one or more JPEG images to test human (YOLO-pose) detection. Returns raw detections per frame.",
)
async def test_face_human_detection(
    files: Annotated[list[UploadFile], File(description="JPEG images")],
    runtime: Runtime = Depends(get_runtime),
) -> list[dict[str, Any]]:
    frames = _decode_frames(files)
    proc = _face_processor(runtime)
    proc._ensure_dependencies()
    results = proc._predict_yolo(
        proc._human_detector,
        frames,
        model_path=proc.settings.human_model_path,
        imgsz=proc.settings.human_imgsz,
        confidence=proc.settings.human_confidence,
        fixed_batch=proc.settings.human_engine_fixed_batch,
    )
    output: list[dict[str, Any]] = []
    for i, result in enumerate(results):
        boxes = proc._boxes(result, proc.settings.human_confidence)
        output.append({
            "frame_index": i,
            "detections": len(boxes),
            "humans": [
                {
                    "bbox": b["bbox"],
                    "confidence": b["confidence"],
                }
                for b in boxes
            ],
        })
    return output


@router.post(
    "/face/full-pipeline",
    summary="Test full face recognition pipeline",
    description="Upload one or more JPEG images to run the full face recognition pipeline. Returns human detections, face detections, quality results, and recognition results.",
)
async def test_face_full_pipeline(
    files: Annotated[list[UploadFile], File(description="JPEG images")],
    runtime: Runtime = Depends(get_runtime),
) -> list[dict[str, Any]]:
    frames = _decode_frames(files)
    packets = _build_packets(frames, TaskName.FACE_RECOGNITION)
    proc = _face_processor(runtime)
    results = proc.process_batch(packets)
    return [
        {
            "source_id": r.source_id,
            "frame_index": r.frame_index,
            "processing_ms": r.processing_ms,
            "error": r.error,
            "data": r.data,
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# Fire / Smoke
# ---------------------------------------------------------------------------

@router.get(
    "/fire-smoke/models",
    summary="Check fire/smoke model files",
    description="Check that fire/smoke model files exist and report settings.",
)
def fire_smoke_models_status(runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    s = runtime.settings
    return {
        "model": _model_file_status(s.fire_model_path),
        "settings": {
            "confidence": s.fire_confidence,
            "smoke_confidence": s.smoke_confidence,
            "imgsz": s.fire_imgsz,
            "batch_size": s.fire_batch_size,
            "device": s.fire_device,
            "fire_class_id": s.fire_class_id,
            "smoke_class_id": s.smoke_class_id,
        },
    }


@router.post(
    "/fire-smoke/detection",
    summary="Test fire/smoke detection",
    description="Upload one or more JPEG images to test fire and smoke detection. Returns raw detections and severity analysis.",
)
async def test_fire_smoke_detection(
    files: Annotated[list[UploadFile], File(description="JPEG images")],
    runtime: Runtime = Depends(get_runtime),
) -> list[dict[str, Any]]:
    frames = _decode_frames(files)
    packets = _build_packets(frames, TaskName.FIRE_SMOKE)
    proc = _fire_processor(runtime)
    results = proc.process_batch(packets)
    return [
        {
            "source_id": r.source_id,
            "frame_index": r.frame_index,
            "processing_ms": r.processing_ms,
            "error": r.error,
            "data": r.data,
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# Plate Recognition
# ---------------------------------------------------------------------------

@router.get(
    "/plate/models",
    summary="Check plate recognition model files",
    description="Check that vehicle detector, plate detector, and OCR model files exist.",
)
def plate_models_status(runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    s = runtime.settings
    return {
        "vehicle_detector": _model_file_status(s.vehicle_detector_weights),
        "plate_detector": _model_file_status(s.plate_detector_weights),
        "recognizer_dir": {
            "path": str(s.plate_recognizer_dir),
            "exists": s.plate_recognizer_dir.is_dir(),
        },
        "settings": {
            "vehicle_confidence": s.vehicle_confidence,
            "plate_confidence": s.plate_confidence,
            "ocr_confidence": s.plate_ocr_confidence,
            "vehicle_imgsz": s.vehicle_imgsz,
            "plate_imgsz": s.plate_imgsz,
            "device": s.plate_device,
            "vehicle_class_ids": list(s.vehicle_class_ids),
            "plate_class_ids": list(s.plate_class_ids),
            "use_fp16": s.plate_use_fp16,
        },
    }


@router.post(
    "/plate/vehicle-detection",
    summary="Test vehicle detection only",
    description="Upload one or more JPEG images to test vehicle detection. Returns raw vehicle detections.",
)
async def test_plate_vehicle_detection(
    files: Annotated[list[UploadFile], File(description="JPEG images")],
    runtime: Runtime = Depends(get_runtime),
) -> list[dict[str, Any]]:
    import numpy as np
    frames = _decode_frames(files)
    proc = _plate_processor(runtime)
    proc._sync_model_selection()
    proc._ensure_yolo("vehicle")
    results, elapsed_ms = proc._predict_yolo(
        model=proc._vehicle_detector,
        frames=frames,
        imgsz=proc.settings.vehicle_imgsz,
        confidence=proc.settings.vehicle_confidence,
        iou=proc.settings.vehicle_iou,
    )
    output: list[dict[str, Any]] = []
    for i, result in enumerate(results):
        boxes = getattr(result, "boxes", None)
        detections: list[dict[str, Any]] = []
        if boxes is not None:
            xyxy = boxes.xyxy[0].tolist() if hasattr(boxes.xyxy, "tolist") else boxes.xyxy.cpu().numpy().tolist()
            confs = boxes.conf.tolist() if hasattr(boxes.conf, "tolist") else boxes.conf.cpu().numpy().tolist()
            classes = boxes.cls.tolist() if hasattr(boxes.cls, "tolist") else boxes.cls.cpu().numpy().tolist()
            for j in range(len(xyxy)):
                detections.append({
                    "bbox": [round(float(v), 2) for v in xyxy[j]],
                    "confidence": round(float(confs[j]), 6),
                    "class_id": int(classes[j]),
                })
        output.append({
            "frame_index": i,
            "detections": len(detections),
            "vehicles": detections,
        })
    return output


@router.post(
    "/plate/full-pipeline",
    summary="Test full plate recognition pipeline",
    description="Upload one or more JPEG images to run the full plate recognition pipeline (vehicle detection -> plate detection -> OCR).",
)
async def test_plate_full_pipeline(
    files: Annotated[list[UploadFile], File(description="JPEG images")],
    runtime: Runtime = Depends(get_runtime),
) -> list[dict[str, Any]]:
    frames = _decode_frames(files)
    packets = _build_packets(frames, TaskName.PLATE_RECOGNITION)
    proc = _plate_processor(runtime)
    results = proc.process_batch(packets)
    return [
        {
            "source_id": r.source_id,
            "frame_index": r.frame_index,
            "processing_ms": r.processing_ms,
            "error": r.error,
            "data": r.data,
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# All-in-one
# ---------------------------------------------------------------------------

@router.get(
    "/all",
    summary="Run all model checks at once",
    description="Check all model files and TensorRT/cuDA availability for every pipeline in a single call.",
)
def all_models_status(runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    return {
        "face": face_models_status(runtime),
        "fire_smoke": fire_smoke_models_status(runtime),
        "plate": plate_models_status(runtime),
    }
