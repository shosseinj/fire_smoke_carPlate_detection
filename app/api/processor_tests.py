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
from app.core.personnel_store import PersonnelStore
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
# Personnel — smoke test (works in mock mode, tests store layer)
# ---------------------------------------------------------------------------


# A valid Iranian national code for testing: 1234567891 (checksum = 1)
_TEST_NATIONAL_CODE = "1234567891"
_TEST_NATIONAL_CODE_2 = "9876543210"


def _personnel_store(runtime: Runtime) -> PersonnelStore:
    return runtime.personnel_store


@router.post(
    "/personnel/smoke",
    summary="Run a personnel module smoke test",
    description=(
        "Creates, reads, updates, searches, and deletes a personnel record with an image. "
        "Works in both mock and real processor modes because it tests the store layer directly. "
        "Returns pass/fail per step."
    ),
)
async def personnel_smoke_test(
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    store = _personnel_store(runtime)
    steps: dict[str, Any] = {}
    test_id = f"smoke-{int(time.time() * 1000)}"

    try:
        # 1. Create personnel
        person = store.create(
            fname="Test",
            lname=f"User{test_id}",
            national_code=_TEST_NATIONAL_CODE,
            employee_type="employee",
            degree="Smoke Test",
        )
        steps["create"] = {
            "status": "PASS",
            "id": person.id,
            "national_code": person.national_code,
        }
        person_id = person.id

        # 2. Get by ID
        fetched = store.get(person_id)
        if fetched is None or fetched.id != person_id:
            steps["get_by_id"] = {"status": "FAIL", "detail": "Record not found after creation"}
        else:
            steps["get_by_id"] = {"status": "PASS", "fname": fetched.fname}

        # 3. Search by national code
        searched = store.get_by_national_code(_TEST_NATIONAL_CODE)
        if searched is None or searched.id != person_id:
            steps["search"] = {"status": "FAIL", "detail": "Search by national code failed"}
        else:
            steps["search"] = {"status": "PASS", "found": True}

        # 4. Update
        updated = store.update(person_id, fname="Updated")
        if updated is None or updated.fname != "Updated":
            steps["update"] = {"status": "FAIL", "detail": "Update failed"}
        else:
            steps["update"] = {"status": "PASS", "fname": updated.fname}

        # 5. List (should include our test record)
        records, total = store.list(limit=100)
        if total < 1:
            steps["list"] = {"status": "FAIL", "detail": "List returned zero records"}
        else:
            steps["list"] = {"status": "PASS", "total": total}

        # 6. Create image record
        storage_key = f"personnel_snapshots/test-{test_id}.jpg"
        img = store.create_image(
            personnel_id=person_id,
            storage_key=storage_key,
            description="Smoke test image",
        )
        if img is None or img.personnel_id != person_id:
            steps["create_image"] = {"status": "FAIL", "detail": "Image creation failed"}
        else:
            steps["create_image"] = {
                "status": "PASS",
                "image_id": img.id,
                "is_primary": img.is_primary,
            }
        image_id = img.id

        # 7. List images
        images = store.list_images(person_id)
        if len(images) < 1:
            steps["list_images"] = {"status": "FAIL", "detail": "No images returned"}
        else:
            steps["list_images"] = {"status": "PASS", "count": len(images)}

        # 8. Set primary image
        primary = store.set_primary_image(image_id)
        if primary is None or not primary.is_primary:
            steps["set_primary"] = {"status": "FAIL", "detail": "Set primary failed"}
        else:
            steps["set_primary"] = {"status": "PASS", "is_primary": primary.is_primary}

        # 9. List with-images
        with_images, wi_total = store.list_with_images(limit=100)
        if wi_total < 1:
            steps["list_with_images"] = {"status": "FAIL", "detail": "with-images returned zero"}
        else:
            steps["list_with_images"] = {"status": "PASS", "total": wi_total}

        # 10. Delete image
        deleted_img = store.delete_image(image_id)
        if not deleted_img:
            steps["delete_image"] = {"status": "FAIL", "detail": "Delete image returned False"}
        else:
            steps["delete_image"] = {"status": "PASS"}

        # 11. Delete personnel
        deleted = store.delete(person_id)
        if not deleted:
            steps["delete_personnel"] = {"status": "FAIL", "detail": "Delete returned False"}
        else:
            steps["delete_personnel"] = {"status": "PASS"}

        # 12. Import template (generates Excel bytes)
        template = store.generate_import_template()
        if not template or len(template) < 100:
            steps["import_template"] = {"status": "FAIL", "detail": "Template too small or empty"}
        else:
            steps["import_template"] = {"status": "PASS", "bytes": len(template)}

        # Summary
        failed = {k: v for k, v in steps.items() if v.get("status") == "FAIL"}
        steps["_summary"] = {
            "total": len(steps),
            "passed": len(steps) - len(failed),
            "failed": len(failed),
        }
        if failed:
            steps["_summary"]["failed_steps"] = list(failed.keys())

    except Exception as exc:
        steps["_unexpected_error"] = f"{type(exc).__name__}: {exc}"
        steps["_summary"] = {"total": len(steps), "passed": 0, "failed": 1}

    return steps


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
        "personnel": {
            "store_ready": True,
            "count": runtime.personnel_store.count(),
        },
    }
