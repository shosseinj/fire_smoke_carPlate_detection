from __future__ import annotations

import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.types import FramePacket, TaskName, TaskResult
from app.core.frontend_messages import LocalizedJSONRoute
from app.core.holiday_store import HolidayStore
from app.core.location_store import LocationStore
from app.core.personnel_store import PersonnelStore
from app.core.request_store import RequestStore
from app.core.shift_store import ShiftStore
from app.processors.face_recognition import FaceRecognitionProcessor
from app.processors.fire_smoke import FireSmokeProcessor
from app.processors.plate import PlateRecognitionProcessor
from app.runtime import Runtime

router = APIRouter(
    prefix="/api/v1/tests",
    tags=["processor-tests"],
    route_class=LocalizedJSONRoute,
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
            raise HTTPException(status_code=415, detail=f"رمزگشایی فایل {upload.filename} امکان‌پذیر نیست")
        frames.append(frame)
    return frames


def _build_packets(frames: list[np.ndarray], task: TaskName) -> list[FramePacket]:
    now = datetime.now(timezone.utc).isoformat()
    return [
        FramePacket(
            source_id=f"test-{index}",
            frame=frame,
            metadata={"source_frame": frame.copy()},
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
        raise HTTPException(status_code=400, detail="پردازشگر چهره در دسترس نیست (حالت mock?)")
    return proc


def _fire_processor(runtime: Runtime) -> FireSmokeProcessor:
    worker = runtime.router.workers.get(TaskName.FIRE_SMOKE)
    if worker is None:
        raise HTTPException(status_code=400, detail="کارگر تشخیص آتش/دود در دسترس نیست")
    proc = worker.processor
    if not isinstance(proc, FireSmokeProcessor):
        raise HTTPException(status_code=400, detail="پردازشگر آتش/دود در دسترس نیست (حالت mock?)")
    return proc


def _plate_processor(runtime: Runtime) -> PlateRecognitionProcessor:
    worker = runtime.router.workers.get(TaskName.PLATE_RECOGNITION)
    if worker is None:
        raise HTTPException(status_code=400, detail="کارگر تشخیص پلاک در دسترس نیست")
    proc = worker.processor
    if not isinstance(proc, PlateRecognitionProcessor):
        raise HTTPException(status_code=400, detail="پردازشگر پلاک در دسترس نیست (حالت mock?)")
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
    active_quality = runtime.face_quality_settings.as_dict()
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
            "human_pose_enabled": active_quality["human_pose_enabled"],
            "human_pose_min_keypoints": active_quality["human_pose_min_keypoints"],
            "human_pose_keypoint_confidence": active_quality["human_pose_keypoint_confidence"],
            "recognition_quality_weight": active_quality["recognition_quality_weight"],
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
        boxes, rejected = proc._human_boxes(result)
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
            "filtered": rejected,
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
    # Get class IDs from the processor if available
    fire_class_id: int = 0
    smoke_class_id: int = 1
    worker = runtime.router.workers.get(TaskName.FIRE_SMOKE)
    if worker is not None:
        proc = worker.processor
        if hasattr(proc, "settings"):
            proc_settings = proc.settings
            fire_class_id = getattr(proc_settings, "fire_class_id", 0)
            smoke_class_id = getattr(proc_settings, "smoke_class_id", 1)
    return {
        "model": _model_file_status(s.fire_model_path),
        "settings": {
            "confidence": s.fire_confidence,
            "smoke_confidence": s.smoke_confidence,
            "imgsz": s.fire_imgsz,
            "batch_size": s.fire_batch_size,
            "device": s.fire_device,
            "fire_class_id": fire_class_id,
            "smoke_class_id": smoke_class_id,
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
# Locations — smoke test (works in mock mode)
# ---------------------------------------------------------------------------


def _location_store(runtime: Runtime) -> LocationStore:
    return runtime.location_store


@router.post(
    "/locations/smoke",
    summary="Run a locations module smoke test",
    description=(
        "Creates, reads, updates, and deletes a building, section, and room. "
        "Tests polygon validation, personnel room access (grant/revoke/check), "
        "and camera assignment. Works in mock mode."
    ),
)
async def locations_smoke_test(
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    store = _location_store(runtime)
    steps: dict[str, Any] = {}
    ts = int(time.time() * 1000)

    try:
        # 1. Create building
        bld = store.create_building(
            name=f"Smoke Building {ts}",
            address=f"123 Test St, Unit {ts}",
            description="Smoke test building",
        )
        steps["create_building"] = {"status": "PASS", "id": bld.id}
        building_id = bld.id

        # 2. Get building
        fetched = store.get_building(building_id)
        if fetched is None or fetched.id != building_id:
            steps["get_building"] = {"status": "FAIL", "detail": "Building not found after creation"}
        else:
            steps["get_building"] = {"status": "PASS"}

        # 3. Update building
        updated = store.update_building(building_id, name="Updated Building")
        if updated is None or updated.name != "Updated Building":
            steps["update_building"] = {"status": "FAIL", "detail": "Update failed"}
        else:
            steps["update_building"] = {"status": "PASS"}

        # 4. List buildings
        blds, total = store.list_buildings(limit=100)
        if total < 1:
            steps["list_buildings"] = {"status": "FAIL", "detail": "No buildings listed"}
        else:
            steps["list_buildings"] = {"status": "PASS", "total": total}

        # 5. Create section
        sec = store.create_section(
            name=f"Section {ts}",
            building_id=building_id,
            description="Smoke test section",
        )
        steps["create_section"] = {"status": "PASS", "id": sec.id}
        section_id = sec.id

        # 6. Get section
        fetched_sec = store.get_section(section_id)
        if fetched_sec is None or fetched_sec.id != section_id:
            steps["get_section"] = {"status": "FAIL", "detail": "Section not found"}
        else:
            steps["get_section"] = {"status": "PASS"}

        # 7. List sections by building
        secs, sec_total = store.list_sections(building_id=building_id)
        if sec_total < 1:
            steps["list_sections_by_building"] = {"status": "FAIL", "detail": "No sections by building"}
        else:
            steps["list_sections_by_building"] = {"status": "PASS", "total": sec_total}

        # 8. Create room with valid polygon (a triangle)
        polygon = "[[0,0],[100,0],[50,100]]"
        room = store.create_room(
            name=f"Room {ts}",
            section_id=section_id,
            description="Smoke test room",
            polygon_json=polygon,
        )
        steps["create_room"] = {"status": "PASS", "id": room.id}
        room_id = room.id

        # 9. Get room
        fetched_room = store.get_room(room_id)
        if fetched_room is None or fetched_room.id != room_id:
            steps["get_room"] = {"status": "FAIL", "detail": "Room not found"}
        else:
            steps["get_room"] = {"status": "PASS"}

        # 10. Update room polygon
        new_polygon = "[[0,0],[200,0],[200,200],[0,200]]"
        updated_room = store.update_room(room_id, polygon_json=new_polygon)
        if updated_room is None or updated_room.polygon_json != new_polygon:
            steps["update_room_polygon"] = {"status": "FAIL", "detail": "Polygon update failed"}
        else:
            steps["update_room_polygon"] = {"status": "PASS"}

        # 11. List rooms by section
        rooms, room_total = store.list_rooms(section_id=section_id)
        if room_total < 1:
            steps["list_rooms_by_section"] = {"status": "FAIL", "detail": "No rooms by section"}
        else:
            steps["list_rooms_by_section"] = {"status": "PASS", "total": room_total}

        # 12. Grant room access
        access = store.grant_room_access(
            personnel_id=1, room_id=room_id, granted_by="smoke_test"
        )
        if access is None or access.room_id != room_id:
            steps["grant_access"] = {"status": "FAIL", "detail": "Grant access failed"}
        else:
            steps["grant_access"] = {"status": "PASS", "access_id": access.id}

        # 13. Check room access
        has_access = store.check_room_access(personnel_id=1, room_id=room_id)
        if not has_access:
            steps["check_access"] = {"status": "FAIL", "detail": "Access check returned False after grant"}
        else:
            steps["check_access"] = {"status": "PASS"}

        # 14. List personnel rooms
        personnel_rooms = store.list_personnel_rooms(personnel_id=1)
        if len(personnel_rooms) < 1:
            steps["list_personnel_rooms"] = {"status": "FAIL", "detail": "No rooms for personnel"}
        else:
            steps["list_personnel_rooms"] = {"status": "PASS", "count": len(personnel_rooms)}

        # 15. List room personnel
        room_personnel = store.list_room_personnel(room_id)
        if len(room_personnel) < 1:
            steps["list_room_personnel"] = {"status": "FAIL", "detail": "No personnel for room"}
        else:
            steps["list_room_personnel"] = {"status": "PASS", "count": len(room_personnel)}

        # 16. Revoke room access
        revoked = store.revoke_room_access(personnel_id=1, room_id=room_id)
        if not revoked:
            steps["revoke_access"] = {"status": "FAIL", "detail": "Revoke returned False"}
        else:
            steps["revoke_access"] = {"status": "PASS"}

        # 17. Test polygon matching (point inside the polygon)
        # The room has polygon [[0,0],[200,0],[200,200],[0,200]]
        # Point (50, 50) should be inside
        from app.core.location_store import point_in_polygon, parse_polygon
        points = parse_polygon(new_polygon)
        inside = point_in_polygon(50.0, 50.0, points)
        if not inside:
            steps["polygon_inside"] = {"status": "FAIL", "detail": "Point should be inside polygon"}
        else:
            steps["polygon_inside"] = {"status": "PASS"}

        # Point (300, 300) should be outside
        outside = point_in_polygon(300.0, 300.0, points)
        if outside:
            steps["polygon_outside"] = {"status": "FAIL", "detail": "Point should be outside polygon"}
        else:
            steps["polygon_outside"] = {"status": "PASS"}

        # 18. Test the camera-assigned room matching path
        matches = store.match_detection_to_room(
            room_id=room_id,
            detection_type="face_recognition",
            detection_event_id=0,
            bbox_center_x=50.0,
            bbox_center_y=50.0,
            personnel_id=1,
            camera_id="smoke-cam-test",
        )
        if len(matches) < 1:
            steps["polygon_matching"] = {"status": "FAIL", "detail": "No matches found for point inside polygon"}
        else:
            steps["polygon_matching"] = {"status": "PASS", "matches": len(matches)}

        # 19. Delete room
        deleted_room = store.delete_room(room_id)
        if not deleted_room:
            steps["delete_room"] = {"status": "FAIL", "detail": "Delete returned False"}
        else:
            steps["delete_room"] = {"status": "PASS"}

        # 20. Delete section
        deleted_sec = store.delete_section(section_id)
        if not deleted_sec:
            steps["delete_section"] = {"status": "FAIL", "detail": "Delete returned False"}
        else:
            steps["delete_section"] = {"status": "PASS"}

        # 21. Delete building
        deleted_bld = store.delete_building(building_id)
        if not deleted_bld:
            steps["delete_building"] = {"status": "FAIL", "detail": "Delete returned False"}
        else:
            steps["delete_building"] = {"status": "PASS"}

        # Summary
        failed = {k: v for k, v in steps.items() if isinstance(v, dict) and v.get("status") == "FAIL"}
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
# Shifts — smoke test
# ---------------------------------------------------------------------------


@router.post(
    "/shifts/smoke",
    summary="Run a shifts module smoke test",
    description="Creates, reads, updates, and deletes a shift. Tests personnel assignment. Works in mock mode.",
)
async def shifts_smoke_test(
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    store: ShiftStore = runtime.shift_store
    steps: dict[str, Any] = {}
    try:
        # 1. Create shift
        shift = store.create(
            shift_name="Smoke Shift",
            shift_type="morning",
            start_time="08:00",
            end_time="16:00",
            works_saturday=True,
            works_sunday=False,
            works_monday=True,
            works_tuesday=True,
            works_wednesday=True,
            works_thursday=True,
            works_friday=False,
        )
        steps["create"] = {"status": "PASS", "id": shift.id}
        shift_id = shift.id

        # 2. Get by ID
        fetched = store.get(shift_id)
        if fetched is None or fetched.id != shift_id:
            steps["get"] = {"status": "FAIL", "detail": "Record not found after creation"}
        else:
            steps["get"] = {"status": "PASS", "name": fetched.shift_name}

        # 3. Update
        updated = store.update(shift_id, shift_name="Updated Shift")
        if updated is None or updated.shift_name != "Updated Shift":
            steps["update"] = {"status": "FAIL", "detail": "Update failed"}
        else:
            steps["update"] = {"status": "PASS"}

        # 4. List
        records, total = store.list(limit=100)
        if total < 1:
            steps["list"] = {"status": "FAIL", "detail": "List returned zero"}
        else:
            steps["list"] = {"status": "PASS", "total": total}

        # 5. Statistics
        stats = store.statistics()
        steps["statistics"] = {"status": "PASS", "total_shifts": stats["total_shifts"]}

        # 6. Delete
        deleted = store.delete(shift_id)
        if not deleted:
            steps["delete"] = {"status": "FAIL", "detail": "Delete returned False"}
        else:
            steps["delete"] = {"status": "PASS"}

        failed = {k: v for k, v in steps.items() if isinstance(v, dict) and v.get("status") == "FAIL"}
        steps["_summary"] = {"total": len(steps), "passed": len(steps) - len(failed), "failed": len(failed)}
    except Exception as exc:
        steps["_unexpected_error"] = f"{type(exc).__name__}: {exc}"
        steps["_summary"] = {"total": len(steps), "passed": 0, "failed": 1}
    return steps


# ---------------------------------------------------------------------------
# Holidays — smoke test
# ---------------------------------------------------------------------------


@router.post(
    "/holidays/smoke",
    summary="Run a holidays module smoke test",
    description="Creates, reads, updates, deletes, and checks a holiday. Works in mock mode.",
)
async def holidays_smoke_test(
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    store: HolidayStore = runtime.holiday_store
    steps: dict[str, Any] = {}
    try:
        # 1. Create holiday
        holiday = store.create(
            name="Smoke Holiday",
            date_value="2026-12-25",
            description="Smoke test holiday",
            holiday_type="national",
            every_year=False,
        )
        steps["create"] = {"status": "PASS", "id": holiday.id}
        holiday_id = holiday.id

        # 2. Get by ID
        fetched = store.get(holiday_id)
        if fetched is None or fetched.id != holiday_id:
            steps["get"] = {"status": "FAIL", "detail": "Record not found after creation"}
        else:
            steps["get"] = {"status": "PASS", "name": fetched.name}

        # 3. Update
        updated = store.update(holiday_id, name="Updated Holiday")
        if updated is None or updated.name != "Updated Holiday":
            steps["update"] = {"status": "FAIL", "detail": "Update failed"}
        else:
            steps["update"] = {"status": "PASS"}

        # 4. Check is_holiday
        is_h = store.is_holiday(date(2026, 12, 25))
        if not is_h:
            steps["is_holiday"] = {"status": "FAIL", "detail": "is_holiday returned False"}
        else:
            steps["is_holiday"] = {"status": "PASS"}

        # 5. List
        records, total = store.list(limit=100)
        if total < 1:
            steps["list"] = {"status": "FAIL", "detail": "List returned zero"}
        else:
            steps["list"] = {"status": "PASS", "total": total}

        # 6. Soft-delete (deactivate)
        deleted = store.delete(holiday_id)
        if not deleted:
            steps["delete"] = {"status": "FAIL", "detail": "Delete returned False"}
        else:
            steps["delete"] = {"status": "PASS"}

        # 7. Hard-delete for cleanup
        store.hard_delete(holiday_id)

        failed = {k: v for k, v in steps.items() if isinstance(v, dict) and v.get("status") == "FAIL"}
        steps["_summary"] = {"total": len(steps), "passed": len(steps) - len(failed), "failed": len(failed)}
    except Exception as exc:
        steps["_unexpected_error"] = f"{type(exc).__name__}: {exc}"
        steps["_summary"] = {"total": len(steps), "passed": 0, "failed": 1}
    return steps


# ---------------------------------------------------------------------------
# Requests — smoke test
# ---------------------------------------------------------------------------


@router.post(
    "/requests/smoke",
    summary="Run a requests module smoke test",
    description="Creates, reads, approves, and deletes a personnel request. Works in mock mode.",
)
async def requests_smoke_test(
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    store: RequestStore = runtime.request_store
    personnel_store: PersonnelStore = runtime.personnel_store
    steps: dict[str, Any] = {}
    test_id = f"req-{int(time.time() * 1000)}"
    test_nat_code = _TEST_NATIONAL_CODE_2  # use second test code
    person = None
    try:
        # Create personnel for test
        person = personnel_store.create(
            fname="Req",
            lname=f"Test{test_id}",
            national_code=test_nat_code,
            employee_type="employee",
        )
        personnel_id = person.id

        # 1. Create request
        req = store.create(
            personnel_id=personnel_id,
            request_type="leave",
            start_date="2026-08-01",
            end_date="2026-08-03",
            reason="Smoke test leave",
        )
        steps["create_request"] = {"status": "PASS", "id": req.id}
        req_id = req.id

        # 2. Get by ID
        fetched = store.get(req_id)
        if fetched is None or fetched.id != req_id:
            steps["get"] = {"status": "FAIL", "detail": "Request not found"}
        else:
            steps["get"] = {"status": "PASS", "status_val": fetched.status}

        # 3. Approve
        approved = store.approve(req_id, approved_by=1)
        if approved is None or approved.status != "approved":
            steps["approve"] = {"status": "FAIL", "detail": "Approve failed"}
        else:
            steps["approve"] = {"status": "PASS"}

        # 4. List
        records, total = store.list(limit=100)
        if total < 1:
            steps["list"] = {"status": "FAIL", "detail": "List returned zero"}
        else:
            steps["list"] = {"status": "PASS", "total": total}

        # 5. Delete
        deleted = store.delete(req_id)
        if not deleted:
            steps["delete"] = {"status": "FAIL", "detail": "Delete returned False"}
        else:
            steps["delete"] = {"status": "PASS"}

        # Cleanup personnel
        personnel_store.delete(personnel_id)

        failed = {k: v for k, v in steps.items() if isinstance(v, dict) and v.get("status") == "FAIL"}
        steps["_summary"] = {"total": len(steps), "passed": len(steps) - len(failed), "failed": len(failed)}
    except Exception as exc:
        if person is not None:
            try:
                personnel_store.delete(person.id)
            except Exception:
                pass
        steps["_unexpected_error"] = f"{type(exc).__name__}: {exc}"
        steps["_summary"] = {"total": len(steps), "passed": 0, "failed": 1}
    return steps


# ---------------------------------------------------------------------------
# All-in-one — comprehensive test runner
# ---------------------------------------------------------------------------

_TEST_PASS = "PASS"
_TEST_FAIL = "FAIL"
_TEST_WARN = "WARN"
_TEST_SKIP = "SKIP"


def _test(name: str, status: str, detail: str = "") -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail}


def _run_store_tests(store: Any, label: str) -> list[dict[str, Any]]:
    """Read-only store validation: count + list."""
    tests: list[dict[str, Any]] = []
    # Test count
    try:
        count = store.count()
        assert isinstance(count, int) and count >= 0
        tests.append(_test(f"{label}_count", _TEST_PASS, f"count={count}"))
    except Exception as exc:
        tests.append(_test(f"{label}_count", _TEST_FAIL, str(exc)))
    # Test list
    try:
        records, total = store.list(limit=5)
        assert isinstance(total, int) and total >= 0
        tests.append(_test(f"{label}_list", _TEST_PASS, f"total={total}"))
    except Exception as exc:
        tests.append(_test(f"{label}_list", _TEST_FAIL, str(exc)))
    return tests


def _run_model_tests(runtime: Runtime, model_paths: dict[str, Path]) -> list[dict[str, Any]]:
    """Read-only model file validation."""
    tests: list[dict[str, Any]] = []
    for label, path in model_paths.items():
        try:
            resolved = path.resolve()
            if resolved.is_file() and resolved.stat().st_size > 0:
                tests.append(_test(f"model_{label}", _TEST_PASS, f"size={resolved.stat().st_size}"))
            elif resolved.is_file():
                tests.append(_test(f"model_{label}", _TEST_WARN, "file exists but is empty"))
            else:
                tests.append(_test(f"model_{label}", _TEST_FAIL, "file not found"))
        except Exception as exc:
            tests.append(_test(f"model_{label}", _TEST_FAIL, str(exc)))
    return tests


@router.get(
    "/all",
    summary="Run read-only tests for every project section",
    description=(
        "Runs read-only validation tests across every section: models, stores, services, "
        "data stores, settings, infrastructure, and model management. Each test returns "
        "PASS/FAIL/WARN/SKIP. Use POST /api/v1/tests/all for active smoke tests "
        "that create and delete test data."
    ),
)
def all_sections_status(runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    all_tests: dict[str, Any] = {}
    all_results: list[dict[str, Any]] = []

    # ── Model artifact tests ──────────────────────────────────────────
    model_results: list[dict[str, Any]] = []
    try:
        s = runtime.settings
        face_paths = {
            "human_detector": s.face_human_model_path,
            "face_detector": s.face_detector_model_path,
            "embedding": s.face_embedding_model_path,
        }
        model_results.extend(_run_model_tests(runtime, face_paths))
        fire_path = s.fire_model_path
        model_results.extend(_run_model_tests(runtime, {"fire_smoke": fire_path}))
        plate_paths = {
            "vehicle_detector": s.vehicle_detector_weights,
            "plate_detector": s.plate_detector_weights,
        }
        model_results.extend(_run_model_tests(runtime, plate_paths))
        # Check recognizer directory
        if s.plate_recognizer_dir.is_dir():
            model_results.append(_test("plate_recognizer_dir", _TEST_PASS))
        else:
            model_results.append(_test("plate_recognizer_dir", _TEST_WARN, "directory not found"))
    except Exception as exc:
        model_results.append(_test("model_checks", _TEST_FAIL, str(exc)))
    all_tests["models"] = {"tests": model_results}
    all_results.extend(model_results)

    # ── Store tests ───────────────────────────────────────────────────
    store_results: list[dict[str, Any]] = []
    for label, store in (
        ("personnel", runtime.personnel_store),
        ("shifts", runtime.shift_store),
        ("holidays", runtime.holiday_store),
        ("requests", runtime.request_store),
    ):
        store_results.extend(_run_store_tests(store, label))
    # Locations (different API)
    try:
        loc = runtime.location_store
        b = loc.count_buildings()
        s = loc.count_sections()
        r = loc.count_rooms()
        assert all(isinstance(v, int) and v >= 0 for v in (b, s, r))
        store_results.append(_test("locations_counts", _TEST_PASS, f"buildings={b},sections={s},rooms={r}"))
    except Exception as exc:
        store_results.append(_test("locations_counts", _TEST_FAIL, str(exc)))
    # Polygon zone functions
    try:
        from app.core.location_store import point_in_polygon, parse_polygon
        square = [[0, 0], [100, 0], [100, 100], [0, 100]]
        inside = point_in_polygon(50, 50, square)
        outside = point_in_polygon(200, 200, square)
        parsed = parse_polygon("[[0,0],[100,0],[100,100],[0,100]]")
        assert inside is True
        assert outside is False
        assert len(parsed) == 4
        store_results.append(_test("polygon_zone_utils", _TEST_PASS))
    except Exception as exc:
        store_results.append(_test("polygon_zone_utils", _TEST_FAIL, str(exc)))
    # Human foot point utility
    try:
        foot_x, foot_y = LocationStore._human_foot_point([10, 20, 100, 200])
        assert abs(foot_x - 55.0) < 0.001
        assert abs(foot_y - 200.0) < 0.001
        store_results.append(_test("human_foot_point", _TEST_PASS))
    except Exception as exc:
        store_results.append(_test("human_foot_point", _TEST_FAIL, str(exc)))
    all_tests["stores"] = {"tests": store_results}
    all_results.extend(store_results)

    # ── Data store tests ──────────────────────────────────────────────
    ds_results: list[dict[str, Any]] = []
    for label, ds in (
        ("plate_logs", runtime.plate_logs),
        ("fire_smoke_logs", runtime.fire_smoke_logs),
        ("human_logs", runtime.human_logs),
    ):
        try:
            count = ds.count()
            assert isinstance(count, int) and count >= 0
            ds_results.append(_test(f"{label}_count", _TEST_PASS, f"count={count}"))
        except Exception as exc:
            ds_results.append(_test(f"{label}_count", _TEST_FAIL, str(exc)))
    try:
        capacity = runtime.results._recent.maxlen
        ds_results.append(_test("result_store_capacity", _TEST_PASS, f"maxlen={capacity}"))
    except Exception as exc:
        ds_results.append(_test("result_store_capacity", _TEST_FAIL, str(exc)))
    all_tests["data_stores"] = {"tests": ds_results}
    all_results.extend(ds_results)

    # ── Settings tests ────────────────────────────────────────────────
    settings_results: list[dict[str, Any]] = []
    try:
        policy = runtime.plate_settings.general()
        settings_results.append(_test("plate_settings", _TEST_PASS if policy else _TEST_WARN,
                                       "general policy available" if policy else "no general policy"))
    except Exception as exc:
        settings_results.append(_test("plate_settings", _TEST_FAIL, str(exc)))
    try:
        qpolicy = runtime.face_quality_settings.get()
        settings_results.append(_test("face_quality_settings", _TEST_PASS if qpolicy else _TEST_WARN,
                                       "policy available" if qpolicy else "no policy"))
    except Exception as exc:
        settings_results.append(_test("face_quality_settings", _TEST_FAIL, str(exc)))
    all_tests["settings"] = {"tests": settings_results}
    all_results.extend(settings_results)

    # ── Infrastructure tests ──────────────────────────────────────────
    infra_results: list[dict[str, Any]] = []
    # Source registry
    try:
        cameras = runtime.registry.list()
        enabled = sum(1 for c in cameras if c.enabled)
        assert isinstance(cameras, list)
        infra_results.append(_test("source_registry", _TEST_PASS, f"total={len(cameras)},enabled={enabled}"))
    except Exception as exc:
        infra_results.append(_test("source_registry", _TEST_FAIL, str(exc)))
    # Broadcast
    try:
        status = runtime.broadcast.status()
        infra_results.append(_test("broadcast", _TEST_PASS, f"enabled={runtime.broadcast.enabled},subscribers={status.get('subscribers',0)}"))
    except Exception as exc:
        infra_results.append(_test("broadcast", _TEST_FAIL, str(exc)))
    # Task router
    try:
        rstatus = runtime.router.status()
        infra_results.append(_test("task_router", _TEST_PASS, f"started={rstatus.get('started',False)},rounds={rstatus.get('rounds_received',0)}"))
    except Exception as exc:
        infra_results.append(_test("task_router", _TEST_FAIL, str(exc)))
    # Video ingestor
    ingestor = runtime.video_ingestor
    if ingestor is None:
        infra_results.append(_test("video_ingestor", _TEST_SKIP, "disabled"))
    else:
        try:
            vs = ingestor.status()
            infra_results.append(_test("video_ingestor", _TEST_PASS,
                                       f"backend={vs.get('backend','?')},running={vs.get('running',False)}"))
        except Exception as exc:
            infra_results.append(_test("video_ingestor", _TEST_FAIL, str(exc)))
    all_tests["infrastructure"] = {"tests": infra_results}
    all_results.extend(infra_results)

    # ── Model management tests ────────────────────────────────────────
    mgmt_results: list[dict[str, Any]] = []
    try:
        snapshot = runtime.models.snapshot()
        resolved = list(snapshot.get("resolved_models", {}).keys())
        mgmt_results.append(_test("model_manager", _TEST_PASS, f"roles={resolved}"))
    except Exception as exc:
        mgmt_results.append(_test("model_manager", _TEST_FAIL, str(exc)))
    try:
        _ = runtime.model_conversions
        mgmt_results.append(_test("model_conversions", _TEST_PASS))
    except Exception as exc:
        mgmt_results.append(_test("model_conversions", _TEST_FAIL, str(exc)))
    all_tests["model_management"] = {"tests": mgmt_results}
    all_results.extend(mgmt_results)

    # ── Compute summary ───────────────────────────────────────────────
    total = len(all_results)
    passed = sum(1 for t in all_results if t["status"] == _TEST_PASS)
    failed = sum(1 for t in all_results if t["status"] == _TEST_FAIL)
    warned = sum(1 for t in all_results if t["status"] == _TEST_WARN)
    skipped = sum(1 for t in all_results if t["status"] == _TEST_SKIP)

    if failed:
        overall = _TEST_FAIL
    elif warned:
        overall = _TEST_WARN
    elif skipped and passed == total - skipped:
        overall = _TEST_PASS
    else:
        overall = _TEST_PASS

    all_tests["_summary"] = {
        "status": overall,
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "warnings": warned,
        "skipped": skipped,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if failed:
        failed_names = [t["name"] for t in all_results if t["status"] == _TEST_FAIL]
        all_tests["_summary"]["failed_tests"] = failed_names

    return all_tests


@router.post(
    "/all",
    summary="Run active smoke tests for every project section",
    description=(
        "Actively runs smoke tests for every store, processor, and service. "
        "This writes and deletes test data. For a read-only status check use GET /api/v1/tests/all."
    ),
)
async def all_sections_smoke_test(
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    smoke_results: dict[str, Any] = {}
    ts = int(time.time() * 1000)

    # 1. Personnel smoke
    try:
        personnel_result = await personnel_smoke_test(runtime)
        smoke_results["personnel"] = _summarize_smoke(personnel_result)
    except Exception as exc:
        smoke_results["personnel"] = {"status": "ERROR", "detail": str(exc)}

    # 2. Locations smoke
    try:
        loc_result = await locations_smoke_test(runtime)
        smoke_results["locations"] = _summarize_smoke(loc_result)
    except Exception as exc:
        smoke_results["locations"] = {"status": "ERROR", "detail": str(exc)}

    # 3. Shifts smoke
    try:
        shift_result = await shifts_smoke_test(runtime)
        smoke_results["shifts"] = _summarize_smoke(shift_result)
    except Exception as exc:
        smoke_results["shifts"] = {"status": "ERROR", "detail": str(exc)}

    # 4. Holidays smoke
    try:
        hol_result = await holidays_smoke_test(runtime)
        smoke_results["holidays"] = _summarize_smoke(hol_result)
    except Exception as exc:
        smoke_results["holidays"] = {"status": "ERROR", "detail": str(exc)}

    # 5. Requests smoke
    try:
        req_result = await requests_smoke_test(runtime)
        smoke_results["requests"] = _summarize_smoke(req_result)
    except Exception as exc:
        smoke_results["requests"] = {"status": "ERROR", "detail": str(exc)}

    # 6. Infrastructure health
    infra_checks = _infrastructure_health_check(runtime)
    smoke_results["infrastructure"] = infra_checks

    # Summary
    total = len(smoke_results)
    passed = sum(1 for v in smoke_results.values() if isinstance(v, dict) and v.get("status") == "PASS")
    skipped = sum(1 for v in smoke_results.values() if isinstance(v, dict) and v.get("status") == "SKIP")
    failed = sum(1 for v in smoke_results.values() if isinstance(v, dict) and v.get("status") in ("FAIL", "ERROR"))
    smoke_results["_summary"] = {
        "total_sections": total,
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    return smoke_results


def _summarize_smoke(result: dict[str, Any]) -> dict[str, Any]:
    summary = result.get("_summary", {})
    total = summary.get("total", 0)
    failed = summary.get("failed", 0)
    if failed > 0:
        return {
            "status": "FAIL",
            "steps_total": total,
            "steps_failed": failed,
            "failed_steps": summary.get("failed_steps", []),
        }
    if total > 0:
        return {"status": "PASS", "steps_total": total}
    return {"status": "ERROR", "detail": "No steps executed"}


def _infrastructure_health_check(runtime: Runtime) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    # Source registry
    try:
        cameras = runtime.registry.list()
        checks["source_registry"] = {
            "status": "PASS",
            "total_cameras": len(cameras),
        }
    except Exception as exc:
        checks["source_registry"] = {"status": "FAIL", "detail": str(exc)}

    # Broadcast
    try:
        enabled = runtime.broadcast.enabled
        checks["broadcast"] = {"status": "PASS", "enabled": enabled}
    except Exception as exc:
        checks["broadcast"] = {"status": "FAIL", "detail": str(exc)}

    # Task router
    try:
        router_status = runtime.router.status()
        checks["task_router"] = {
            "status": "PASS",
            "started": router_status.get("started", False),
        }
    except Exception as exc:
        checks["task_router"] = {"status": "FAIL", "detail": str(exc)}

    # Video ingestor
    ingestor = runtime.video_ingestor
    if ingestor is None:
        checks["video_ingestor"] = {"status": "SKIP", "detail": "Video ingestion is disabled"}
    else:
        try:
            ingestor_status = ingestor.status()
            checks["video_ingestor"] = {
                "status": "PASS",
                "running": ingestor_status.get("running", False),
            }
        except Exception as exc:
            checks["video_ingestor"] = {"status": "FAIL", "detail": str(exc)}

    # Model management
    try:
        snapshot = runtime.models.snapshot()
        checks["model_manager"] = {"status": "PASS", "resolved_models": len(snapshot.get("resolved_models", {}))}
    except Exception as exc:
        checks["model_manager"] = {"status": "FAIL", "detail": str(exc)}

    # Compute overall status
    any_fail = any(v.get("status") == "FAIL" for v in checks.values())
    any_skip = any(v.get("status") == "SKIP" for v in checks.values())
    any_error = any(v.get("status") == "ERROR" for v in checks.values())
    if any_fail or any_error:
        overall = "FAIL"
    elif any_skip:
        overall = "SKIP"
    else:
        overall = "PASS"

    return {
        "status": overall,
        "checks": checks,
    }
