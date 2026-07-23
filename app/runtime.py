from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.config import Settings, settings
from app.database import Database, get_database
from app.core.auth import initialize_auth_store
from app.core.general_settings_store import GeneralSettingsStore
from app.core.operational_settings import OperationalSettings, CAMERA_SETTINGS_METADATA_KEY
from app.core.result_store import ResultStore
from app.core.broadcast import AnnotatedBroadcastHub
from app.core.plate_log_store import PlateLogStore
from app.core.car_plate_store import CarPlateStore
from app.core.plate_settings_store import PlateDetectionPolicy, PlateSettingsStore
from app.core.model_management import (
    ModelConversionManager,
    ModelManager,
    ModelSelectionConfig,
)
from app.core.media_preview_publisher import MediaPreviewPublisher
from app.core.fire_smoke_log_store import FireSmokeLogStore
from app.core.human_log_store import HumanLogStore
from app.core.face_quality_store import FaceQualityPolicy, FaceQualitySettingsStore
from app.core.location_store import LocationStore
from app.core.personnel_store import PersonnelStore
from app.core.shift_store import ShiftStore
from app.core.holiday_store import HolidayStore
from app.core.request_store import RequestStore
from app.core.detection_log_store import DetectionLogStore
from app.core.import_progress_store import ImportProgressStore
from app.core.router import TaskRouter
from app.core.source_registry import SourceRecord, SourceRegistry
from app.core.types import FramePacket, TaskName, TaskResult
from app.core.worker import TaskWorker
from app.core.deepstream_ingestor import DeepStreamIngestor
from app.core.video_ingestor import VideoFileIngestor
from app.processors.fire_smoke import FireSmokeProcessor, FireSmokeSettings
from app.processors.base import BatchProcessor
from app.processors.face_recognition import (
    FaceRecognitionProcessor,
    FaceRecognitionSettings,
)
from app.fire_core.policy import FireSmokePolicyConfig
from app.processors.mock import MockProcessor
from app.processors.plate import PlateRecognitionProcessor, PlateSettings
from app.processors.ultralytics_loader import preload_model_dependencies


LOGGER = logging.getLogger("uvicorn.error")


@dataclass(slots=True)
class Runtime:
    settings: Settings
    database: Database
    registry: SourceRegistry
    results: ResultStore
    router: TaskRouter
    broadcast: AnnotatedBroadcastHub
    plate_logs: PlateLogStore
    car_plates: CarPlateStore
    plate_settings: PlateSettingsStore
    models: ModelManager
    model_conversions: ModelConversionManager
    fire_smoke_logs: FireSmokeLogStore
    human_logs: HumanLogStore
    face_quality_settings: FaceQualitySettingsStore
    face_processor: BatchProcessor
    personnel_store: PersonnelStore
    location_store: LocationStore
    shift_store: ShiftStore
    holiday_store: HolidayStore
    request_store: RequestStore
    detection_log_store: DetectionLogStore
    import_progress: ImportProgressStore
    general_settings: GeneralSettingsStore
    video_ingestor: VideoFileIngestor | DeepStreamIngestor | None = None
    media_preview: MediaPreviewPublisher | None = None

    def operational_settings(self):
        return self.general_settings.get().operational

    def resolve_camera_settings(self, camera_id: str) -> dict[str, object]:
        general = self.operational_settings().to_dict()
        gs = self.general_settings.get()
        force = gs.force
        camera = self.registry.require(camera_id)
        overrides = dict(camera.metadata.get(CAMERA_SETTINGS_METADATA_KEY) or {})
        if force:
            from app.core.settings_policy import resolve_all_camera_settings
            return resolve_all_camera_settings(overrides, general, force=True)
        return {**general, **overrides}

    def update_camera_overrides(self, camera_id: str, changes: dict[str, object]) -> dict[str, object]:
        camera = self.registry.require(camera_id)
        current = dict(camera.metadata.get(CAMERA_SETTINGS_METADATA_KEY) or {})
        candidate = self.operational_settings().updated({**current, **{k: v for k, v in changes.items() if v is not None}})
        for key, value in changes.items():
            if value is None:
                current.pop(key, None)
            else:
                current[key] = value
        metadata = dict(camera.metadata)
        if current:
            metadata[CAMERA_SETTINGS_METADATA_KEY] = current
        else:
            metadata.pop(CAMERA_SETTINGS_METADATA_KEY, None)
        self.registry.update(camera_id, metadata=metadata)
        if self.video_ingestor is not None and hasattr(self.video_ingestor, "restart_source"):
            self.video_ingestor.restart_source(camera_id)
        return self.resolve_camera_settings(camera_id)

    def apply_operational_settings(self) -> None:
        from dataclasses import replace
        current = self.operational_settings()
        workers = self.router.workers
        fire = workers.get(TaskName.FIRE_SMOKE)
        if fire is not None and hasattr(fire.processor, "settings"):
            fire.processor.settings = replace(fire.processor.settings, fire_candidate_confidence=current.fire_confidence, smoke_candidate_confidence=current.smoke_confidence)
        plate = workers.get(TaskName.PLATE_RECOGNITION)
        if plate is not None and hasattr(plate.processor, "settings"):
            plate.processor.settings = replace(plate.processor.settings, detector_confidence=current.plate_confidence, detector_iou=current.plate_iou, vehicle_confidence=current.vehicle_confidence, vehicle_iou=current.vehicle_iou)
        face = workers.get(TaskName.FACE_RECOGNITION)
        if face is not None and hasattr(face.processor, "settings"):
            face.processor.settings = replace(face.processor.settings, human_confidence=current.face_human_confidence, face_confidence=current.face_detection_confidence, recognition_threshold=current.face_recognition_threshold)
        if self.video_ingestor is not None:
            for name in ("target_fps", "preview_fps", "rtsp_reconnect_seconds"):
                source = {"target_fps": "video_ingest_fps", "preview_fps": "video_preview_fps", "rtsp_reconnect_seconds": "rtsp_reconnect_seconds"}[name]
                if hasattr(self.video_ingestor, name):
                    setattr(self.video_ingestor, name, getattr(current, source))
        self.broadcast.set_enabled(current.broadcast_enabled)
        for camera in self.registry.list():
            if self.video_ingestor is not None and hasattr(self.video_ingestor, "restart_source"):
                self.video_ingestor.restart_source(camera.source_id)

    def selected_model_records(self) -> list[dict[str, object]]:
        """Return the exact startup model choices in runtime load order."""

        snapshot = self.models.snapshot()
        records: list[dict[str, object]] = []
        for role in self.models.ROLE_FIELDS:
            resolved = snapshot["resolved_models"][role]
            candidates = list(resolved["candidates"])
            records.append(
                {
                    "role": role,
                    "selected": resolved["selected"],
                    "active": resolved["active_choice"],
                    "fallbacks": candidates[1:] if candidates else [],
                }
            )

        recognizer_directory = self.settings.plate_recognizer_dir.resolve()
        recognizer_model = recognizer_directory / "model.pt"
        records.append(
            {
                "role": "plate_recognizer",
                "selected": str(recognizer_directory),
                "active": str(recognizer_model) if recognizer_model.is_file() else None,
                "fallbacks": [],
            }
        )
        for role, path in (
            ("face_human_detector", self.settings.face_human_model_path),
            ("face_detector", self.settings.face_detector_model_path),
            ("face_embedding", self.settings.face_embedding_model_path),
        ):
            records.append(
                {
                    "role": role,
                    "selected": str(path.resolve()),
                    "active": str(path.resolve()) if path.is_file() else None,
                    "fallbacks": [],
                }
            )
        return records

    def _log_selected_models(self) -> None:
        for record in self.selected_model_records():
            LOGGER.info(
                "MODEL_SELECTED role=%s selected=%s active=%s fallbacks=%s",
                record["role"],
                record["selected"],
                record["active"] or "MISSING",
                record["fallbacks"] or "none",
            )

    def start(self) -> None:
        self._log_selected_models()
        if self.settings.processor_mode == "real":
            preload_model_dependencies()
            if isinstance(self.face_processor, FaceRecognitionProcessor):
                try:
                    self.face_processor.preload()
                except Exception as exc:
                    LOGGER.warning("FACE_RECOGNITION_NOT_READY %s", exc)
        self.router.start()
        try:
            if self.media_preview is not None:
                try:
                    self.media_preview.start()
                except Exception as exc:
                    LOGGER.warning("MEDIA_PREVIEW_NOT_READY %s", exc)
            if self.video_ingestor is not None:
                self.video_ingestor.start()
        except Exception:
            if self.media_preview is not None:
                self.media_preview.close()
            self.router.close()
            raise

    def close(self) -> None:
        # End long-lived MJPEG responses first so Uvicorn reload/shutdown cannot
        # wait forever for frontend clients that still have streams open.
        self.broadcast.close()
        self.model_conversions.close()
        if self.media_preview is not None:
            self.media_preview.close()
        if self.video_ingestor is not None:
            self.video_ingestor.close()
        self.router.close()
        self.fire_smoke_logs.close()
        self.plate_logs.close()
        self.human_logs.close()
        self.registry.close()
        self.database.dispose()

    def status(self) -> dict:
        value = self.router.status()
        value["video_ingestor"] = (
            self.video_ingestor.status()
            if self.video_ingestor is not None
            else {"enabled": False, "running": False}
        )
        value["broadcast"] = self.broadcast.status()
        value["media_preview"] = (
            self.media_preview.status()
            if self.media_preview is not None
            else {"enabled": False, "running": False}
        )
        value["plate_log_count"] = self.plate_logs.count()
        value["plate_logs"] = self.plate_logs.status()
        value["fire_smoke_logs"] = self.fire_smoke_logs.status()
        value["human_logs"] = self.human_logs.status()
        value["personnel_count"] = self.personnel_store.count()
        value["location_counts"] = {
            "buildings": self.location_store.count_buildings(),
            "sections": self.location_store.count_sections(),
            "rooms": self.location_store.count_rooms(),
        }
        value["shift_count"] = self.shift_store.count()
        value["holiday_count"] = self.holiday_store.count_active()
        value["request_count"] = self.request_store.count()
        value["detection_log_count"] = self.detection_log_store.count_by_status()
        return value


def _seed_registry(
    registry: SourceRegistry,
    source_registry_path: Path,
    project_root: Path,
) -> None:
    if registry.list():
        return
    seed_paths = (
        source_registry_path,
        project_root / "examples" / "initial_sources.json",
    )
    for seed_path in seed_paths:
        if not seed_path.is_file():
            continue
        payload = json.loads(seed_path.read_text(encoding="utf-8"))
        registry.import_if_empty(SourceRecord.from_dict(item) for item in payload)
        return


def build_runtime(app_settings: Settings = settings) -> Runtime:
    database = get_database(
        app_settings.database_url,
        echo=app_settings.database_echo,
        pool_size=app_settings.database_pool_size,
        max_overflow=app_settings.database_max_overflow,
    )
    database.verify_schema()
    general_settings = GeneralSettingsStore(
        database,
        OperationalSettings.from_app_settings(app_settings),
    )
    operational = general_settings.get().operational
    initialize_auth_store(database, app_settings)
    registry = SourceRegistry(database)
    _seed_registry(
        registry,
        app_settings.source_registry_path,
        Path(__file__).resolve().parents[1],
    )
    results = ResultStore(app_settings.recent_results_limit)
    broadcast = AnnotatedBroadcastHub(
        enabled=operational.broadcast_enabled,
        jpeg_quality=operational.broadcast_jpeg_quality,
        wall_jpeg_quality=operational.broadcast_wall_jpeg_quality,
        wall_max_width=operational.broadcast_wall_max_width,
        wall_max_height=operational.broadcast_wall_max_height,
        face_overlay_ttl_ms=app_settings.broadcast_face_overlay_ttl_ms,
    )
    registry.add_listener(broadcast.publish_source_change)
    plate_settings = PlateSettingsStore(
        database,
        default_policy=PlateDetectionPolicy(
            vehicle_confidence=operational.vehicle_confidence,
            plate_confidence=operational.plate_confidence,
            ocr_confidence=app_settings.plate_ocr_confidence,
            min_vehicle_width_pixels=app_settings.min_vehicle_width_pixels,
            min_vehicle_height_pixels=app_settings.min_vehicle_height_pixels,
            min_vehicle_area_ratio=app_settings.min_vehicle_area_ratio,
            vehicle_crop_padding_ratio=app_settings.vehicle_crop_padding_ratio,
        ),
    )
    registry.add_listener(plate_settings.on_source_change)
    face_quality_settings = FaceQualitySettingsStore(
        database,
        FaceQualityPolicy(
            quality_threshold=app_settings.face_quality_threshold,
            blur_threshold=app_settings.face_blur_threshold,
            min_face_width=app_settings.face_min_width,
            min_face_height=app_settings.face_min_height,
            min_eye_distance=app_settings.face_min_eye_distance,
            max_abs_yaw=app_settings.face_max_abs_yaw,
            max_abs_pitch=app_settings.face_max_abs_pitch,
            max_abs_roll=app_settings.face_max_abs_roll,
            require_landmarks=app_settings.face_require_landmarks,
        ),
    )
    face_quality_policy = face_quality_settings.get()
    model_root = app_settings.model_root_path.resolve()

    def model_relative(path: Path) -> str:
        try:
            return path.resolve().relative_to(model_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Configured model must be inside MODEL_ROOT_PATH: {path}"
            ) from exc

    models = ModelManager(
        database,
        model_root,
        default_config=ModelSelectionConfig(
            fire_smoke_model=model_relative(app_settings.fire_model_path),
            vehicle_detector_model=model_relative(
                app_settings.vehicle_detector_weights
            ),
            plate_detector_model=model_relative(
                app_settings.plate_detector_weights
            ),
            preferred_format=app_settings.model_preferred_format,
            allow_onnx_fallback=app_settings.model_allow_onnx_fallback,
            allow_pt_fallback=app_settings.model_allow_pt_fallback,
            export_imgsz=app_settings.model_export_imgsz,
            export_batch_size=app_settings.model_export_batch_size,
            export_workspace_gb=app_settings.model_export_workspace_gb,
            export_half=app_settings.model_export_half,
            export_dynamic=app_settings.model_export_dynamic,
            export_timeout_seconds=app_settings.model_export_timeout_seconds,
        ),
    )
    model_conversions = ModelConversionManager(models)
    plate_logs = PlateLogStore(
        database,
        app_settings.draw_info,
        app_settings.save_plate_snapshot,
        queue_size=app_settings.plate_log_queue_size,
    )
    car_plates = CarPlateStore(database)
    fire_smoke_logs = FireSmokeLogStore(
        database,
        app_settings.saved_media_path,
        default_policy=FireSmokePolicyConfig(
            window_seconds=app_settings.fire_severity_window_seconds,
            low_count=app_settings.fire_low_incident_count,
            medium_count=app_settings.fire_medium_incident_count,
            high_count=app_settings.fire_high_incident_count,
        ),
    )
    human_logs = HumanLogStore(
        database,
        app_settings.saved_media_path,
        queue_size=app_settings.human_media_queue_size,
        video_fps=app_settings.human_video_fps,
        video_idle_seconds=app_settings.human_video_idle_seconds,
        snapshot_min_improvement=app_settings.human_snapshot_min_improvement,
    )
    personnel_store = PersonnelStore(
        database,
        app_settings.saved_media_path,
    )
    location_store = LocationStore(
        database,
    )
    shift_store = ShiftStore(database)
    holiday_store = HolidayStore(database)
    request_store = RequestStore(database)
    detection_log_store = DetectionLogStore(database)
    import_progress = ImportProgressStore(database)

    if app_settings.processor_mode == "mock":
        fire_processor = MockProcessor(TaskName.FIRE_SMOKE)
        plate_processor = MockProcessor(TaskName.PLATE_RECOGNITION)
        face_processor: BatchProcessor = MockProcessor(TaskName.FACE_RECOGNITION)
    elif app_settings.processor_mode == "real":
        fire_processor = FireSmokeProcessor(
            FireSmokeSettings(
                model_path=app_settings.fire_model_path,
                device=app_settings.fire_device,
                imgsz=app_settings.fire_imgsz,
                batch_size=app_settings.fire_batch_size,
                engine_fixed_batch=app_settings.fire_engine_fixed_batch,
                fire_candidate_confidence=operational.fire_confidence,
                smoke_candidate_confidence=operational.smoke_confidence,
                fire_low_confidence=app_settings.fire_low_severity_confidence,
                fire_medium_confidence=app_settings.fire_medium_severity_confidence,
                fire_high_confidence=app_settings.fire_high_severity_confidence,
                smoke_low_confidence=app_settings.smoke_low_severity_confidence,
                smoke_medium_confidence=app_settings.smoke_medium_severity_confidence,
                smoke_high_confidence=app_settings.smoke_high_severity_confidence,
            ),
            policy_provider=fire_smoke_logs.policy_snapshot,
            model_provider=models.provider("fire_smoke"),
        )
        plate_processor = PlateRecognitionProcessor(
            PlateSettings(
                detector_weights=app_settings.plate_detector_weights,
                vehicle_detector_weights=app_settings.vehicle_detector_weights,
                recognizer_model_dir=app_settings.plate_recognizer_dir,
                device=app_settings.plate_device,
                detector_imgsz=app_settings.plate_imgsz,
                detector_confidence=operational.plate_confidence,
                detector_iou=operational.plate_iou,
                plate_crop_batch_size=app_settings.plate_crop_batch_size,
                plate_class_ids=app_settings.plate_class_ids,
                vehicle_confidence=operational.vehicle_confidence,
                vehicle_iou=operational.vehicle_iou,
                vehicle_imgsz=app_settings.vehicle_imgsz,
                vehicle_max_per_frame=app_settings.vehicle_max_per_frame,
                vehicle_class_ids=app_settings.vehicle_class_ids,
                vehicle_crop_padding_ratio=app_settings.vehicle_crop_padding_ratio,
                ocr_confidence=app_settings.plate_ocr_confidence,
                min_vehicle_width_pixels=app_settings.min_vehicle_width_pixels,
                min_vehicle_height_pixels=app_settings.min_vehicle_height_pixels,
                min_vehicle_area_ratio=app_settings.min_vehicle_area_ratio,
                use_fp16=app_settings.plate_use_fp16,
            ),
            settings_provider=plate_settings.resolve,
            vehicle_model_provider=models.provider("vehicle_detector"),
            plate_model_provider=models.provider("plate_detector"),
        )
        face_processor = FaceRecognitionProcessor(
            FaceRecognitionSettings(
                human_model_path=app_settings.face_human_model_path,
                face_model_path=app_settings.face_detector_model_path,
                embedding_model_path=app_settings.face_embedding_model_path,
                device=app_settings.face_device,
                batch_size=app_settings.face_batch_size,
                human_imgsz=app_settings.face_human_imgsz,
                face_imgsz=app_settings.face_detector_imgsz,
                human_engine_fixed_batch=app_settings.face_human_engine_fixed_batch,
                face_engine_fixed_batch=app_settings.face_detector_engine_fixed_batch,
                human_confidence=operational.face_human_confidence,
                face_confidence=operational.face_detection_confidence,
                recognition_threshold=operational.face_recognition_threshold,
                min_face_width=face_quality_policy.min_face_width,
                min_face_height=face_quality_policy.min_face_height,
                blur_threshold=face_quality_policy.blur_threshold,
                min_eye_distance=face_quality_policy.min_eye_distance,
                quality_threshold=face_quality_policy.quality_threshold,
                max_abs_yaw=face_quality_policy.max_abs_yaw,
                max_abs_pitch=face_quality_policy.max_abs_pitch,
                max_abs_roll=face_quality_policy.max_abs_roll,
                require_landmarks=face_quality_policy.require_landmarks,
                tracker_high_threshold=app_settings.face_tracker_high_threshold,
                tracker_low_threshold=app_settings.face_tracker_low_threshold,
                tracker_new_threshold=app_settings.face_tracker_new_threshold,
                tracker_match_threshold=app_settings.face_tracker_match_threshold,
                tracker_max_missed=app_settings.face_tracker_max_missed,
                history_size=app_settings.face_history_size,
                stable_min_hits=app_settings.face_stable_min_hits,
                embedding_batch_size=app_settings.face_embedding_batch_size,
                vector_size=app_settings.face_vector_size,
                qdrant_collection=app_settings.face_qdrant_collection,
                qdrant_url=app_settings.face_qdrant_url,
                qdrant_api_key=app_settings.face_qdrant_api_key,
            ),
            database=database,
        )
    else:
        raise ValueError("PROCESSOR_MODE must be 'real' or 'mock'")

    # ── Location observer: polygon-based detection-to-room matching ──

    def _build_location_observer(
        ls: LocationStore,
        reg: SourceRegistry,
    ) -> Callable[[FramePacket, TaskResult], None]:
        """Return a location_observer that matches detected objects to room polygons."""
        def location_observer(packet: FramePacket, result: TaskResult) -> None:
            if result.error:
                return
            if result.task not in (
                TaskName.FACE_RECOGNITION,
                TaskName.PLATE_RECOGNITION,
                TaskName.FIRE_SMOKE,
            ):
                return
            source_id = result.source_id
            frame_index = result.frame_index
            if not source_id:
                return
            # Resolve section_id from camera
            cam = reg.get(source_id)
            if cam is None:
                return
            section_id = cam.metadata.get("section_id") if cam.metadata else None
            if section_id is None:
                return
            detections: list[dict[str, object]] = []

            if result.task == TaskName.FACE_RECOGNITION:
                faces = result.data.get("faces", [])
                for face in faces:
                    bbox = face.get("bbox")
                    if bbox and len(bbox) >= 4:
                        cx = (bbox[0] + bbox[2]) / 2.0
                        cy = (bbox[1] + bbox[3]) / 2.0
                        detections.append({
                            "cx": cx,
                            "cy": cy,
                            "personnel_id": face.get("personnel_id") or face.get("person"),
                            "detection_event_id": frame_index,
                        })
            elif result.task == TaskName.PLATE_RECOGNITION:
                plates = result.data.get("plates", [])
                for plate in plates:
                    bbox = plate.get("bbox")
                    if bbox and len(bbox) >= 4:
                        cx = (bbox[0] + bbox[2]) / 2.0
                        cy = (bbox[1] + bbox[3]) / 2.0
                        detections.append({
                            "cx": cx,
                            "cy": cy,
                            "personnel_id": None,
                            "detection_event_id": frame_index,
                        })
            elif result.task == TaskName.FIRE_SMOKE:
                tracks = result.data.get("tracks", [])
                for track in tracks:
                    bbox = track.get("bbox")
                    if bbox and len(bbox) >= 4:
                        cx = (bbox[0] + bbox[2]) / 2.0
                        cy = (bbox[1] + bbox[3]) / 2.0
                        detections.append({
                            "cx": cx,
                            "cy": cy,
                            "personnel_id": None,
                            "detection_event_id": frame_index,
                        })

            for det in detections:
                try:
                    ls.match_detection_to_rooms(
                        section_id=section_id,
                        detection_type=result.task.value,
                        detection_event_id=det["detection_event_id"],
                        bbox_center_x=det["cx"],
                        bbox_center_y=det["cy"],
                        personnel_id=det["personnel_id"],
                        camera_id=source_id,
                    )
                except Exception:
                    LOGGER.exception(
                        "Location matching failed: source=%s task=%s",
                        source_id,
                        result.task.value,
                    )

        return location_observer

    location_obs = _build_location_observer(location_store, registry)

    workers = {
        TaskName.FIRE_SMOKE: TaskWorker(
            processor=fire_processor,
            result_store=results,
            batch_size=app_settings.fire_batch_size,
            max_wait_ms=app_settings.fire_max_wait_ms,
            num_threads=app_settings.worker_threads,
            queue_policy=app_settings.task_queue_policy,
            queue_capacity=app_settings.task_queue_capacity,
            queue_block_timeout_ms=app_settings.task_queue_block_timeout_ms,
            result_callback=broadcast.publish_result,
            result_observer=fire_smoke_logs.observe_result,
            location_observer=location_obs,
        ),
        TaskName.PLATE_RECOGNITION: TaskWorker(
            processor=plate_processor,
            result_store=results,
            batch_size=app_settings.plate_batch_size,
            max_wait_ms=app_settings.plate_max_wait_ms,
            num_threads=app_settings.worker_threads,
            queue_policy=app_settings.task_queue_policy,
            queue_capacity=app_settings.task_queue_capacity,
            queue_block_timeout_ms=app_settings.task_queue_block_timeout_ms,
            result_callback=broadcast.publish_result,
            result_observer=plate_logs.observe_result,
            location_observer=location_obs,
        ),
        TaskName.FACE_RECOGNITION: TaskWorker(
            processor=face_processor,
            result_store=results,
            batch_size=app_settings.face_batch_size,
            max_wait_ms=app_settings.face_max_wait_ms,
            num_threads=app_settings.worker_threads,
            queue_policy=app_settings.task_queue_policy,
            queue_capacity=app_settings.task_queue_capacity,
            queue_block_timeout_ms=app_settings.task_queue_block_timeout_ms,
            result_callback=broadcast.publish_result,
            result_observer=human_logs.observe_result,
            location_observer=location_obs,
        ),
    }
    router = TaskRouter(
        registry=registry,
        workers=workers,
        result_store=results,
        play_only_callback=broadcast.publish_passthrough,
    )
    project_root = Path(__file__).resolve().parents[1]
    video_ingestor = None
    if app_settings.video_ingestion_enabled:
        common_ingestor_settings = {
            "registry": registry,
            "router": router,
            "project_root": project_root,
            "target_fps": operational.video_ingest_fps,
            "gpu_resize_enabled": app_settings.gpu_resize_enabled,
            "loop": operational.video_loop,
            "rtsp_transport": operational.rtsp_transport,
            "rtsp_reconnect_seconds": operational.rtsp_reconnect_seconds,
        }
        if app_settings.video_ingest_backend == "deepstream":
            print('\n\n\n\ningest video with deepstream\n\n')
            video_ingestor = DeepStreamIngestor(
                **common_ingestor_settings,
                rtsp_enabled=app_settings.rtsp_ingestion_enabled,
                rtsp_latency_ms=operational.deepstream_rtsp_latency_ms,
                preview_fps=operational.video_preview_fps,
                rtsp_stall_timeout_seconds=(
                    operational.deepstream_rtsp_stall_timeout_seconds
                ),
                skip_taskless_sources=app_settings.skip_taskless_sources,
            )
        elif app_settings.video_ingest_backend == "opencv":
            video_ingestor = VideoFileIngestor(
                **common_ingestor_settings,
                rtsp_open_timeout_ms=operational.rtsp_open_timeout_ms,
                rtsp_read_timeout_ms=operational.rtsp_read_timeout_ms,
            )
        else:
            raise ValueError("VIDEO_INGEST_BACKEND must be 'deepstream' or 'opencv'")
    media_preview = MediaPreviewPublisher(
        registry=registry,
        project_root=project_root,
        publish_base=app_settings.media_preview_publish_base,
        enabled=app_settings.media_preview_enabled,
        rtsp_enabled=app_settings.rtsp_ingestion_enabled,
        rtsp_transport=operational.rtsp_transport,
        rtsp_latency_ms=operational.deepstream_rtsp_latency_ms,
        reconnect_seconds=operational.rtsp_reconnect_seconds,
    )
    return Runtime(
        settings=app_settings,
        database=database,
        registry=registry,
        results=results,
        router=router,
        broadcast=broadcast,
        plate_logs=plate_logs,
        car_plates=car_plates,
        plate_settings=plate_settings,
        models=models,
        model_conversions=model_conversions,
        fire_smoke_logs=fire_smoke_logs,
        human_logs=human_logs,
        face_quality_settings=face_quality_settings,
        face_processor=face_processor,
        personnel_store=personnel_store,
        location_store=location_store,
        shift_store=shift_store,
        holiday_store=holiday_store,
        request_store=request_store,
        detection_log_store=detection_log_store,
        import_progress=import_progress,
        general_settings=general_settings,
        video_ingestor=video_ingestor,
        media_preview=media_preview,
    )
