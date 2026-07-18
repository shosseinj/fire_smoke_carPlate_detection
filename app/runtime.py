from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, settings
from app.core.result_store import ResultStore
from app.core.broadcast import AnnotatedBroadcastHub
from app.core.plate_log_store import PlateLogStore
from app.core.plate_settings_store import PlateDetectionPolicy, PlateSettingsStore
from app.core.model_management import (
    ModelConversionManager,
    ModelManager,
    ModelSelectionConfig,
)
from app.core.fire_smoke_log_store import FireSmokeLogStore
from app.core.human_log_store import HumanLogStore
from app.core.face_quality_store import FaceQualityPolicy, FaceQualitySettingsStore
from app.core.router import TaskRouter
from app.core.source_registry import SourceRecord, SourceRegistry
from app.core.types import TaskName
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
    registry: SourceRegistry
    results: ResultStore
    router: TaskRouter
    broadcast: AnnotatedBroadcastHub
    plate_logs: PlateLogStore
    plate_settings: PlateSettingsStore
    models: ModelManager
    model_conversions: ModelConversionManager
    fire_smoke_logs: FireSmokeLogStore
    human_logs: HumanLogStore
    face_quality_settings: FaceQualitySettingsStore
    face_processor: BatchProcessor
    video_ingestor: VideoFileIngestor | DeepStreamIngestor | None = None

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
            if self.video_ingestor is not None:
                self.video_ingestor.start()
        except Exception:
            self.router.close()
            raise

    def close(self) -> None:
        # End long-lived MJPEG responses first so Uvicorn reload/shutdown cannot
        # wait forever for frontend clients that still have streams open.
        self.broadcast.set_enabled(False)
        self.model_conversions.close()
        if self.video_ingestor is not None:
            self.video_ingestor.close()
        self.router.close()
        self.fire_smoke_logs.close()
        self.human_logs.close()
        self.registry.close()

    def status(self) -> dict:
        value = self.router.status()
        value["video_ingestor"] = (
            self.video_ingestor.status()
            if self.video_ingestor is not None
            else {"enabled": False, "running": False}
        )
        value["broadcast"] = self.broadcast.status()
        value["plate_log_count"] = self.plate_logs.count()
        value["fire_smoke_logs"] = self.fire_smoke_logs.status()
        value["human_logs"] = self.human_logs.status()
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
    registry = SourceRegistry(app_settings.camera_db_path)
    _seed_registry(
        registry,
        app_settings.source_registry_path,
        Path(__file__).resolve().parents[1],
    )
    results = ResultStore(app_settings.recent_results_limit)
    broadcast = AnnotatedBroadcastHub(
        enabled=app_settings.broadcast_enabled,
        jpeg_quality=app_settings.broadcast_jpeg_quality,
    )
    registry.add_listener(broadcast.publish_source_change)
    plate_settings = PlateSettingsStore(
        app_settings.camera_db_path,
        default_policy=PlateDetectionPolicy(
            vehicle_confidence=app_settings.vehicle_confidence,
            plate_confidence=app_settings.plate_confidence,
            ocr_confidence=app_settings.plate_ocr_confidence,
            min_vehicle_width_pixels=app_settings.min_vehicle_width_pixels,
            min_vehicle_height_pixels=app_settings.min_vehicle_height_pixels,
            min_vehicle_area_ratio=app_settings.min_vehicle_area_ratio,
            vehicle_crop_padding_ratio=app_settings.vehicle_crop_padding_ratio,
        ),
    )
    registry.add_listener(plate_settings.on_source_change)
    face_quality_settings = FaceQualitySettingsStore(
        app_settings.plate_log_db_path,
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
        app_settings.camera_db_path,
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
    plate_logs = PlateLogStore(app_settings.plate_log_db_path, app_settings.draw_info , app_settings.save_plate_snapshot)
    fire_smoke_logs = FireSmokeLogStore(
        app_settings.plate_log_db_path,
        app_settings.saved_media_path,
        default_policy=FireSmokePolicyConfig(
            window_seconds=app_settings.fire_severity_window_seconds,
            low_count=app_settings.fire_low_incident_count,
            medium_count=app_settings.fire_medium_incident_count,
            high_count=app_settings.fire_high_incident_count,
        ),
    )
    human_logs = HumanLogStore(
        app_settings.plate_log_db_path,
        app_settings.saved_media_path,
        video_fps=app_settings.human_video_fps,
        video_idle_seconds=app_settings.human_video_idle_seconds,
        snapshot_min_improvement=app_settings.human_snapshot_min_improvement,
    )

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
                fire_candidate_confidence=app_settings.fire_confidence,
                smoke_candidate_confidence=app_settings.smoke_confidence,
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
                detector_confidence=app_settings.plate_confidence,
                detector_iou=app_settings.plate_iou,
                plate_crop_batch_size=app_settings.plate_crop_batch_size,
                plate_class_ids=app_settings.plate_class_ids,
                vehicle_confidence=app_settings.vehicle_confidence,
                vehicle_iou=app_settings.vehicle_iou,
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
                human_confidence=app_settings.face_human_confidence,
                face_confidence=app_settings.face_detection_confidence,
                recognition_threshold=app_settings.face_recognition_threshold,
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
                qdrant_path=app_settings.face_qdrant_path,
                qdrant_api_key=app_settings.face_qdrant_api_key,
            )
        )
    else:
        raise ValueError("PROCESSOR_MODE must be 'real' or 'mock'")

    workers = {
        TaskName.FIRE_SMOKE: TaskWorker(
            processor=fire_processor,
            result_store=results,
            batch_size=app_settings.fire_batch_size,
            max_wait_ms=app_settings.fire_max_wait_ms,
            result_callback=broadcast.publish_result,
            result_observer=fire_smoke_logs.observe_result,
        ),
        TaskName.PLATE_RECOGNITION: TaskWorker(
            processor=plate_processor,
            result_store=results,
            batch_size=app_settings.plate_batch_size,
            max_wait_ms=app_settings.plate_max_wait_ms,
            result_callback=broadcast.publish_result,
            result_observer=plate_logs.insert_result,
        ),
        TaskName.FACE_RECOGNITION: TaskWorker(
            processor=face_processor,
            result_store=results,
            batch_size=app_settings.face_batch_size,
            max_wait_ms=app_settings.face_max_wait_ms,
            result_callback=broadcast.publish_result,
            result_observer=human_logs.observe_result,
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
            "target_fps": app_settings.video_ingest_fps,
            "loop": app_settings.video_loop,
            "rtsp_transport": app_settings.rtsp_transport,
            "rtsp_reconnect_seconds": app_settings.rtsp_reconnect_seconds,
        }
        if app_settings.video_ingest_backend == "deepstream":
            print('\n\n\n\ningest video with deepstream\n\n')
            video_ingestor = DeepStreamIngestor(
                **common_ingestor_settings,
                rtsp_enabled=app_settings.rtsp_ingestion_enabled,
                rtsp_latency_ms=app_settings.deepstream_rtsp_latency_ms,
                preview_fps=app_settings.video_preview_fps,
                rtsp_stall_timeout_seconds=(
                    app_settings.deepstream_rtsp_stall_timeout_seconds
                ),
            )
        elif app_settings.video_ingest_backend == "opencv":
            video_ingestor = VideoFileIngestor(
                **common_ingestor_settings,
                rtsp_open_timeout_ms=app_settings.rtsp_open_timeout_ms,
                rtsp_read_timeout_ms=app_settings.rtsp_read_timeout_ms,
            )
        else:
            raise ValueError("VIDEO_INGEST_BACKEND must be 'deepstream' or 'opencv'")
    return Runtime(
        settings=app_settings,
        registry=registry,
        results=results,
        router=router,
        broadcast=broadcast,
        plate_logs=plate_logs,
        plate_settings=plate_settings,
        models=models,
        model_conversions=model_conversions,
        fire_smoke_logs=fire_smoke_logs,
        human_logs=human_logs,
        face_quality_settings=face_quality_settings,
        face_processor=face_processor,
        video_ingestor=video_ingestor,
    )
