from __future__ import annotations

import json
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
from app.core.router import TaskRouter
from app.core.source_registry import SourceRecord, SourceRegistry
from app.core.types import TaskName
from app.core.worker import TaskWorker
from app.core.deepstream_ingestor import DeepStreamIngestor
from app.core.video_ingestor import VideoFileIngestor
from app.processors.fire_smoke import FireSmokeProcessor, FireSmokeSettings
from app.fire_core.policy import FireSmokePolicyConfig
from app.processors.mock import MockProcessor
from app.processors.plate import PlateRecognitionProcessor, PlateSettings
from app.processors.ultralytics_loader import preload_model_dependencies


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
    video_ingestor: VideoFileIngestor | DeepStreamIngestor | None = None

    def start(self) -> None:
        if self.settings.processor_mode == "real":
            preload_model_dependencies()
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

    if app_settings.processor_mode == "mock":
        fire_processor = MockProcessor(TaskName.FIRE_SMOKE)
        plate_processor = MockProcessor(TaskName.PLATE_RECOGNITION)
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
        video_ingestor=video_ingestor,
    )
