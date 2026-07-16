from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, settings
from app.core.result_store import ResultStore
from app.core.broadcast import AnnotatedBroadcastHub
from app.core.plate_log_store import PlateLogStore
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
            ),
            policy_provider=fire_smoke_logs.policy_snapshot,
        )
        plate_processor = PlateRecognitionProcessor(
            PlateSettings(
                detector_weights=app_settings.plate_detector_weights,
                recognizer_model_dir=app_settings.plate_recognizer_dir,
                device=app_settings.plate_device,
                detector_imgsz=app_settings.plate_imgsz,
                detector_confidence=app_settings.plate_confidence,
                detector_iou=app_settings.plate_iou,
                use_fp16=app_settings.plate_use_fp16,
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
    }
    router = TaskRouter(registry=registry, workers=workers, result_store=results)
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
        fire_smoke_logs=fire_smoke_logs,
        video_ingestor=video_ingestor,
    )
