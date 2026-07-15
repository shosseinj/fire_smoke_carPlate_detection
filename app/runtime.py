from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, settings
from app.core.result_store import ResultStore
from app.core.broadcast import AnnotatedBroadcastHub
from app.core.plate_log_store import PlateLogStore
from app.core.router import TaskRouter
from app.core.source_registry import SourceRecord, SourceRegistry
from app.core.types import TaskName
from app.core.worker import TaskWorker
from app.core.video_ingestor import VideoFileIngestor
from app.processors.fire_smoke import FireSmokeProcessor, FireSmokeSettings
from app.processors.mock import MockProcessor
from app.processors.plate import PlateRecognitionProcessor, PlateSettings


@dataclass(slots=True)
class Runtime:
    settings: Settings
    registry: SourceRegistry
    results: ResultStore
    router: TaskRouter
    broadcast: AnnotatedBroadcastHub
    plate_logs: PlateLogStore
    video_ingestor: VideoFileIngestor | None = None

    def start(self) -> None:
        self.router.start()
        if self.video_ingestor is not None:
            self.video_ingestor.start()

    def close(self) -> None:
        # End long-lived MJPEG responses first so Uvicorn reload/shutdown cannot
        # wait forever for frontend clients that still have streams open.
        self.broadcast.set_enabled(False)
        if self.video_ingestor is not None:
            self.video_ingestor.close()
        self.router.close()

    def status(self) -> dict:
        value = self.router.status()
        value["video_ingestor"] = (
            self.video_ingestor.status()
            if self.video_ingestor is not None
            else {"enabled": False, "running": False}
        )
        value["broadcast"] = self.broadcast.status()
        value["plate_log_count"] = self.plate_logs.count()
        return value


def _seed_registry(registry: SourceRegistry, project_root: Path) -> None:
    if registry.list():
        return
    seed_path = project_root / "examples" / "initial_sources.json"
    if not seed_path.is_file():
        return
    for item in json.loads(seed_path.read_text(encoding="utf-8")):
        registry.create(SourceRecord.from_dict(item))


def build_runtime(app_settings: Settings = settings) -> Runtime:
    registry = SourceRegistry(app_settings.source_registry_path)
    _seed_registry(registry, Path(__file__).resolve().parents[1])
    results = ResultStore(app_settings.recent_results_limit)
    broadcast = AnnotatedBroadcastHub(
        enabled=app_settings.broadcast_enabled,
        jpeg_quality=app_settings.broadcast_jpeg_quality,
    )
    plate_logs = PlateLogStore(app_settings.plate_log_db_path, app_settings.draw_info , app_settings.save_plate_snapshot)

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
            )
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
        video_ingestor = VideoFileIngestor(
            registry=registry,
            router=router,
            project_root=project_root,
            target_fps=app_settings.video_ingest_fps,
            loop=app_settings.video_loop,
        )
    return Runtime(
        settings=app_settings,
        registry=registry,
        results=results,
        router=router,
        broadcast=broadcast,
        plate_logs=plate_logs,
        video_ingestor=video_ingestor,
    )
