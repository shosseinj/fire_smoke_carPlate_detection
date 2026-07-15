from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in {None, ""} else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in {None, ""} else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default)).expanduser()
    return value if value.is_absolute() else (ROOT / value).resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Unified Video AI Task Router")
    processor_mode: str = os.getenv("PROCESSOR_MODE", "real").strip().lower()
    source_registry_path: Path = _path("SOURCE_REGISTRY_PATH", "data/sources.json")
    recent_results_limit: int = _env_int("RECENT_RESULTS_LIMIT", 2000)

    video_ingestion_enabled: bool = _env_bool("VIDEO_INGESTION_ENABLED", True)
    video_ingest_fps: float = _env_float("VIDEO_INGEST_FPS", 5.0)
    video_loop: bool = _env_bool("VIDEO_LOOP", True)
    broadcast_enabled: bool = _env_bool("BROADCAST_ENABLED", True)
    broadcast_jpeg_quality: int = _env_int("BROADCAST_JPEG_QUALITY", 82)
    plate_log_db_path: Path = _path("PLATE_LOG_DB_PATH", "plate_logs.sqlite3")

    fire_model_path: Path = _path("FIRE_SMOKE_MODEL_PATH", "weights/fire_smoke/model.engine")
    fire_device: str = os.getenv("FIRE_SMOKE_DEVICE", "0")
    fire_batch_size: int = _env_int("FIRE_SMOKE_BATCH_SIZE", 8)
    fire_max_wait_ms: float = _env_float("FIRE_SMOKE_MAX_WAIT_MS", 25.0)
    fire_imgsz: int = _env_int("FIRE_SMOKE_IMGSZ", 640)
    fire_engine_fixed_batch: int = _env_int("FIRE_SMOKE_ENGINE_FIXED_BATCH", 8)

    plate_detector_weights: Path = _path("PLATE_DETECTOR_WEIGHTS", "weights/plate_detector/model.pt")
    plate_recognizer_dir: Path = _path("PLATE_RECOGNIZER_DIR", "weights/plate_recognizer")
    plate_device: str = os.getenv("PLATE_DEVICE", "0")
    plate_batch_size: int = _env_int("PLATE_BATCH_SIZE", 8)
    plate_max_wait_ms: float = _env_float("PLATE_MAX_WAIT_MS", 25.0)
    plate_imgsz: int = _env_int("PLATE_IMGSZ", 640)
    plate_confidence: float = _env_float("PLATE_CONFIDENCE", 0.35)
    plate_iou: float = _env_float("PLATE_IOU", 0.45)
    plate_use_fp16: bool = _env_bool("PLATE_USE_FP16", True)

    draw_info: bool = _env_bool("draw_info", True)
    save_plate_snapshot: bool = _env_bool("save_plate_snapshot", False)


settings = Settings()
