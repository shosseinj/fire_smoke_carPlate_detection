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
    camera_db_path: Path = _path("CAMERA_DB_PATH", "data/cameras.sqlite3")
    source_registry_path: Path = _path("SOURCE_REGISTRY_PATH", "data/sources.json")
    recent_results_limit: int = _env_int("RECENT_RESULTS_LIMIT", 2000)

    video_ingestion_enabled: bool = _env_bool("VIDEO_INGESTION_ENABLED", True)
    video_ingest_backend: str = 'deepstream' #os.getenv("VIDEO_INGEST_BACKEND", "deepstream").strip().lower()
    video_ingest_fps: float = _env_float("VIDEO_INGEST_FPS", 5.0)
    video_loop: bool = _env_bool("VIDEO_LOOP", True)
    rtsp_transport: str = os.getenv("RTSP_TRANSPORT", "tcp")
    rtsp_ingestion_enabled: bool = _env_bool("RTSP_INGESTION_ENABLED", True)
    rtsp_open_timeout_ms: int = _env_int("RTSP_OPEN_TIMEOUT_MS", 20000)
    rtsp_read_timeout_ms: int = _env_int("RTSP_READ_TIMEOUT_MS", 10000)
    rtsp_reconnect_seconds: float = _env_float("RTSP_RECONNECT_SECONDS", 3.0)
    deepstream_rtsp_latency_ms: int = _env_int("DEEPSTREAM_RTSP_LATENCY_MS", 500)
    broadcast_enabled: bool = _env_bool("BROADCAST_ENABLED", True)
    broadcast_jpeg_quality: int = _env_int("BROADCAST_JPEG_QUALITY", 82)
    plate_log_db_path: Path = _path("PLATE_LOG_DB_PATH", "plate_logs.sqlite3")
    saved_media_path: Path = _path("SAVED_MEDIA_PATH", "saved_media")
    fire_severity_window_seconds: float = _env_float(
        "FIRE_SEVERITY_WINDOW_SECONDS", 3.0
    )
    fire_low_incident_count: int = _env_int("FIRE_LOW_INCIDENT_COUNT", 5)
    fire_medium_incident_count: int = _env_int("FIRE_MEDIUM_INCIDENT_COUNT", 10)
    fire_high_incident_count: int = _env_int("FIRE_HIGH_INCIDENT_COUNT", 20)

    fire_model_path: Path = _path("FIRE_SMOKE_MODEL_PATH", "weights/fire_smoke/best_nano_111.pt")
    fire_device: str = os.getenv("FIRE_SMOKE_DEVICE", "0")
    fire_batch_size: int = _env_int("FIRE_SMOKE_BATCH_SIZE", 8)
    fire_max_wait_ms: float = _env_float("FIRE_SMOKE_MAX_WAIT_MS", 25.0)
    fire_imgsz: int = _env_int("FIRE_SMOKE_IMGSZ", 640)
    fire_engine_fixed_batch: int = _env_int("FIRE_SMOKE_ENGINE_FIXED_BATCH", 8)
    fire_confidence: float = _env_float("FIRE_CONFIDENCE", 0.30)
    smoke_confidence: float = _env_float("SMOKE_CONFIDENCE", 0.30)

    plate_detector_weights: Path = _path("PLATE_DETECTOR_WEIGHTS", "weights/plate_detector/model.pt")
    plate_recognizer_dir: Path = _path("PLATE_RECOGNIZER_DIR", "weights/plate_recognizer")
    plate_device: str = os.getenv("PLATE_DEVICE", "0")
    plate_batch_size: int = _env_int("PLATE_BATCH_SIZE", 8)
    plate_max_wait_ms: float = _env_float("PLATE_MAX_WAIT_MS", 25.0)
    plate_imgsz: int = _env_int("PLATE_IMGSZ", 640)
    plate_confidence: float = _env_float("PLATE_CONFIDENCE", 0.30)
    plate_iou: float = _env_float("PLATE_IOU", 0.45)
    plate_use_fp16: bool = _env_bool("PLATE_USE_FP16", True)

    draw_info: bool = _env_bool("draw_info", True)
    save_plate_snapshot: bool = _env_bool("save_plate_snapshot", True)


settings = Settings()
