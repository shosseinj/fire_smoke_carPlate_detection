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


def _env_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.getenv(name)
    if value in {None, ""}:
        return default
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


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
    model_root_path: Path = _path("MODEL_ROOT_PATH", "weights")
    model_preferred_format: str = os.getenv(
        "MODEL_PREFERRED_FORMAT", "engine"
    ).strip().lower()
    model_allow_onnx_fallback: bool = _env_bool("MODEL_ALLOW_ONNX_FALLBACK", True)
    model_allow_pt_fallback: bool = _env_bool("MODEL_ALLOW_PT_FALLBACK", True)
    model_export_imgsz: int = _env_int("MODEL_EXPORT_IMGSZ", 640)
    model_export_batch_size: int = _env_int("MODEL_EXPORT_BATCH_SIZE", 16)
    model_export_workspace_gb: float = _env_float("MODEL_EXPORT_WORKSPACE_GB", 4.0)
    model_export_half: bool = _env_bool("MODEL_EXPORT_HALF", True)
    model_export_dynamic: bool = _env_bool("MODEL_EXPORT_DYNAMIC", True)
    model_export_timeout_seconds: int = _env_int("MODEL_EXPORT_TIMEOUT_SECONDS", 300)

    video_ingestion_enabled: bool = _env_bool("VIDEO_INGESTION_ENABLED", True)
    video_ingest_backend: str = 'deepstream' #os.getenv("VIDEO_INGEST_BACKEND", "deepstream").strip().lower()
    video_ingest_fps: float = _env_float("VIDEO_INGEST_FPS", 25.0)
    video_preview_fps: float = _env_float("VIDEO_PREVIEW_FPS", 25.0)
    video_loop: bool = _env_bool("VIDEO_LOOP", True)
    rtsp_transport: str = os.getenv("RTSP_TRANSPORT", "tcp")
    rtsp_ingestion_enabled: bool = _env_bool("RTSP_INGESTION_ENABLED", True)
    rtsp_open_timeout_ms: int = _env_int("RTSP_OPEN_TIMEOUT_MS", 20000)
    rtsp_read_timeout_ms: int = _env_int("RTSP_READ_TIMEOUT_MS", 10000)
    rtsp_reconnect_seconds: float = _env_float("RTSP_RECONNECT_SECONDS", 3.0)
    deepstream_rtsp_latency_ms: int = _env_int("DEEPSTREAM_RTSP_LATENCY_MS", 500)
    deepstream_rtsp_stall_timeout_seconds: int = _env_int(
        "DEEPSTREAM_RTSP_STALL_TIMEOUT_SECONDS", 30
    )
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
    vehicle_detector_weights: Path = _path(
        "VEHICLE_DETECTOR_WEIGHTS", "weights/vehicle_detector/yolo11n.pt"
    )
    plate_recognizer_dir: Path = _path("PLATE_RECOGNIZER_DIR", "weights/plate_recognizer")
    plate_device: str = os.getenv("PLATE_DEVICE", "0")
    plate_batch_size: int = _env_int("PLATE_BATCH_SIZE", 8)
    plate_max_wait_ms: float = _env_float("PLATE_MAX_WAIT_MS", 25.0)
    plate_imgsz: int = _env_int("PLATE_IMGSZ", 640)
    plate_confidence: float = _env_float("PLATE_CONFIDENCE", 0.30)
    plate_iou: float = _env_float("PLATE_IOU", 0.45)
    plate_crop_batch_size: int = _env_int("PLATE_CROP_BATCH_SIZE", 16)
    plate_class_ids: tuple[int, ...] = _env_int_tuple("PLATE_CLASS_IDS", (0,))
    vehicle_confidence: float = _env_float("VEHICLE_CONFIDENCE", 0.35)
    vehicle_iou: float = _env_float("VEHICLE_IOU", 0.45)
    vehicle_imgsz: int = _env_int("VEHICLE_IMGSZ", 640)
    vehicle_max_per_frame: int = _env_int("VEHICLE_MAX_PER_FRAME", 12)
    vehicle_class_ids: tuple[int, ...] = _env_int_tuple(
        "VEHICLE_CLASS_IDS", (2, 3, 5, 7)
    )
    vehicle_crop_padding_ratio: float = _env_float(
        "VEHICLE_CROP_PADDING_RATIO", 0.05
    )
    plate_ocr_confidence: float = _env_float("PLATE_OCR_CONFIDENCE", 0.50)
    min_vehicle_width_pixels: int = _env_int("MIN_VEHICLE_WIDTH_PIXELS", 120)
    min_vehicle_height_pixels: int = _env_int("MIN_VEHICLE_HEIGHT_PIXELS", 80)
    min_vehicle_area_ratio: float = _env_float("MIN_VEHICLE_AREA_RATIO", 0.025)
    plate_use_fp16: bool = _env_bool("PLATE_USE_FP16", True)

    face_human_model_path: Path = _path(
        "FACE_HUMAN_MODEL", "weights/face_recognition/yolo26s-pose_batch8.pt"
    )
    face_detector_model_path: Path = _path(
        "FACE_DETECTOR_MODEL", "weights/face_recognition/yolov8n-face_batch8.pt"
    )
    face_embedding_model_path: Path = _path(
        "FACE_EMBEDDING_MODEL", "weights/face_recognition/arcface_fp16.onnx"
    )
    face_device: str = os.getenv("FACE_DEVICE", "0")
    face_batch_size: int = _env_int("FACE_BATCH_SIZE", 8)
    face_max_wait_ms: float = _env_float("FACE_MAX_WAIT_MS", 25.0)
    face_human_imgsz: int = _env_int("FACE_HUMAN_IMGSZ", 640)
    face_detector_imgsz: int = _env_int("FACE_DETECTOR_IMGSZ", 640)
    face_human_engine_fixed_batch: int = _env_int("FACE_HUMAN_ENGINE_FIXED_BATCH", 8)
    face_detector_engine_fixed_batch: int = _env_int("FACE_DETECTOR_ENGINE_FIXED_BATCH", 8)
    face_human_confidence: float = _env_float("FACE_HUMAN_CONFIDENCE", 0.40)
    face_detection_confidence: float = _env_float("FACE_DETECTION_CONFIDENCE", 0.50)
    face_recognition_threshold: float = _env_float("FACE_RECOGNITION_THRESHOLD", 0.45)
    face_min_width: int = _env_int(
        "FACE_MIN_WIDTH", _env_int("FACE_MIN_SIZE", 24)
    )
    face_min_height: int = _env_int(
        "FACE_MIN_HEIGHT", _env_int("FACE_MIN_SIZE", 24)
    )
    face_blur_threshold: float = _env_float("FACE_BLUR_THRESHOLD", 20.0)
    face_min_eye_distance: float = _env_float("FACE_MIN_EYE_DISTANCE", 8.0)
    face_quality_threshold: float = _env_float("FACE_QUALITY_THRESHOLD", 0.55)
    face_max_abs_yaw: float = _env_float("FACE_MAX_ABS_YAW", 45.0)
    face_max_abs_pitch: float = _env_float("FACE_MAX_ABS_PITCH", 55.0)
    face_max_abs_roll: float = _env_float("FACE_MAX_ABS_ROLL", 35.0)
    face_require_landmarks: bool = _env_bool("FACE_REQUIRE_LANDMARKS", True)
    face_tracker_high_threshold: float = _env_float(
        "FACE_TRACKER_HIGH_THRESHOLD", 0.40
    )
    face_tracker_low_threshold: float = _env_float(
        "FACE_TRACKER_LOW_THRESHOLD", 0.10
    )
    face_tracker_new_threshold: float = _env_float(
        "FACE_TRACKER_NEW_THRESHOLD", 0.40
    )
    face_tracker_match_threshold: float = _env_float(
        "FACE_TRACKER_MATCH_THRESHOLD", 0.80
    )
    face_tracker_max_missed: int = _env_int("FACE_TRACKER_MAX_MISSED", 30)
    face_history_size: int = _env_int("FACE_HISTORY_SIZE", 30)
    face_stable_min_hits: int = _env_int("FACE_STABLE_MIN_HITS", 3)
    face_embedding_batch_size: int = _env_int("FACE_EMBEDDING_BATCH_SIZE", 64)
    face_vector_size: int = _env_int("FACE_VECTOR_SIZE", 512)
    face_qdrant_collection: str = os.getenv("FACE_QDRANT_COLLECTION", "faces")
    face_qdrant_url: str | None = os.getenv("FACE_QDRANT_URL") or None
    face_qdrant_path: Path = _path("FACE_QDRANT_PATH", "data/qdrant")
    face_qdrant_api_key: str | None = os.getenv("FACE_QDRANT_API_KEY") or None

    human_video_fps: float = _env_float("HUMAN_VIDEO_FPS", 10.0)
    human_video_idle_seconds: float = _env_float("HUMAN_VIDEO_IDLE_SECONDS", 5.0)
    human_snapshot_min_improvement: float = _env_float(
        "HUMAN_SNAPSHOT_MIN_IMPROVEMENT", 0.01
    )

    draw_info: bool = _env_bool("draw_info", True)
    save_plate_snapshot: bool = _env_bool("save_plate_snapshot", True)


settings = Settings()
