from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_URL = (
    "postgresql+psycopg2://postgres:Asd12345@host.docker.internal:5432/ai_database"
)


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




def _env_first(names: tuple[str, ...], default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value not in {None, ""}:
            return value
    return default


def _env_int_fallback(primary: str, fallback: str, default: int) -> int:
    value = _env_first((primary, fallback))
    return int(value) if value not in {None, ""} else default


def _refresh_minutes() -> int:
    direct = os.getenv("JWT_REFRESH_EXPIRY_MINUTES")
    if direct not in {None, ""}:
        return int(direct)
    legacy_days = os.getenv("REFRESH_TOKEN_EXPIRE_DAYS")
    if legacy_days not in {None, ""}:
        return int(legacy_days) * 24 * 60
    return 10080

def _path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default)).expanduser()
    return value if value.is_absolute() else (ROOT / value).resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Unified Video AI Task Router")
    processor_mode: str = os.getenv("PROCESSOR_MODE", "real").strip().lower()
    database_url: str = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    database_echo: bool = _env_bool("DATABASE_ECHO", False)
    database_pool_size: int = _env_int("DATABASE_POOL_SIZE", 10)
    database_max_overflow: int = _env_int("DATABASE_MAX_OVERFLOW", 20)
    seed_sample_detections: bool = _env_bool("SEED_SAMPLE_DETECTIONS", False)
    data_path: Path = _path("DATA_PATH", "data")
    business_timezone_name: str = os.getenv("BUSINESS_TIMEZONE", "Asia/Tehran")
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
    video_ingest_backend: str = os.getenv(
        "VIDEO_INGEST_BACKEND", "deepstream"
    ).strip().lower()
    video_loop: bool = _env_bool("VIDEO_LOOP", True)
    gpu_resize_enabled: bool = _env_bool("GPU_RESIZE_ENABLED", True)
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
    broadcast_wall_jpeg_quality: int = _env_int("BROADCAST_WALL_JPEG_QUALITY", 70)
    broadcast_wall_max_width: int = _env_int("BROADCAST_WALL_MAX_WIDTH", 320)
    broadcast_wall_max_height: int = _env_int("BROADCAST_WALL_MAX_HEIGHT", 320)
    broadcast_source_only_render_threads: int = _env_int(
        "BROADCAST_SOURCE_ONLY_RENDER_THREADS", 4
    )
    broadcast_render_threads: int = _env_int("BROADCAST_RENDER_THREADS", 2)
    broadcast_face_overlay_ttl_ms: float = _env_float(
        "BROADCAST_FACE_OVERLAY_TTL_MS", 750.0
    )
    media_preview_enabled: bool = _env_bool("MEDIA_PREVIEW_ENABLED", False)
    media_preview_publish_base: str = os.getenv(
        "MEDIA_PREVIEW_PUBLISH_BASE", "rtsp://mediamtx:8554"
    ).strip()
    media_preview_whep_base_url: str = os.getenv(
        "MEDIA_PREVIEW_WHEP_BASE_URL", ""
    ).rstrip("/")
    saved_media_path: Path = _path("SAVED_MEDIA_PATH", "saved_media/personnel")
    static_video_upload_path: Path = _path("STATIC_VIDEO_UPLOAD_PATH", "saved_media/static_videos")
    fire_severity_window_seconds: float = _env_float(
        "FIRE_SEVERITY_WINDOW_SECONDS", 3.0
    )
    fire_low_incident_count: int = _env_int("FIRE_LOW_INCIDENT_COUNT", 5)
    fire_medium_incident_count: int = _env_int("FIRE_MEDIUM_INCIDENT_COUNT", 10)
    fire_high_incident_count: int = _env_int("FIRE_HIGH_INCIDENT_COUNT", 20)
    fire_video_fps: float = _env_float("FIRE_VIDEO_FPS", 5.0)
    fire_video_max_frames: int = _env_int("FIRE_VIDEO_MAX_FRAMES", 30)
    fire_video_update_interval_frames: int = _env_int("FIRE_VIDEO_UPDATE_INTERVAL_FRAMES", 5)

    fire_model_path: Path = _path(
        "FIRE_SMOKE_MODEL_PATH",
        "weights/fire_smoke/linux_trt10/best_nano_111_dynamic_b26_trt107.engine",
    )
    fire_device: str = os.getenv("FIRE_SMOKE_DEVICE", "0")
    fire_batch_size: int = _env_int("FIRE_BATCH_SIZE", 26)
    fire_max_wait_ms: float = _env_float("FIRE_MAX_WAIT_MS", 60.0)
    fire_imgsz: int = _env_int("FIRE_SMOKE_IMGSZ", 640)
    fire_engine_fixed_batch: int = _env_int("FIRE_SMOKE_ENGINE_FIXED_BATCH", 0)
    fire_confidence: float = _env_float("FIRE_CONFIDENCE", 0.30)
    smoke_confidence: float = _env_float("SMOKE_CONFIDENCE", 0.30)
    fire_low_severity_confidence: float = _env_float("FIRE_LOW_SEVERITY_CONFIDENCE", 0.45)
    fire_medium_severity_confidence: float = _env_float("FIRE_MEDIUM_SEVERITY_CONFIDENCE", 0.50)
    fire_high_severity_confidence: float = _env_float("FIRE_HIGH_SEVERITY_CONFIDENCE", 0.60)
    smoke_low_severity_confidence: float = _env_float("SMOKE_LOW_SEVERITY_CONFIDENCE", 0.40)
    smoke_medium_severity_confidence: float = _env_float("SMOKE_MEDIUM_SEVERITY_CONFIDENCE", 0.45)
    smoke_high_severity_confidence: float = _env_float("SMOKE_HIGH_SEVERITY_CONFIDENCE", 0.55)

    plate_detector_weights: Path = _path(
        "PLATE_DETECTOR_WEIGHTS",
        "weights/plate_detector/model_dynamic_b26_trt107.engine",
    )
    vehicle_detector_weights: Path = _path(
        "VEHICLE_DETECTOR_WEIGHTS",
        "weights/vehicle_detector/yolo11n_dynamic_b26_trt107.engine",
    )
    plate_recognizer_dir: Path = _path("PLATE_RECOGNIZER_DIR", "weights/plate_recognizer")
    plate_device: str = os.getenv("PLATE_DEVICE", "0")
    plate_batch_size: int = _env_int("PLATE_BATCH_SIZE", 26)
    plate_max_wait_ms: float = _env_float("PLATE_MAX_WAIT_MS", 60.0)
    plate_log_queue_size: int = _env_int("PLATE_LOG_QUEUE_SIZE", 128)
    plate_imgsz: int = _env_int("PLATE_IMGSZ", 640)
    plate_confidence: float = _env_float("PLATE_CONFIDENCE", 0.30)
    plate_iou: float = _env_float("PLATE_IOU", 0.45)
    plate_crop_batch_size: int = _env_int("PLATE_CROP_BATCH_SIZE", 26)
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
        "FACE_HUMAN_MODEL",
        "weights/face_recognition/linux_trt10/yolo26s-pose_dynamic_b8_trt107.engine",
    )
    face_detector_model_path: Path = _path(
        "FACE_DETECTOR_MODEL",
        "weights/face_recognition/linux_trt10/yolov8n-face_dynamic_b8_trt107.engine",
    )
    face_embedding_model_path: Path = _path(
        "FACE_EMBEDDING_MODEL",
        "weights/face_recognition/linux_trt10/arcface_dynamic_b64_trt107.engine",
    )
    face_device: str = os.getenv("FACE_DEVICE", "0")
    face_batch_size: int = _env_int("FACE_BATCH_SIZE", 8)
    face_max_wait_ms: float = _env_float("FACE_MAX_WAIT_MS", 60.0)
    face_human_imgsz: int = _env_int("FACE_HUMAN_IMGSZ", 640)
    face_detector_imgsz: int = _env_int("FACE_DETECTOR_IMGSZ", 640)
    face_human_engine_fixed_batch: int = _env_int("FACE_HUMAN_ENGINE_FIXED_BATCH", 0)
    face_detector_engine_fixed_batch: int = _env_int("FACE_DETECTOR_ENGINE_FIXED_BATCH", 0)
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
    face_human_pose_enabled: bool = _env_bool("FACE_HUMAN_POSE_ENABLED", True)
    face_human_pose_min_keypoints: int = _env_int("FACE_HUMAN_POSE_MIN_KEYPOINTS", 4)
    face_human_pose_keypoint_confidence: float = _env_float(
        "FACE_HUMAN_POSE_KEYPOINT_CONFIDENCE", 0.25
    )
    face_recognition_quality_weight: float = _env_float(
        "FACE_RECOGNITION_QUALITY_WEIGHT", 0.5
    )
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
    max_images_per_request: int = _env_int("MAX_IMAGES_PER_REQUEST", 10)
    max_upload_bytes_per_image: int = _env_int("MAX_UPLOAD_BYTES_PER_IMAGE", 10 * 1024 * 1024)
    max_decoded_width: int = _env_int("MAX_DECODED_WIDTH", 4096)
    max_decoded_height: int = _env_int("MAX_DECODED_HEIGHT", 4096)
    max_total_decoded_pixels: int = _env_int("MAX_TOTAL_DECODED_PIXELS", 16_000_000)
    supported_image_extensions: tuple[str, ...] = _env_int_tuple(
        "SUPPORTED_IMAGE_EXTENSIONS", (".jpg", ".jpeg", ".png", ".bmp")
    )
    store_cropped_face: bool = _env_bool("STORE_CROPPED_FACE", False)
    face_vector_size: int = _env_int("FACE_VECTOR_SIZE", 512)
    face_qdrant_collection: str = os.getenv("FACE_QDRANT_COLLECTION", "faces")
    face_qdrant_url: str | None = os.getenv("FACE_QDRANT_URL") or None
    face_qdrant_api_key: str | None = os.getenv("FACE_QDRANT_API_KEY") or None

    human_media_queue_size: int = _env_int("HUMAN_MEDIA_QUEUE_SIZE", 256)
    human_video_fps: float = _env_float("HUMAN_VIDEO_FPS", 5.0)
    human_video_idle_seconds: float = _env_float("HUMAN_VIDEO_IDLE_SECONDS", 5.0)
    human_snapshot_min_improvement: float = _env_float(
        "HUMAN_SNAPSHOT_MIN_IMPROVEMENT", 0.01
    )
    human_face_candidate_limit: int = _env_int("HUMAN_FACE_CANDIDATE_LIMIT", 5)

    draw_info: bool = _env_bool("draw_info", True)
    save_plate_snapshot: bool = _env_bool("save_plate_snapshot", True)

    # Performance tuning
    worker_threads: int = _env_int("WORKER_THREADS", 1)
    task_queue_policy: str = os.getenv("TASK_QUEUE_POLICY", "latest_per_source").strip().lower()
    task_queue_capacity: int = _env_int("TASK_QUEUE_CAPACITY", 512)
    task_queue_block_timeout_ms: float = _env_float("TASK_QUEUE_BLOCK_TIMEOUT_MS", 500.0)
    skip_taskless_sources: bool = _env_bool("SKIP_TASKLESS_SOURCES", False)

    # Ingestion tuning
    rtsp_source_count: int = _env_int("RTSP_SOURCE_COUNT", 512)
    static_video_source_count: int = _env_int("STATIC_VIDEO_SOURCE_COUNT", 256)

    # Authentication / JWT. Current names take precedence; legacy names are fallbacks.
    jwt_secret_key: str = str(
        _env_first(
            ("JWT_SECRET_KEY", "SECRET_KEY"),
            "change-me-in-production-use-a-strong-random-secret",
        )
    )
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_expiry_minutes: int = _env_int_fallback(
        "JWT_EXPIRY_MINUTES", "ACCESS_TOKEN_EXPIRE_MINUTES", 1440
    )
    jwt_refresh_expiry_minutes: int = _refresh_minutes()
    auth_default_admin_username: str = str(
        _env_first(("AUTH_DEFAULT_ADMIN_USERNAME", "SUPERUSER_USERNAME"), "superadmin")
    )
    auth_default_admin_password: str = str(
        _env_first(("AUTH_DEFAULT_ADMIN_PASSWORD", "SUPERUSER_PASSWORD"), "SuperAdmin123!")
    )
    auth_default_admin_email: str | None = _env_first(("AUTH_DEFAULT_ADMIN_EMAIL", "SUPERUSER_EMAIL"))
    auth_login_max_attempts: int = _env_int("AUTH_LOGIN_MAX_ATTEMPTS", 5)
    auth_login_lockout_minutes: int = _env_int("AUTH_LOGIN_LOCKOUT_MINUTES", 15)
    disable_auth: bool = _env_bool("DISABLE_AUTH", False)


settings = Settings()
