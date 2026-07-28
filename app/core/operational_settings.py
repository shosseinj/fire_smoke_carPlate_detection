from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from app.config import Settings


@dataclass(frozen=True, slots=True)
class OperationalSettings:
    video_loop: bool = True
    rtsp_transport: str = "tcp"
    rtsp_open_timeout_ms: int = 20000
    rtsp_read_timeout_ms: int = 10000
    rtsp_reconnect_seconds: float = 3.0
    deepstream_rtsp_latency_ms: int = 500
    deepstream_rtsp_stall_timeout_seconds: int = 30
    broadcast_enabled: bool = True
    broadcast_jpeg_quality: int = 82
    broadcast_wall_jpeg_quality: int = 70
    broadcast_wall_max_width: int = 320
    broadcast_wall_max_height: int = 320
    fire_confidence: float = 0.30
    smoke_confidence: float = 0.30
    plate_confidence: float = 0.30
    plate_iou: float = 0.45
    vehicle_confidence: float = 0.35
    vehicle_iou: float = 0.45
    face_human_confidence: float = 0.40
    face_detection_confidence: float = 0.50
    face_recognition_threshold: float = 0.45
    rtsp_source_count: int = 256
    static_video_source_count: int = 256

    @classmethod
    def from_app_settings(cls, settings: Settings) -> "OperationalSettings":
        return cls(**{name: getattr(settings, name) for name in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def updated(self, changes: dict[str, Any]) -> "OperationalSettings":
        unknown = set(changes) - set(self.__dataclass_fields__)
        if unknown:
            raise ValueError(f"تنظیمات عملیاتی نامعتبر: {', '.join(sorted(unknown))}")
        candidate = replace(self, **changes)
        candidate.validate()
        return candidate

    def validate(self) -> None:
        for name in ("fire_confidence", "smoke_confidence", "plate_confidence", "plate_iou", "vehicle_confidence", "vehicle_iou", "face_human_confidence", "face_detection_confidence", "face_recognition_threshold"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.rtsp_transport not in {"tcp", "udp", "udp_multicast", "http"}:
            raise ValueError("rtsp_transport must be tcp, udp, udp_multicast, or http")
        for name in ("rtsp_open_timeout_ms", "rtsp_read_timeout_ms", "deepstream_rtsp_latency_ms", "deepstream_rtsp_stall_timeout_seconds", "broadcast_wall_max_width", "broadcast_wall_max_height"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        for name in ("rtsp_source_count", "static_video_source_count"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.rtsp_reconnect_seconds < 0.5:
            raise ValueError("rtsp_reconnect_seconds must be at least 0.5")
        for name in ("broadcast_jpeg_quality", "broadcast_wall_jpeg_quality"):
            if not 1 <= int(getattr(self, name)) <= 100:
                raise ValueError(f"{name} must be between 1 and 100")


OVERRIDABLE_FIELDS = frozenset(OperationalSettings.__dataclass_fields__)
CAMERA_SETTINGS_METADATA_KEY = "_settings_overrides"
LEGACY_FPS_FIELDS = frozenset({"video_ingest_fps", "video_preview_fps"})
