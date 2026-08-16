from __future__ import annotations

import logging
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from app.config import Settings, settings
from app.database import Database, get_database
from app.core.auth import initialize_auth_store
from app.core.general_settings_store import GeneralSettingsStore
from app.core.source_settings_store import SourceSettingsStore
from app.core.operational_settings import (
    CAMERA_SETTINGS_METADATA_KEY,
    LEGACY_FPS_FIELDS,
    OperationalSettings,
)
from app.core.result_store import ResultStore
from app.core.broadcast import AnnotatedBroadcastHub, SourceDrawSettings
from app.core.plate_log_store import PlateLogStore
from app.core.car_plate_store import CarPlateStore
from app.core.plate_settings_store import PlateDetectionPolicy, PlateSettingsStore
from app.core.model_management import (
    ModelConversionManager,
    ModelManager,
    ModelSelectionConfig,
)
from app.core.media_preview_publisher import MediaPreviewPublisher
from app.core.live_branch import GpuLiveBranchManager
from app.core.recording_coordinator import RecordingCoordinator
from app.core.recording_executor import ScheduledRecordingExecutor
from app.core.recording_job_store import RecordingJobStore
from app.core.recording_scheduler import RecordingScheduler
from app.core.recording_storage import LocalSpoolLifecycle, RecordingStorageService, RecordingStorageSettings, SpoolSettings
from app.core.fire_smoke_log_store import FireSmokeLogStore
from app.core.human_log_store import HumanLogStore
from app.core.face_quality_store import FaceQualityPolicy, FaceQualitySettingsStore
from app.core.location_store import LocationStore
from app.core.cam_store import CamStore
from app.core.personnel_store import PersonnelStore
from app.core.employee_type_store import EmployeeTypeStore
from app.core.shift_store import ShiftStore
from app.core.holiday_store import HolidayStore
from app.core.request_store import RequestStore
from app.core.detection_log_store import DetectionLogStore
from app.core.detection_event_publisher import DetectionEventPublisher
from app.core.human_detection_event_observer import HumanDetectionEventObserver
from app.core.human_event_media_worker import HumanEventMediaWorker
from app.core.human_event_audit_store import HumanEventAuditStore
from app.core.recording_segment_store import RecordingSegment, RecordingSegmentStore
from app.core.recording_segment_dispatcher import RecordingSegmentDispatcher
from app.core.durable_event_outbox import DurableHumanEventOutbox
from app.core.recent_detection_service import (
    build_recent_detection_refresh_message,
    get_single_detection_payload_by_id,
)
from app.core.import_progress_store import ImportProgressStore
from app.core.excel_import_manager import ExcelImportManager
from app.core.personnel_zip_import_manager import PersonnelZipImportManager
from app.core.static_video_store import StaticVideoStore
from app.core.static_video_lifecycle import StaticVideoLifecycle
from app.core.init_db import init_database
from app.core.router import TaskRouter
from app.core.source_registry import SourceChange, SourceRegistry
from app.core.types import FramePacket, TaskName, TaskResult
from app.core.worker import TaskWorker
from app.core.deepstream_ingestor import DeepStreamIngestor
from app.core.video_ingestor import StaticVideoFileIngestor, VideoFileIngestor
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


def _observe_human_detection_event_once(
    observer: HumanDetectionEventObserver,
    packet: FramePacket,
    result: TaskResult,
    room_ids_by_track: dict[int, int],
    counts_for_attendance: bool,
) -> None:
    """Narrow runtime seam that keeps event observation independent of legacy persistence."""
    observer.observe(
        packet,
        result,
        room_ids_by_track=room_ids_by_track,
        counts_for_attendance=counts_for_attendance,
    )


def _build_human_event_only_observer(
    observer: HumanDetectionEventObserver,
    registry: SourceRegistry,
) -> Callable[[FramePacket, TaskResult], None]:
    def observe(packet: FramePacket, result: TaskResult) -> None:
        source = registry.get(result.source_id) if result.source_id else None
        _observe_human_detection_event_once(
            observer,
            packet,
            result,
            {},
            bool(source and source.counts_for_attendance),
        )

    return observe


def _select_detection_observers(
    enabled: bool,
    legacy_face: Callable[[FramePacket, TaskResult], None],
    event_face: Callable[[FramePacket, TaskResult], None],
    fire: Callable[[FramePacket, TaskResult], None],
    plate: Callable[[FramePacket, TaskResult], None],
    location: Callable[[FramePacket, TaskResult], None],
) -> tuple[Callable[[FramePacket, TaskResult], None] | None, ...]:
    if enabled:
        return legacy_face, fire, plate, location
    return event_face, None, None, None


def _safe_int(value: object) -> int | None:
    """Coerce a value to int, returning None for non-numeric or None inputs."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
    employee_type_store: EmployeeTypeStore
    location_store: LocationStore
    cam_store: CamStore
    shift_store: ShiftStore
    holiday_store: HolidayStore
    request_store: RequestStore
    detection_log_store: DetectionLogStore
    import_progress: ImportProgressStore
    excel_imports: ExcelImportManager
    personnel_zip_imports: PersonnelZipImportManager
    static_video_store: StaticVideoStore
    static_video_lifecycle: StaticVideoLifecycle
    general_settings: GeneralSettingsStore
    source_settings: SourceSettingsStore
    video_ingestor: VideoFileIngestor | DeepStreamIngestor | None = None
    static_video_ingestor: VideoFileIngestor | None = None
    media_preview: MediaPreviewPublisher | None = None
    live_branch: GpuLiveBranchManager | None = None
    recording_coordinator: RecordingCoordinator | None = None
    recording_redis: object | None = None
    recording_storage: RecordingStorageService | None = None
    recording_error: str | None = None
    detection_event_publisher: DetectionEventPublisher | None = None
    detection_event_redis: object | None = None
    detection_event_error: str | None = None
    human_event_media_worker: HumanEventMediaWorker | None = None
    human_event_media_redis: object | None = None
    human_event_media_error: str | None = None
    human_event_outbox: DurableHumanEventOutbox | None = None
    recording_segment_dispatcher: RecordingSegmentDispatcher | None = None
    human_event_observer: HumanDetectionEventObserver | None = None
    _closed: bool = False
    _general_teardown_complete: bool = False

    def operational_settings(self):
        gs = self.general_settings.get()
        merged = gs.operational.to_dict()
        # Override the 9 confidence fields from the sources __default__ row
        source_default = self.source_settings.get_default().to_dict()
        for field in ("fire_confidence", "smoke_confidence", "plate_confidence",
                       "plate_iou", "vehicle_confidence", "vehicle_iou",
                       "face_human_confidence", "face_detection_confidence",
                       "face_recognition_threshold"):
            merged[field] = source_default.get(field, merged[field])
        return OperationalSettings(**merged)

    def resolve_camera_settings(self, camera_id: str) -> dict[str, object]:
        general = self.operational_settings().to_dict()
        gs = self.general_settings.get()
        force = gs.force
        camera = self.registry.require(camera_id)
        overrides = dict(camera.metadata.get(CAMERA_SETTINGS_METADATA_KEY) or {})
        for field in LEGACY_FPS_FIELDS:
            overrides.pop(field, None)
        if force:
            from app.core.settings_policy import resolve_all_camera_settings
            return resolve_all_camera_settings(overrides, general, force=True)
        return {**general, **overrides}

    def update_camera_overrides(self, camera_id: str, changes: dict[str, object]) -> dict[str, object]:
        camera = self.registry.require(camera_id)
        current = dict(camera.metadata.get(CAMERA_SETTINGS_METADATA_KEY) or {})
        for field in LEGACY_FPS_FIELDS:
            current.pop(field, None)
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
        self._restart_ingestor_source(camera_id)
        return self.resolve_camera_settings(camera_id)

    def apply_operational_settings(self) -> None:
        from dataclasses import replace
        current = self.operational_settings()
        workers = self.router.workers
        plate = workers.get(TaskName.PLATE_RECOGNITION)
        if plate is not None and hasattr(plate.processor, "settings"):
            plate.processor.settings = replace(plate.processor.settings, detector_confidence=current.plate_confidence, detector_iou=current.plate_iou, vehicle_confidence=current.vehicle_confidence, vehicle_iou=current.vehicle_iou)
        face = workers.get(TaskName.FACE_RECOGNITION)
        if face is not None and hasattr(face.processor, "update_runtime_thresholds"):
            face.processor.update_runtime_thresholds(
                human_confidence=current.face_human_confidence,
                face_confidence=current.face_detection_confidence,
                recognition_threshold=current.face_recognition_threshold,
            )
        if self.video_ingestor is not None:
            if hasattr(self.video_ingestor, "rtsp_reconnect_seconds"):
                setattr(
                    self.video_ingestor,
                    "rtsp_reconnect_seconds",
                    current.rtsp_reconnect_seconds,
                )
            if hasattr(self.video_ingestor, "max_sources"):
                setattr(self.video_ingestor, "max_sources", current.rtsp_source_count)
        if self.static_video_ingestor is not None:
            if hasattr(self.static_video_ingestor, "max_sources"):
                setattr(self.static_video_ingestor, "max_sources", current.static_video_source_count)
        self.broadcast.set_enabled(current.broadcast_enabled)
        self.broadcast.jpeg_quality = current.broadcast_jpeg_quality
        self.broadcast.wall_jpeg_quality = current.broadcast_wall_jpeg_quality
        self.broadcast.wall_max_width = current.broadcast_wall_max_width
        self.broadcast.wall_max_height = current.broadcast_wall_max_height
        # Push draw_zones and refresh zone polygons per source
        gs = self.general_settings.get()
        self.broadcast.set_draw_zones(gs.draw_zones)
        self._refresh_all_source_zones()
        for camera in self.registry.list():
            self._restart_ingestor_source(camera.source_uri)

    def _release_ingestor_source(self, source_uri: str) -> None:
        """Drop any cached ingestor state for a source URI immediately."""
        if self.static_video_ingestor is not None and hasattr(self.static_video_ingestor, "release_source"):
            self.static_video_ingestor.release_source(source_uri)
        if self.video_ingestor is not None and hasattr(self.video_ingestor, "_close_source"):
            self.video_ingestor._close_source(source_uri)

    def _restart_ingestor_source(self, camera_id: str, *, previous_source_uri: str | None = None) -> None:
        """Restart the source in whichever ingestor owns it."""
        if previous_source_uri and previous_source_uri != camera_id:
            self._release_ingestor_source(previous_source_uri)
        cam = self.registry.get(camera_id)
        if cam is None:
            return
        if cam.source_type == "static_video":
            if self.static_video_ingestor is not None and hasattr(self.static_video_ingestor, "restart_source"):
                self.static_video_ingestor.restart_source(camera_id)
        else:
            if self.video_ingestor is not None and hasattr(self.video_ingestor, "restart_source"):
                self.video_ingestor.restart_source(camera_id)

    def _refresh_all_source_zones(self) -> None:
        """Push zone polygon data for every registered source to the broadcast hub."""
        for source in self.registry.list():
            if source.room_id is None:
                self.broadcast.clear_source_zones(source.source_uri)
                continue
            polygons = self.location_store.get_camera_polygons_for_room(source.room_id)
            if polygons:
                self.broadcast.set_source_zones(source.source_uri, polygons)
            else:
                self.broadcast.clear_source_zones(source.source_uri)

    def _synchronize_source_room_assignments(self) -> int:
        """Bridge legacy room.cam_id ownership to SourceRecord.room_id.

        Rooms are returned newest-first, so recreating a room for a camera makes
        that room the active polygon source for the camera after a restart.
        """
        rooms, _ = self.location_store.list_rooms(limit=1000)
        newest_active_room_by_cam: dict[int, int] = {}
        for room in rooms:
            if room.cam_id is None or not room.is_active:
                continue
            newest_active_room_by_cam.setdefault(int(room.cam_id), int(room.id))

        cameras, _ = self.cam_store.list(limit=1000)
        updated = 0
        for camera in cameras:
            room_id = newest_active_room_by_cam.get(camera.id)
            if room_id is None:
                continue
            source = self.registry.get(camera.url)
            if source is None or source.room_id == room_id:
                continue
            self.registry.update(camera.url, room_id=room_id)
            updated += 1
        return updated

    def _refresh_all_source_draw_settings(self) -> None:
        for source in self.registry.list():
            self.broadcast.set_source_draw_settings(
                source.source_uri,
                self.broadcast_source_draw_settings(source.source_uri),
            )

    def broadcast_source_draw_settings(self, source_uri: str) -> SourceDrawSettings:
        source = self.registry.require(source_uri)
        return SourceDrawSettings(
            draw_human=source.draw_human,
            draw_zone=source.draw_zone,
            draw_fire=source.draw_fire,
            draw_smoke=source.draw_smoke,
            draw_vehicle=source.draw_vehicle,
            draw_plate=source.draw_plate,
        )

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
            if self.live_branch is not None:
                self.live_branch.start()
            if self.recording_coordinator is not None:
                try:
                    self.recording_coordinator.start()
                except Exception as exc:
                    self.recording_error = type(exc).__name__
                    LOGGER.warning("RECORDING_NOT_READY error=%s", type(exc).__name__)
            if self.human_event_media_worker is not None:
                self.human_event_media_worker.start()
            if self.human_event_outbox is not None:
                self.human_event_outbox.start()
            if self.recording_segment_dispatcher is not None:
                self.recording_segment_dispatcher.start()
            if self.video_ingestor is not None:
                self.video_ingestor.start()
            if self.static_video_ingestor is not None:
                self.static_video_ingestor.start()
        except Exception:
            coordinator_stopped = True
            if self.recording_coordinator is not None:
                try:
                    self.recording_coordinator.close()
                except Exception as exc:
                    coordinator_stopped = False
                    self.recording_error = type(exc).__name__
                    LOGGER.error("RECORDING_SHUTDOWN_BLOCKED error=%s", type(exc).__name__)
            if self.media_preview is not None:
                self.media_preview.close()
            if coordinator_stopped and self.live_branch is not None:
                self.live_branch.close()
            if self.static_video_ingestor is not None:
                self.static_video_ingestor.close()
            if self.video_ingestor is not None:
                self.video_ingestor.close()
            self.router.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        media_shutdown_error = None
        observer = getattr(self, "human_event_observer", None)
        if observer is not None and not observer.close(
                getattr(getattr(self, "settings", None), "human_event_worker_shutdown_seconds", 5.0)):
            media_shutdown_error = RuntimeError("human event completions remain undurable")
        media_worker = getattr(self, "human_event_media_worker", None)
        shutdown_seconds = getattr(getattr(self, "settings", None), "human_event_worker_shutdown_seconds", 5.0)
        if media_worker is not None and not media_worker.close(shutdown_seconds):
            media_shutdown_error = RuntimeError("human event media worker is still active")
            self.human_event_media_error = "shutdown_timed_out"
        for dispatcher in (getattr(self, "human_event_outbox", None), getattr(self, "recording_segment_dispatcher", None)):
            if dispatcher is not None and not dispatcher.close(shutdown_seconds):
                media_shutdown_error = RuntimeError("durable event dispatcher is still active")
        media_redis = getattr(self, "human_event_media_redis", None)
        if media_redis is not None and hasattr(media_redis, "close"):
            media_redis.close()
            self.human_event_media_redis = None
        publisher_stopped = True
        detection_shutdown_error: Exception | None = None
        if self.detection_event_publisher is not None:
            try:
                publisher_stopped = self.detection_event_publisher.close()
                if not publisher_stopped:
                    self.detection_event_error = "shutdown_timed_out"
                    detection_shutdown_error = RuntimeError("detection event publisher is still active; Redis dependency was preserved")
            except Exception as exc:
                publisher_stopped = False
                detection_shutdown_error = exc
                self.detection_event_error = type(exc).__name__
                LOGGER.error("DETECTION_EVENT_PUBLISHER_SHUTDOWN_FAILED error=%s", type(exc).__name__)
        if publisher_stopped and self.detection_event_redis is not None and hasattr(self.detection_event_redis, "close"):
            try:
                self.detection_event_redis.close()
                self.detection_event_redis = None
                self.detection_event_error = None
            except Exception as exc:
                detection_shutdown_error = exc
                self.detection_event_error = type(exc).__name__
                LOGGER.error("DETECTION_EVENT_REDIS_SHUTDOWN_FAILED error=%s", type(exc).__name__)
        coordinator_error: Exception | None = None
        if not self._general_teardown_complete:
            # These dependencies are independent of detection-event publishing and
            # must not be closed again merely because its Redis shutdown is retried.
            self.broadcast.close()
            self.personnel_zip_imports.close()
            self.excel_imports.close()
            self.model_conversions.close()
            if self.recording_coordinator is not None:
                try:
                    self.recording_coordinator.close()
                except Exception as exc:
                    coordinator_error = exc
                    self.recording_error = type(exc).__name__
                    LOGGER.error("RECORDING_SHUTDOWN_BLOCKED error=%s", type(exc).__name__)
            if coordinator_error is None and self.recording_redis is not None and hasattr(self.recording_redis, "close"):
                self.recording_redis.close()
            if self.media_preview is not None:
                self.media_preview.close()
            if coordinator_error is None and self.live_branch is not None:
                self.live_branch.close()
            if self.static_video_ingestor is not None:
                self.static_video_ingestor.close()
            if self.video_ingestor is not None:
                self.video_ingestor.close()
            self.static_video_lifecycle.close()
            self.router.close()
            self.fire_smoke_logs.close()
            self.plate_logs.close()
            self.human_logs.close()
            self.registry.close()
            if coordinator_error is None:
                self.database.dispose()
                self._general_teardown_complete = True
        if coordinator_error is not None:
            message = "recording coordinator is still active; recording dependencies were preserved"
            if detection_shutdown_error is not None:
                message += "; detection event publisher shutdown was also incomplete"
            raise RuntimeError(message) from coordinator_error
        if detection_shutdown_error is not None:
            raise RuntimeError("detection event publisher shutdown was incomplete; Redis dependency was preserved") from detection_shutdown_error
        if media_shutdown_error is not None:
            raise RuntimeError("human event media shutdown was incomplete; unrelated teardown completed") from media_shutdown_error
        self._closed = True

    def status(self) -> dict:
        value = self.router.status()
        value["video_ingestor"] = (
            self.video_ingestor.status()
            if self.video_ingestor is not None
            else {"enabled": False, "running": False}
        )
        value["static_video_ingestor"] = (
            self.static_video_ingestor.status()
            if self.static_video_ingestor is not None
            else {"enabled": False, "running": False}
        )
        value["broadcast"] = self.broadcast.status()
        value["media_preview"] = (
            self.media_preview.status()
            if self.media_preview is not None
            else {"enabled": False, "running": False}
        )
        value["live_branch"] = (
            self.live_branch.status()
            if self.live_branch is not None
            else {"enabled": False, "branches": {}}
        )
        value["recording"] = (
            self.recording_coordinator.status()
            if self.recording_coordinator is not None
            else {"enabled": self.settings.recording_enabled, "running": False, "error": self.recording_error}
        )
        if self.recording_error is not None:
            value["recording"]["error"] = self.recording_error
        value["detection_events"] = (
            self.detection_event_publisher.status()
            if self.detection_event_publisher is not None
            else {"enabled": self.settings.detection_events_enabled, "running": False, "closed": False,
                  "published": 0, "queued": 0, "dropped": 0,
                  "retried": 0, "failed": 0, "error": self.detection_event_error}
        )
        if self.detection_event_error is not None:
            value["detection_events"]["error"] = self.detection_event_error
        value["human_event_media"] = (
            getattr(self, "human_event_media_worker", None).status()
            if getattr(self, "human_event_media_worker", None) is not None
            else {"enabled": getattr(self.settings, "human_event_media_enabled", False), "running": False}
        )
        value["human_event_media"].update({"write_enabled": getattr(self.settings, "human_event_media_write_enabled", False),
                                            "error": getattr(self, "human_event_media_error", None)})
        if getattr(self, "human_event_outbox", None) is not None:
            value["human_event_media"]["outbox"] = self.human_event_outbox.status()
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

def build_runtime(app_settings: Settings = settings) -> Runtime:
    if app_settings.human_event_media_enabled and not app_settings.detection_events_enabled:
        raise ValueError("HUMAN_EVENT_MEDIA_ENABLED requires DETECTION_EVENTS_ENABLED=true")
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
    source_settings = SourceSettingsStore(
        database,
        OperationalSettings.from_app_settings(app_settings),
    )
    operational = general_settings.get().operational
    initialize_auth_store(database, app_settings)
    registry = SourceRegistry(database)
    results = ResultStore(app_settings.recent_results_limit)
    general_record = general_settings.get()
    broadcast = AnnotatedBroadcastHub(
        enabled=operational.broadcast_enabled,
        jpeg_quality=operational.broadcast_jpeg_quality,
        wall_jpeg_quality=operational.broadcast_wall_jpeg_quality,
        wall_max_width=operational.broadcast_wall_max_width,
        wall_max_height=operational.broadcast_wall_max_height,
        source_only_render_threads=app_settings.broadcast_source_only_render_threads,
        render_threads=app_settings.broadcast_render_threads,
        face_overlay_ttl_ms=app_settings.broadcast_face_overlay_ttl_ms,
        draw_zones=general_record.draw_zones,
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
            human_pose_enabled=app_settings.face_human_pose_enabled,
            human_pose_min_keypoints=app_settings.face_human_pose_min_keypoints,
            human_pose_keypoint_confidence=app_settings.face_human_pose_keypoint_confidence,
            recognition_quality_weight=app_settings.face_recognition_quality_weight,
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
        media_root=app_settings.saved_media_path,
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
        video_fps=app_settings.fire_video_fps,
        video_max_frames=app_settings.fire_video_max_frames,
        video_update_interval_frames=app_settings.fire_video_update_interval_frames,
    )
    detection_log_store = DetectionLogStore(database)
    human_logs = HumanLogStore(
        database,
        app_settings.saved_media_path,
        queue_size=app_settings.human_media_queue_size,
        video_fps=app_settings.human_video_fps,
        video_idle_seconds=app_settings.human_video_idle_seconds,
        video_pre_roll_frames=app_settings.human_video_pre_roll_frames,
        video_post_roll_frames=app_settings.human_video_post_roll_frames,
        video_pre_roll_max_bytes=app_settings.human_video_pre_roll_max_bytes,
        snapshot_min_improvement=app_settings.human_snapshot_min_improvement,
        face_candidate_limit=app_settings.human_face_candidate_limit,
        detection_log_store=detection_log_store,
    )
    personnel_store = PersonnelStore(
        database,
        app_settings.saved_media_path,
    )
    employee_type_store = EmployeeTypeStore(database)
    location_store = LocationStore(
        database,
    )
    cam_store = CamStore(database)
    shift_store = ShiftStore(database)
    holiday_store = HolidayStore(database)
    request_store = RequestStore(database)
    import_progress = ImportProgressStore(database)
    excel_imports = ExcelImportManager(import_progress)
    static_video_store = StaticVideoStore(database)

    # Seed database with foundational records and sample data (idempotent)
    try:
        init_database(
            personnel_store=personnel_store,
            employee_type_store=employee_type_store,
            detection_log_store=detection_log_store,
            location_store=location_store,
            shift_store=shift_store,
            holiday_store=holiday_store,
            registry=registry,
            cam_store=cam_store,
            seed_sample_detections=app_settings.seed_sample_detections,
        )
    except Exception as exc:
        LOGGER.warning("INIT_DB seeding failed: %s", exc)

    if app_settings.processor_mode == "mock":
        fire_processor = MockProcessor(TaskName.FIRE_SMOKE)
        plate_processor = MockProcessor(TaskName.PLATE_RECOGNITION)
        face_processor: BatchProcessor = MockProcessor(TaskName.FACE_RECOGNITION)
    elif app_settings.processor_mode == "real":
        def fire_source_thresholds(source_id: str) -> tuple[int, float, float]:
            resolved = source_settings.resolve(source_id)
            fire_confidence = resolved.get("fire_confidence")
            smoke_confidence = resolved.get("smoke_confidence")
            return (
                source_settings.revision,
                float(app_settings.fire_confidence if fire_confidence is None else fire_confidence),
                float(app_settings.smoke_confidence if smoke_confidence is None else smoke_confidence),
            )

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
            settings_provider=fire_source_thresholds,
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
                human_pose_enabled=face_quality_policy.human_pose_enabled,
                human_pose_min_keypoints=face_quality_policy.human_pose_min_keypoints,
                human_pose_keypoint_confidence=face_quality_policy.human_pose_keypoint_confidence,
                recognition_quality_weight=face_quality_policy.recognition_quality_weight,
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
            settings_provider=source_settings.resolve,
        )
    else:
        raise ValueError("PROCESSOR_MODE must be 'real' or 'mock'")

    # ── Location observer: polygon-based detection-to-room matching ──

    def _build_location_observer(
        ls: LocationStore,
        reg: SourceRegistry,
    ) -> Callable[[FramePacket, TaskResult], None]:
        """Return a location_observer that matches non-face detections to room polygons.

        Used for PLATE_RECOGNITION and FIRE_SMOKE results.
        For FACE_RECOGNITION, see _build_face_polygon_observer which combines
        polygon matching with human log gating.
        """
        def location_observer(packet: FramePacket, result: TaskResult) -> None:
            if result.error:
                return
            if result.task not in (
                TaskName.PLATE_RECOGNITION,
                TaskName.FIRE_SMOKE,
            ):
                return
            source_id = result.source_id
            frame_index = result.frame_index
            if not source_id:
                return
            cam = reg.get(source_id)
            if cam is None:
                return
            room_id = cam.room_id
            if room_id is None:
                return
            detections: list[dict[str, object]] = []

            if result.task == TaskName.PLATE_RECOGNITION:
                plates = result.data.get("plates", [])
                for plate in plates:
                    bbox = plate.get("bbox")
                    if bbox and len(bbox) >= 4:
                        cx = (bbox[0] + bbox[2]) / 2.0
                        cy = (bbox[1] + bbox[3]) / 2.0
                        detections.append({
                            "cx": cx,
                            "cy": cy,
                            "track_id": None,
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
                            "track_id": None,
                            "personnel_id": None,
                            "detection_event_id": frame_index,
                        })

            for det in detections:
                try:
                    ls.match_detection_to_camera_rooms(
                        anchor_room_id=room_id,
                        detection_type=result.task.value,
                        detection_event_id=det["detection_event_id"],
                        bbox_center_x=det["cx"],
                        bbox_center_y=det["cy"],
                        personnel_id=det["personnel_id"],
                        camera_id=source_id,
                        track_id=det.get("track_id"),
                    )
                except Exception:
                    LOGGER.exception(
                        "Location matching failed: source=%s task=%s",
                        source_id,
                        result.task.value,
                    )

        return location_observer

    human_event_observer = HumanDetectionEventObserver(
        lambda: human_event_outbox,
        clip_padding_seconds=app_settings.human_event_clip_padding_seconds,
    )

    def _build_face_polygon_observer(
        ls: LocationStore,
        reg: SourceRegistry,
        human_log_store: HumanLogStore,
    ) -> Callable[[FramePacket, TaskResult], None]:
        """Gate face logs on containment in the camera's explicit room polygon.

        Flow:
        1. Computes human foot point ((x1+x2)/2, y2)
        2. Checks the polygon of the room assigned to the source camera.
        3. Caches evidence only for tracks currently inside that polygon.
        4. Remembers the admitting room until the track expires.
        5. Finalizes only admitted tracks with room_id and camera_id.
        6. Missing/invalid polygons and the default fallback never admit logs.
        """
        def face_observer(packet: FramePacket, result: TaskResult) -> None:
            if result.error:
                return
            source_id = result.source_id
            if not source_id:
                return
            cam = reg.get(source_id)
            if cam is None:
                return

            # Only an explicitly configured room polygon can admit evidence.
            room_id = cam.room_id
            has_polygons = bool(ls.get_camera_polygon_rooms_for_room(room_id))
            valid_room_ids_by_track: dict[int, int] = {}
            observed_room_ids_by_track: dict[int, set[int]] = {}
            has_disappeared = bool(result.data.get("disappeared_humans"))

            for human in result.data.get("humans", []):
                bbox = human.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                foot_x, foot_y = LocationStore._human_foot_point(bbox)
                track_id = human.get("track_id")

                if has_polygons:
                    # Custom polygon zones exist — full matching with DB insert
                    matches = ls.match_detection_to_camera_rooms(
                        anchor_room_id=room_id,
                        detection_type=TaskName.FACE_RECOGNITION.value,
                        detection_event_id=result.frame_index,
                        bbox_center_x=foot_x,
                        bbox_center_y=foot_y,
                        personnel_id=_safe_int(human.get("personnel_id")),
                        camera_id=source_id,
                        track_id=track_id,
                    )
                    inside_matches = [
                        match
                        for match in matches
                        if match.transition_type != "exited"
                    ]
                    if inside_matches and track_id is not None:
                        observed_room_ids_by_track[int(track_id)] = {
                            int(match.room_id) for match in inside_matches
                        }
                    inside_match = next(
                        (
                            match
                            for match in inside_matches
                            if match.transition_type == "entered"
                        ),
                        inside_matches[0] if inside_matches else None,
                    )
                    if inside_match is not None and track_id is not None:
                        valid_room_ids_by_track[int(track_id)] = int(
                            inside_match.room_id
                        )
            # Cache only evidence captured while the track is inside the room.
            try:
                _observe_human_detection_event_once(
                    human_event_observer,
                    packet,
                    result,
                    valid_room_ids_by_track,
                    cam.counts_for_attendance,
                )
            except Exception:
                LOGGER.exception(
                    "Human detection event observer failed: source=%s", source_id
                )
            try:
                human_log_store.observe_result(
                    packet,
                    result,
                    persist_human_log=False,
                    room_ids_by_track=valid_room_ids_by_track,
                    observed_room_ids_by_track=observed_room_ids_by_track,
                    counts_for_attendance=cam.counts_for_attendance,
                )
            except Exception:
                LOGGER.exception(
                    "Face evidence observer failed: source=%s", source_id
                )

            if valid_room_ids_by_track or has_disappeared:
                try:
                    human_log_store.observe_result(
                        packet,
                        result,
                        room_ids_by_track=valid_room_ids_by_track,
                        observed_room_ids_by_track=observed_room_ids_by_track,
                        counts_for_attendance=cam.counts_for_attendance,
                    )
                except Exception:
                    LOGGER.exception(
                        "Human log observer failed: source=%s", source_id
                    )

        return face_observer

    face_result_obs, fire_result_obs, plate_result_obs, location_obs = (
        _select_detection_observers(
            app_settings.legacy_detection_persistence_enabled,
            _build_face_polygon_observer(location_store, registry, human_logs),
            _build_human_event_only_observer(human_event_observer, registry),
            fire_smoke_logs.observe_result,
            plate_logs.observe_result,
            _build_location_observer(location_store, registry),
        )
    )

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
            result_observer=fire_result_obs,
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
            result_observer=plate_result_obs,
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
            # Combined observer: polygon matching + human log gating on transitions
            result_observer=face_result_obs,
            location_observer=None,
        ),
    }
    router = TaskRouter(
        registry=registry,
        workers=workers,
        result_store=results,
        play_only_callback=broadcast.publish_passthrough,
        source_only_callback=broadcast.publish_source_only,
    )
    static_video_lifecycle = StaticVideoLifecycle(static_video_store, registry, router)
    project_root = Path(__file__).resolve().parents[1]
    video_ingestor = None
    static_video_ingestor = None
    live_branch = GpuLiveBranchManager(
        publish_base=app_settings.live_branch_publish_base,
        browser_base=app_settings.live_branch_browser_base,
        enabled=app_settings.live_branch_enabled,
        grace_seconds=app_settings.live_branch_grace_seconds,
        heartbeat_timeout_seconds=app_settings.live_branch_heartbeat_timeout_seconds,
        recording_segment_seconds=app_settings.live_recording_segment_seconds,
        recording_spool_path=app_settings.continuous_recording_temp_path,
        recording_archive_path=app_settings.continuous_recording_local_path,
        camera_id_resolver=lambda source_id: (
            record.id if (record := registry.get(source_id)) is not None else None
        ),
    )
    recording_coordinator = None
    recording_redis = None
    recording_storage = None
    recording_error = None
    detection_event_publisher = None
    detection_event_redis = None
    detection_event_error = None
    if app_settings.detection_events_enabled:
        try:
            if app_settings.detection_events_queue_capacity <= 0:
                raise ValueError("DETECTION_EVENTS_QUEUE_CAPACITY must be positive")
            if not all((app_settings.detection_events_human_stream,
                        app_settings.detection_events_fire_smoke_stream,
                        app_settings.detection_events_plate_stream,
                        app_settings.detection_events_recording_segment_stream)):
                raise ValueError("detection event stream names must not be empty")
            import redis
            detection_event_redis = redis.Redis.from_url(
                app_settings.recording_redis_url, decode_responses=True,
                socket_connect_timeout=3.0, socket_timeout=5.0, retry_on_timeout=False,
            )
            detection_event_publisher = DetectionEventPublisher(
                detection_event_redis,
                {"human": app_settings.detection_events_human_stream,
                 "fire_smoke": app_settings.detection_events_fire_smoke_stream,
                 "plate": app_settings.detection_events_plate_stream,
                 "recording_segment": app_settings.detection_events_recording_segment_stream},
                queue_capacity=app_settings.detection_events_queue_capacity,
            )
        except Exception as exc:
            detection_event_error = type(exc).__name__
            if detection_event_redis is not None and hasattr(detection_event_redis, "close"):
                try:
                    detection_event_redis.close()
                except Exception as close_exc:
                    detection_event_error = f"{type(exc).__name__}; cleanup={type(close_exc).__name__}"
                    LOGGER.warning("DETECTION_EVENT_REDIS_CLEANUP_FAILED error=%s", type(close_exc).__name__)
                detection_event_redis = None
            LOGGER.warning("DETECTION_EVENT_PUBLISHER_NOT_READY error=%s", type(exc).__name__)
    if app_settings.recording_enabled:
        try:
            import redis

            recording_redis = redis.Redis.from_url(
                app_settings.recording_redis_url,
                decode_responses=True,
                socket_connect_timeout=3.0,
                socket_timeout=5.0,
                retry_on_timeout=False,
            )
            recording_redis.ping()
            recording_storage = RecordingStorageService(RecordingStorageSettings(
                endpoint=app_settings.recording_minio_endpoint,
                access_key=app_settings.recording_minio_access_key,
                secret_key=app_settings.recording_minio_secret_key,
                bucket_name=app_settings.recording_minio_bucket,
                secure=app_settings.recording_minio_secure,
                minio_retention_days=30,
            ))
            recording_store = RecordingJobStore(database)
            recording_scheduler = RecordingScheduler(recording_store, recording_redis, global_concurrency=1)
            spool = LocalSpoolLifecycle(SpoolSettings(
                app_settings.recording_spool_path, failed_retention_days=7,
                high_water_percent=app_settings.recording_spool_high_water_percent,
            ))
            app_settings.recording_spool_path.mkdir(parents=True, exist_ok=True)
            recording_executor = ScheduledRecordingExecutor(
                live_branches=live_branch, spool_path=app_settings.recording_spool_path,
            )
            recording_coordinator = RecordingCoordinator(
                recording_store, recording_scheduler, recording_executor, recording_storage, spool,
                poll_seconds=app_settings.recording_poll_seconds,
            )
        except Exception as exc:
            recording_error = type(exc).__name__
            LOGGER.warning("RECORDING_CONFIGURATION_UNAVAILABLE error=%s", type(exc).__name__)
    segment_store = RecordingSegmentStore(database)
    human_event_audit_store = None
    if app_settings.detection_events_enabled or app_settings.human_event_media_enabled:
        human_event_audit_store = HumanEventAuditStore(app_settings.human_event_media_temp_path)
    human_event_outbox = None
    recording_segment_dispatcher = None
    if app_settings.detection_events_enabled and detection_event_redis is not None:
        human_event_outbox = DurableHumanEventOutbox(
            database, detection_event_redis, app_settings.detection_events_human_stream,
            app_settings.human_event_outbox_path,
            max_spool_files=app_settings.human_event_outbox_max_files,
            audit_store=human_event_audit_store,
        )
        recording_segment_dispatcher = RecordingSegmentDispatcher(
            segment_store, detection_event_redis,
            app_settings.detection_events_recording_segment_stream,
        )
        def finalize_recording_segment(upload, width: int, height: int, fps: float, checksum: str) -> None:
            segment_id = hashlib.sha256(upload.object_name.encode("utf-8")).hexdigest()
            segment_store.upsert(RecordingSegment(
                segment_id, upload.source_id or upload.camera_id, app_settings.recording_minio_bucket,
                upload.object_name, datetime.fromisoformat(upload.start_time),
                datetime.fromisoformat(upload.end_time), width, height, fps, checksum,
            ))
        live_branch.recording_segment_finalizer = finalize_recording_segment
    human_event_media_worker = None
    human_event_media_redis = None
    human_event_media_error = None
    if app_settings.human_event_media_enabled:
        try:
            import redis
            if recording_storage is None:
                recording_storage = RecordingStorageService(RecordingStorageSettings(
                    endpoint=app_settings.recording_minio_endpoint,
                    access_key=app_settings.recording_minio_access_key,
                    secret_key=app_settings.recording_minio_secret_key,
                    bucket_name=app_settings.recording_minio_bucket,
                    secure=app_settings.recording_minio_secure,
                ))
            human_event_media_redis = redis.Redis.from_url(
                app_settings.recording_redis_url, decode_responses=True,
                socket_connect_timeout=3.0, socket_timeout=max(5.0, app_settings.human_event_media_block_ms / 1000 + 1),
                retry_on_timeout=False,
            )
            human_event_media_worker = HumanEventMediaWorker(
                human_event_media_redis, app_settings.detection_events_human_stream,
                segment_store, recording_storage, human_logs.finalize_event_media,
                group=app_settings.human_event_media_group,
                consumer=app_settings.human_event_media_consumer,
                dead_letter_stream=app_settings.human_event_media_dead_letter_stream,
                block_ms=app_settings.human_event_media_block_ms,
                claim_idle_ms=app_settings.human_event_media_claim_idle_ms,
                max_attempts=app_settings.human_event_media_max_attempts,
                max_segments=app_settings.human_event_media_max_segments,
                max_duration_seconds=app_settings.human_event_media_max_duration_seconds,
                max_temp_bytes=app_settings.human_event_media_max_temp_bytes,
                temp_root=app_settings.human_event_media_temp_path,
                local_root=app_settings.human_event_media_local_path,
                audit_store=human_event_audit_store,
                write_enabled=app_settings.human_event_media_write_enabled,
            )
        except Exception as exc:
            human_event_media_error = type(exc).__name__
            LOGGER.warning("HUMAN_EVENT_MEDIA_NOT_READY error=%s", type(exc).__name__)
    if app_settings.video_ingestion_enabled:
        common_ingestor_settings = {
            "registry": registry,
            "router": router,
            "project_root": project_root,
            "gpu_resize_enabled": app_settings.gpu_resize_enabled,
            "rtsp_transport": operational.rtsp_transport,
            "rtsp_reconnect_seconds": operational.rtsp_reconnect_seconds,
            "live_branch_manager": live_branch,
        }
        if app_settings.video_ingest_backend == "deepstream":
            print('\n\n\n\ningest video with deepstream\n\n')
            video_ingestor = DeepStreamIngestor(
                **common_ingestor_settings,
                source_type_filter="rtsp",
                loop=operational.video_loop,
                max_sources=operational.rtsp_source_count,
                rtsp_enabled=app_settings.rtsp_ingestion_enabled,
                rtsp_latency_ms=operational.deepstream_rtsp_latency_ms,
                rtsp_stall_timeout_seconds=(
                    operational.deepstream_rtsp_stall_timeout_seconds
                ),
                skip_taskless_sources=app_settings.skip_taskless_sources,
            )
        elif app_settings.video_ingest_backend == "opencv":
            video_ingestor = VideoFileIngestor(
                **common_ingestor_settings,
                source_type_filter="rtsp",
                loop=operational.video_loop,
                max_sources=operational.rtsp_source_count,
                rtsp_open_timeout_ms=operational.rtsp_open_timeout_ms,
                rtsp_read_timeout_ms=operational.rtsp_read_timeout_ms,
            )
        else:
            raise ValueError("VIDEO_INGEST_BACKEND must be 'deepstream' or 'opencv'")
        if app_settings.video_ingest_backend == "deepstream":
            # Keep static files on the GPU decode/convert path as well. The
            # OpenCV fallback copies and resizes every frame on the CPU before
            # the task workers can batch it.
            static_video_ingestor = DeepStreamIngestor(
                **common_ingestor_settings,
                source_type_filter="static_video",
                max_sources=operational.static_video_source_count,
                loop=False,
                rtsp_enabled=False,
                rtsp_latency_ms=operational.deepstream_rtsp_latency_ms,
                rtsp_stall_timeout_seconds=(
                    operational.deepstream_rtsp_stall_timeout_seconds
                ),
                skip_taskless_sources=app_settings.skip_taskless_sources,
                on_source_started=static_video_lifecycle.on_source_started,
                on_source_eos=static_video_lifecycle.on_source_eos,
                on_source_failed=static_video_lifecycle.on_source_failed,
            )
        else:
            static_video_ingestor = StaticVideoFileIngestor(
                registry=registry,
                router=router,
                project_root=project_root,
                loop=False,
                max_sources=operational.static_video_source_count,
                gpu_resize_enabled=app_settings.gpu_resize_enabled,
                on_source_started=static_video_lifecycle.on_source_started,
                on_source_eos=static_video_lifecycle.on_source_eos,
                on_source_failed=static_video_lifecycle.on_source_failed,
            )
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
    personnel_zip_imports = PersonnelZipImportManager(
        personnel_store,
        import_progress,
        face_processor,
    )
    runtime_obj = Runtime(
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
        employee_type_store=employee_type_store,
        location_store=location_store,
        cam_store=cam_store,
        shift_store=shift_store,
        holiday_store=holiday_store,
        request_store=request_store,
        detection_log_store=detection_log_store,
        import_progress=import_progress,
        excel_imports=excel_imports,
        personnel_zip_imports=personnel_zip_imports,
        static_video_store=static_video_store,
        static_video_lifecycle=static_video_lifecycle,
        general_settings=general_settings,
        source_settings=source_settings,
        video_ingestor=video_ingestor,
        static_video_ingestor=static_video_ingestor,
        media_preview=media_preview,
        live_branch=live_branch,
        recording_coordinator=recording_coordinator,
        recording_redis=recording_redis,
        recording_storage=recording_storage,
        recording_error=recording_error,
        detection_event_publisher=detection_event_publisher,
        detection_event_redis=detection_event_redis,
        detection_event_error=detection_event_error,
        human_event_media_worker=human_event_media_worker,
        human_event_media_redis=human_event_media_redis,
        human_event_media_error=human_event_media_error,
        human_event_outbox=human_event_outbox,
        recording_segment_dispatcher=recording_segment_dispatcher,
        human_event_observer=human_event_observer,
    )

    def _publish_detection_change(action: str, record: object) -> None:
        log_id = int(getattr(record, "id"))
        payload = get_single_detection_payload_by_id(runtime_obj, log_id)
        if payload is None:
            return
        runtime_obj.broadcast.publish_control_event(
            build_recent_detection_refresh_message(
                payload,
                log_id,
                reason="log_created" if action == "created" else "log_updated",
            )
        )

    detection_log_store.add_listener(_publish_detection_change)
    synchronized_rooms = runtime_obj._synchronize_source_room_assignments()
    if synchronized_rooms:
        LOGGER.info(
            "Synchronized %d source room assignment(s) from camera-owned rooms",
            synchronized_rooms,
        )
    registry.add_listener(lambda _change: runtime_obj._refresh_all_source_zones())
    registry.add_listener(lambda _change: runtime_obj._refresh_all_source_draw_settings())
    def _on_source_change(change: SourceChange) -> None:
        if change.action == "deleted":
            runtime_obj._release_ingestor_source(change.source_uri)
            return
        runtime_obj._restart_ingestor_source(
            change.source_uri,
            previous_source_uri=change.previous_source_uri,
        )

    registry.add_listener(_on_source_change)
    static_video_lifecycle.recover()
    # Push zone polygons to broadcast hub for all registered sources
    runtime_obj._refresh_all_source_zones()
    runtime_obj._refresh_all_source_draw_settings()
    return runtime_obj
