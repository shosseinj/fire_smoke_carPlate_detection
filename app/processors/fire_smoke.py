from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from app.core.jalali_utils import to_jalali_local_string
from app.core.types import FramePacket, TaskName, TaskResult
from app.fire_core.severity import (
    CameraRiskState,
    HazardSeverity,
    SeverityAnalyzer,
    SeverityPolicy,
    SeverityThreshold,
)
from app.fire_core.policy import FireSmokePolicyConfig
from app.fire_core.tracking import (
    Detection,
    StableObjectTracker,
    TrackerPolicy,
    TrackerSettings,
)
from app.processors.base import BatchProcessor
from app.processors.ultralytics_loader import load_yolo_class, serialized_model_load


LOGGER = logging.getLogger("uvicorn.error")


@dataclass(frozen=True, slots=True)
class FireSmokeSettings:
    model_path: Path
    device: str = "0"
    imgsz: int = 640
    batch_size: int = 8
    engine_fixed_batch: int | None = 8
    fire_class_id: int = 0
    smoke_class_id: int = 1
    fire_candidate_confidence: float = 0.30
    smoke_candidate_confidence: float = 0.30
    iou: float = 0.45
    max_detections: int = 50

    rolling_history: int = 15
    required_positive_detections: int = 8
    required_positive_ratio: float = 0.60
    consecutive_detections: int = 3
    fire_average_confidence: float = 0.45
    smoke_average_confidence: float = 0.40
    alert_release_after_missing: int = 10
    track_removal_after_missing: int = 18
    track_iou: float = 0.20
    track_center_distance: float = 0.75
    bbox_smoothing_alpha: float = 0.65
    confidence_ema_alpha: float = 0.35
    evidence_min_track_hits: int = 1

    severity_timeline_seconds: float = 3.0
    low_severity_min_count: int = 5
    low_severity_min_ratio: float = 0.0
    medium_severity_min_count: int = 10
    medium_severity_min_ratio: float = 0.0
    high_severity_min_count: int = 20
    high_severity_min_ratio: float = 0.0
    fire_low_confidence: float = 0.45
    fire_medium_confidence: float = 0.50
    fire_high_confidence: float = 0.60
    smoke_low_confidence: float = 0.40
    smoke_medium_confidence: float = 0.45
    smoke_high_confidence: float = 0.55
    demotion_hold_seconds: float = 5.0

    incident_start_severity: HazardSeverity = HazardSeverity.MEDIUM
    alert_start_severity: HazardSeverity = HazardSeverity.HIGH
    incident_end_grace_seconds: float = 10.0


@dataclass(slots=True)
class IncidentState:
    incident_id: str
    started_at_utc: str
    maximum_severity: HazardSeverity
    alert_sent: bool = False
    inactive_since: float | None = None


@dataclass(slots=True)
class PerSourceState:
    tracker: StableObjectTracker
    risk: CameraRiskState
    analyzer: SeverityAnalyzer
    incident: IncidentState | None = None
    last_frame_index: int = -1
    policy_revision: int = -1
    settings_revision: int = -1
    fire_candidate_confidence: float = 0.30
    smoke_candidate_confidence: float = 0.30


class FireSmokeProcessor(BatchProcessor):
    task = TaskName.FIRE_SMOKE

    def __init__(
        self,
        settings: FireSmokeSettings,
        *,
        model: Any | None = None,
        policy_provider: Callable[[], tuple[int, FireSmokePolicyConfig]] | None = None,
        settings_provider: Callable[[str], tuple[int, float, float]] | None = None,
        model_provider: Callable[[], tuple[int, list[Path]]] | None = None,
    ) -> None:
        self.settings = settings
        self._model = model
        self._policy_provider = policy_provider
        self._settings_provider = settings_provider
        self._model_provider = model_provider
        self._load_lock = threading.Lock()
        self._load_error: Exception | None = None
        self._model_revision = -1
        self._model_candidates = [settings.model_path]
        self._model_candidate_index = 0
        self._active_model_path = settings.model_path
        self._model_fallbacks = 0
        self._states: dict[str, PerSourceState] = {}
        self._processed_batches = 0
        self._processed_frames = 0
        self._last_inference_ms = 0.0

    def _severity_policy(
        self,
        label: str,
        dynamic: FireSmokePolicyConfig | None = None,
    ) -> SeverityPolicy:
        if label == "fire":
            low, medium, high = (
                self.settings.fire_low_confidence,
                self.settings.fire_medium_confidence,
                self.settings.fire_high_confidence,
            )
        else:
            low, medium, high = (
                self.settings.smoke_low_confidence,
                self.settings.smoke_medium_confidence,
                self.settings.smoke_high_confidence,
            )
        return SeverityPolicy(
            timeline_seconds=(
                dynamic.window_seconds if dynamic else self.settings.severity_timeline_seconds
            ),
            low=SeverityThreshold(
                dynamic.low_count if dynamic else self.settings.low_severity_min_count,
                self.settings.low_severity_min_ratio,
                low,
            ),
            medium=SeverityThreshold(
                dynamic.medium_count if dynamic else self.settings.medium_severity_min_count,
                self.settings.medium_severity_min_ratio,
                medium,
            ),
            high=SeverityThreshold(
                dynamic.high_count if dynamic else self.settings.high_severity_min_count,
                self.settings.high_severity_min_ratio,
                high,
            ),
            demotion_hold_seconds=self.settings.demotion_hold_seconds,
        )

    def _source_thresholds(self, source_id: str) -> tuple[int, float, float]:
        if self._settings_provider is None:
            return (
                -1,
                float(self.settings.fire_candidate_confidence),
                float(self.settings.smoke_candidate_confidence),
            )
        revision, fire_confidence, smoke_confidence = self._settings_provider(source_id)
        return revision, float(fire_confidence), float(smoke_confidence)

    def _new_state(self, source_id: str) -> PerSourceState:
        revision = -1
        dynamic = None
        if self._policy_provider is not None:
            revision, dynamic = self._policy_provider()
        settings_revision, fire_confidence, smoke_confidence = self._source_thresholds(source_id)
        return PerSourceState(
            tracker=StableObjectTracker(
                TrackerSettings(
                    track_iou=self.settings.track_iou,
                    track_center_distance=self.settings.track_center_distance,
                    bbox_smoothing_alpha=self.settings.bbox_smoothing_alpha,
                    confidence_ema_alpha=self.settings.confidence_ema_alpha,
                ),
                TrackerPolicy(
                    rolling_history=self.settings.rolling_history,
                    required_positive_detections=self.settings.required_positive_detections,
                    required_positive_ratio=self.settings.required_positive_ratio,
                    consecutive_detections=self.settings.consecutive_detections,
                    fire_average_confidence=self.settings.fire_average_confidence,
                    smoke_average_confidence=self.settings.smoke_average_confidence,
                    alert_release_after_missing=self.settings.alert_release_after_missing,
                    track_removal_after_missing=self.settings.track_removal_after_missing,
                ),
            ),
            risk=CameraRiskState(),
            analyzer=SeverityAnalyzer(
                fire_policy=self._severity_policy("fire", dynamic),
                smoke_policy=self._severity_policy("smoke", dynamic),
            ),
            policy_revision=revision,
            settings_revision=settings_revision,
            fire_candidate_confidence=fire_confidence,
            smoke_candidate_confidence=smoke_confidence,
        )

    def _sync_policy(self, state: PerSourceState) -> None:
        if self._policy_provider is None:
            return
        revision, dynamic = self._policy_provider()
        if revision == state.policy_revision:
            return
        state.analyzer.fire_policy = self._severity_policy("fire", dynamic)
        state.analyzer.smoke_policy = self._severity_policy("smoke", dynamic)
        state.policy_revision = revision

    def _sync_source_settings(self, state: PerSourceState, source_id: str) -> None:
        if self._settings_provider is None:
            return
        revision, fire_confidence, smoke_confidence = self._source_thresholds(source_id)
        if revision == state.settings_revision:
            return
        state.fire_candidate_confidence = fire_confidence
        state.smoke_candidate_confidence = smoke_confidence
        state.settings_revision = revision

    def _sync_model_selection(self) -> None:
        if self._model_provider is None:
            return
        revision, candidates = self._model_provider()
        if revision == self._model_revision:
            return
        if not candidates:
            raise FileNotFoundError("No usable fire/smoke model artifact is available")
        with self._load_lock:
            self._model_candidates = list(candidates)
            self._model_candidate_index = 0
            self._active_model_path = self._model_candidates[0]
            self._model = None
            self._load_error = None
            self._model_revision = revision

    def _advance_model_fallback(self, error: Exception) -> bool:
        with self._load_lock:
            self._load_error = error
            next_index = self._model_candidate_index + 1
            if next_index >= len(self._model_candidates):
                return False
            self._model_candidate_index = next_index
            self._active_model_path = self._model_candidates[next_index]
            self._model = None
            self._load_error = None
            self._model_fallbacks += 1
            return True

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            while self._model_candidate_index < len(self._model_candidates):
                path = self._model_candidates[self._model_candidate_index]
                try:
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"Fire/smoke model was not found: {path}"
                        )
                    with serialized_model_load():
                        YOLO = load_yolo_class()
                        self._model = YOLO(str(path), task="detect")
                    self._active_model_path = path
                    self._load_error = None
                    return
                except Exception as exc:
                    self._load_error = exc
                    self._model_candidate_index += 1
                    self._model_fallbacks += 1
            raise RuntimeError(
                f"All fire/smoke model candidates failed: {self._load_error}"
            )

    def _predict(self, frames: list[np.ndarray], *, conf_threshold: float) -> list[Any]:
        self._sync_model_selection()
        while True:
            self._ensure_model()
            assert self._model is not None
            real_count = len(frames)
            source = list(frames)
            fixed_batch = self.settings.engine_fixed_batch
            if self._active_model_path.suffix.lower() == ".engine" and fixed_batch:
                if real_count > fixed_batch:
                    raise ValueError(f"موتور تشخیص آتش/دود حداکثر {fixed_batch} فریم را می‌پذیرد")
                template = frames[0]
                while len(source) < fixed_batch:
                    source.append(np.zeros_like(template))

            kwargs: dict[str, Any] = {
                "source": source,
                "batch": fixed_batch or self.settings.batch_size,
                "imgsz": self.settings.imgsz,
                "conf": conf_threshold,
                "iou": self.settings.iou,
                "max_det": self.settings.max_detections,
                "classes": [self.settings.fire_class_id, self.settings.smoke_class_id],
                "device": self.settings.device,
                "augment": False,
                "rect": False,
                "stream": False,
                "verbose": False,
            }
            if self._active_model_path.suffix.lower() == ".pt" and self.settings.device != "cpu":
                kwargs["quantize"] = 16
            started = time.perf_counter()
            try:
                results = list(self._model.predict(**kwargs))
                self._last_inference_ms = (time.perf_counter() - started) * 1000.0
                if len(results) < real_count:
                    raise RuntimeError("Fire/smoke model returned fewer results than input frames")
                return results[:real_count]
            except Exception as exc:
                if not self._advance_model_fallback(exc):
                    raise

    def _extract_detections(
        self,
        result: Any,
        *,
        source_id: str,
        frame_index: int,
        fire_threshold: float,
        smoke_threshold: float,
    ) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        data = getattr(boxes, "data", None)
        if data is not None:
            rows = data.detach().cpu().numpy() if hasattr(data, "detach") else np.asarray(data)
        else:
            rows = []
            for box in boxes:
                xyxy = np.asarray(box.xyxy[0].tolist(), dtype=np.float32)
                confidence = float(box.conf[0].item())
                class_id = int(box.cls[0].item())
                rows.append([*xyxy.tolist(), confidence, class_id])
            rows = np.asarray(rows)

        detections: list[Detection] = []
        skipped_below_threshold: list[dict[str, float | str]] = []
        for row in rows:
            if len(row) < 6:
                continue
            x1, y1, x2, y2, confidence, class_value = row[:6]
            class_id = int(class_value)
            if class_id == self.settings.fire_class_id:
                label = "fire"
                threshold = fire_threshold
            elif class_id == self.settings.smoke_class_id:
                label = "smoke"
                threshold = smoke_threshold
            else:
                continue
            if float(confidence) < threshold:
                skipped_below_threshold.append(
                    {
                        "label": label,
                        "confidence": round(float(confidence), 6),
                        "threshold": round(float(threshold), 6),
                    }
                )
                continue
            detections.append(
                Detection(
                    class_id=class_id,
                    label=label,
                    confidence=float(confidence),
                    bbox=np.asarray([x1, y1, x2, y2], dtype=np.float32),
                )
            )
        if skipped_below_threshold:
            LOGGER.warning(
                "FIRE_DETECTION_FILTERED source=%s frame=%s kept=%s filtered=%s details=%s",
                source_id,
                frame_index,
                len(detections),
                len(skipped_below_threshold),
                skipped_below_threshold,
            )
        return detections

    @staticmethod
    def _area_ratio(bbox: np.ndarray, frame: np.ndarray) -> float:
        x1, y1, x2, y2 = bbox.tolist()
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return area / max(float(frame.shape[0] * frame.shape[1]), 1.0)

    def _incident_events(
        self,
        state: PerSourceState,
        severity: HazardSeverity,
        now_monotonic: float,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        now_utc = datetime.now(timezone.utc).isoformat()
        now_jalali = to_jalali_local_string(now_utc)
        if severity >= self.settings.incident_start_severity:
            if state.incident is None:
                state.incident = IncidentState(
                    incident_id=uuid.uuid4().hex,
                    started_at_utc=now_utc,
                    maximum_severity=severity,
                )
                events.append(
                    {
                        "event_type": "incident_started",
                        "incident_id": state.incident.incident_id,
                        "started_at": now_jalali,
                        "severity": severity.label,
                    }
                )
            state.incident.maximum_severity = max(state.incident.maximum_severity, severity)
            state.incident.inactive_since = None
            if severity >= self.settings.alert_start_severity and not state.incident.alert_sent:
                state.incident.alert_sent = True
                events.append(
                    {
                        "event_type": "alert_started",
                        "incident_id": state.incident.incident_id,
                        "created_at": now_jalali,
                        "severity": severity.label,
                    }
                )
        elif state.incident is not None:
            if state.incident.inactive_since is None:
                state.incident.inactive_since = now_monotonic
            elif now_monotonic - state.incident.inactive_since >= self.settings.incident_end_grace_seconds:
                events.append(
                    {
                        "event_type": "incident_ended",
                        "incident_id": state.incident.incident_id,
                        "ended_at": now_jalali,
                        "maximum_severity": state.incident.maximum_severity.label,
                    }
                )
                state.incident = None
        return events

    def _process_one(
        self,
        packet: FramePacket,
        result: Any,
        batch_ms: float,
        state: PerSourceState,
    ) -> TaskResult:
        started = time.perf_counter()
        self._sync_source_settings(state, packet.source_id)
        self._sync_policy(state)
        detections = self._extract_detections(
            result,
            source_id=packet.source_id,
            frame_index=packet.frame_index,
            fire_threshold=state.fire_candidate_confidence,
            smoke_threshold=state.smoke_candidate_confidence,
        )
        tracks, transitions = state.tracker.update(
            detections,
            packet.frame_index,
            packet.source_time_seconds or 0.0,
        )
        credible = [
            track
            for track in tracks
            if track.missed_updates == 0
            and track.total_hits >= self.settings.evidence_min_track_hits
        ]
        if detections and not credible:
            LOGGER.warning(
                "FIRE_TRACK_NOT_YET_CREDIBLE source=%s frame=%s detections=%s tracks=%s evidence_min_track_hits=%s track_state=%s",
                packet.source_id,
                packet.frame_index,
                len(detections),
                len(tracks),
                self.settings.evidence_min_track_hits,
                [
                    {
                        "track_id": track.track_id,
                        "label": track.label,
                        "total_hits": track.total_hits,
                        "missed_updates": track.missed_updates,
                        "confirmed": bool(track.confirmed),
                        "alert_active": bool(track.alert_active),
                    }
                    for track in tracks
                ],
            )
        fire_tracks = [track for track in credible if track.label == "fire"]
        smoke_tracks = [track for track in credible if track.label == "smoke"]
        fire_conf = max((track.confidence_ema for track in fire_tracks), default=0.0)
        smoke_conf = max((track.confidence_ema for track in smoke_tracks), default=0.0)
        fire_area = sum(self._area_ratio(track.bbox, packet.frame) for track in fire_tracks)
        smoke_area = sum(self._area_ratio(track.bbox, packet.frame) for track in smoke_tracks)
        risk = state.analyzer.update(
            state=state.risk,
            timestamp=packet.captured_monotonic,
            fire_confidence=fire_conf,
            smoke_confidence=smoke_conf,
            fire_area_ratio=fire_area,
            smoke_area_ratio=smoke_area,
            fire_track_count=len(fire_tracks),
            smoke_track_count=len(smoke_tracks),
            fire_positive_threshold=state.fire_candidate_confidence,
            smoke_positive_threshold=state.smoke_candidate_confidence,
        )
        overall: HazardSeverity = risk["overall"]
        events = self._incident_events(state, overall, packet.captured_monotonic)
        state.last_frame_index = packet.frame_index

        track_payload = [
            {
                "track_id": track.track_id,
                "label": track.label,
                "confidence": round(float(track.confidence_ema), 6),
                "bbox": [round(float(value), 2) for value in track.bbox.tolist()],
                "confirmed": bool(track.confirmed),
                "alert_active": bool(track.alert_active),
                "total_hits": track.total_hits,
                "missed_updates": track.missed_updates,
            }
            for track in tracks
            if track.missed_updates == 0
        ]
        detection_payload = [
            {
                "label": detection.label,
                "confidence": round(float(detection.confidence), 6),
                "bbox": [round(float(value), 2) for value in detection.bbox.tolist()],
            }
            for detection in detections
        ]
        if detections and not fire_tracks and not smoke_tracks:
            LOGGER.warning(
                "FIRE_DETECTION_NOT_ESCALATED source=%s frame=%s detections=%s fire_threshold=%.3f smoke_threshold=%.3f severity=%s fire_snapshot=%s smoke_snapshot=%s",
                packet.source_id,
                packet.frame_index,
                detection_payload,
                state.fire_candidate_confidence,
                state.smoke_candidate_confidence,
                overall.label,
                state.analyzer.snapshot_dict(risk["fire"]),
                state.analyzer.snapshot_dict(risk["smoke"]),
            )
        transition_payload = [
            {
                "status": item.status,
                "track_id": item.track_id,
                "label": item.label,
                "reason": item.reason,
            }
            for item in transitions
        ]
        data = {
            "severity": overall.label,
            "previous_severity": risk["previous_overall"].label,
            "severity_changed": bool(risk["overall_changed"]),
            "severity_window_seconds": state.analyzer.fire_policy.timeline_seconds,
            "incident_id": (
                state.incident.incident_id if state.incident is not None else None
            ),
            "fire": state.analyzer.snapshot_dict(risk["fire"]),
            "smoke": state.analyzer.snapshot_dict(risk["smoke"]),
            "tracks": track_payload,
            "detections": detection_payload,
            "track_transitions": transition_payload,
            "events": events,
            "batch_inference_ms": round(batch_ms, 3),
        }
        elapsed = (time.perf_counter() - started) * 1000.0
        return TaskResult.success(
            task=self.task,
            packet=packet,
            processing_ms=elapsed,
            data=data,
        )

    def process_batch(self, packets: Sequence[FramePacket]) -> list[TaskResult]:
        if not packets:
            return []
        states: list[PerSourceState] = []
        batch_conf = min(
            float(self.settings.fire_candidate_confidence),
            float(self.settings.smoke_candidate_confidence),
        )
        for packet in packets:
            state = self._states.get(packet.source_id)
            if state is None:
                state = self._new_state(packet.source_id)
                self._states[packet.source_id] = state
            else:
                self._sync_source_settings(state, packet.source_id)
            self._sync_policy(state)
            states.append(state)
            batch_conf = min(
                batch_conf,
                state.fire_candidate_confidence,
                state.smoke_candidate_confidence,
            )
        results = self._predict([packet.frame for packet in packets], conf_threshold=batch_conf)
        batch_ms = self._last_inference_ms
        self._processed_batches += 1
        self._processed_frames += len(packets)
        return [
            self._process_one(packet, result, batch_ms, state)
            for packet, result, state in zip(packets, results, states)
        ]

    def status(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "model_path": str(self._active_model_path),
            "model_exists": self._active_model_path.is_file(),
            "model_candidates": [str(path) for path in self._model_candidates],
            "model_revision": self._model_revision,
            "model_fallbacks": self._model_fallbacks,
            "model_loaded": self._model is not None,
            "model_load_error": str(self._load_error) if self._load_error else None,
            "processed_batches": self._processed_batches,
            "processed_frames": self._processed_frames,
            "last_inference_ms": round(self._last_inference_ms, 3),
            "tracked_sources": len(self._states),
        }
