from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from app.core.types import FramePacket, TaskName, TaskResult
from app.processors.base import BatchProcessor
from app.processors.ultralytics_loader import (
    load_hezar_model_class,
    load_yolo_class,
    serialized_model_load,
)

PERSIAN_TO_LATIN_DIGITS = str.maketrans(
    {
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
        "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    }
)
LATIN_TO_PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def normalize_plate_text(text: str, output_persian_digits: bool = False) -> str:
    normalized = str(text).strip().translate(PERSIAN_TO_LATIN_DIGITS)
    normalized = normalized.replace("ي", "ی").replace("ك", "ک")
    normalized = re.sub(r"[\s\-_.:/\\|]+", "", normalized)
    normalized = re.sub(r"[^0-9A-Za-zآ-ی♿]", "", normalized)
    return normalized.translate(LATIN_TO_PERSIAN_DIGITS) if output_persian_digits else normalized


@dataclass(frozen=True, slots=True)
class PlateSettings:
    detector_weights: Path
    vehicle_detector_weights: Path
    recognizer_model_dir: Path
    device: str = "0"
    detector_imgsz: int = 640
    detector_confidence: float = 0.30
    detector_iou: float = 0.45
    plate_crop_batch_size: int = 16
    use_fp16: bool = True
    plate_class_ids: tuple[int, ...] = ()
    vehicle_class_ids: tuple[int, ...] = (2, 3, 5, 7)
    vehicle_confidence: float = 0.35
    vehicle_iou: float = 0.45
    vehicle_imgsz: int = 640
    vehicle_max_per_frame: int = 12
    vehicle_crop_padding_ratio: float = 0.05
    output_persian_digits: bool = False


class PlateRecognitionProcessor(BatchProcessor):
    task = TaskName.PLATE_RECOGNITION
    REQUIRED_RECOGNIZER_FILES = (
        Path("model.pt"),
        Path("model_config.yaml"),
        Path("preprocessor") / "image_processor_config.yaml",
    )

    def __init__(
        self,
        settings: PlateSettings,
        *,
        vehicle_detector: Any | None = None,
        detector: Any | None = None,
        recognizer: Any | None = None,
    ) -> None:
        self.settings = settings
        self._vehicle_detector = vehicle_detector
        self._detector = detector
        self._recognizer = recognizer
        self._load_lock = threading.Lock()
        self._load_error: Exception | None = None
        self._processed_batches = 0
        self._processed_frames = 0
        self._last_inference_ms = 0.0
        self._last_vehicle_inference_ms = 0.0
        self._last_plate_inference_ms = 0.0

    def recognizer_missing_files(self) -> list[Path]:
        root = self.settings.recognizer_model_dir
        return [root / relative for relative in self.REQUIRED_RECOGNIZER_FILES if not (root / relative).is_file()]

    def _torch_device(self) -> str:
        configured = self.settings.device.strip().lower()
        if configured == "cpu" or configured.startswith("cuda"):
            return configured
        if configured.isdigit():
            return f"cuda:{configured}"
        return configured

    def _ensure_models(self) -> None:
        if (
            self._vehicle_detector is not None
            and self._detector is not None
            and self._recognizer is not None
        ):
            return
        if self._load_error is not None:
            raise RuntimeError(f"Plate models loading previously failed: {self._load_error}")
        with self._load_lock:
            if (
                self._vehicle_detector is not None
                and self._detector is not None
                and self._recognizer is not None
            ):
                return
            with serialized_model_load():
                try:
                    if not self.settings.detector_weights.is_file():
                        raise FileNotFoundError(
                            f"Plate detector checkpoint was not found: {self.settings.detector_weights}"
                        )
                    if not self.settings.vehicle_detector_weights.is_file():
                        raise FileNotFoundError(
                            "Vehicle detector checkpoint was not found: "
                            f"{self.settings.vehicle_detector_weights}"
                        )
                    if not self.settings.recognizer_model_dir.is_dir():
                        raise FileNotFoundError(
                            f"Plate recognizer directory was not found: {self.settings.recognizer_model_dir}"
                        )
                    missing = self.recognizer_missing_files()
                    if missing:
                        raise FileNotFoundError(
                            "Plate recognizer directory is incomplete. Missing: "
                            + ", ".join(str(path) for path in missing)
                        )
                    YOLO = load_yolo_class()
                    Model = load_hezar_model_class()
                    vehicle_detector = YOLO(
                        str(self.settings.vehicle_detector_weights)
                    )
                    detector = YOLO(str(self.settings.detector_weights))
                    recognizer = Model.load(
                        str(self.settings.recognizer_model_dir), load_locally=True
                    )
                    recognizer.eval()
                    recognizer.to(self._torch_device())
                    self._vehicle_detector = vehicle_detector
                    self._detector = detector
                    self._recognizer = recognizer
                except Exception as exc:
                    self._load_error = exc
                    raise

    @staticmethod
    def _normalize_class_name(value: Any) -> str:
        return str(value).strip().lower().replace("-", "_").replace(" ", "_")

    def _plate_class_ids(self) -> set[int]:
        if self.settings.plate_class_ids:
            return set(self.settings.plate_class_ids)
        assert self._detector is not None
        names = getattr(self._detector, "names", {})
        if isinstance(names, list):
            names = {index: item for index, item in enumerate(names)}
        names = {int(index): str(name) for index, name in dict(names).items()}
        if len(names) == 1:
            return set(names)
        matched = {
            class_id
            for class_id, name in names.items()
            if any(
                token in self._normalize_class_name(name)
                for token in ("plate", "license_plate", "number_plate", "پلاک")
            )
        }
        if not matched:
            raise RuntimeError("Could not identify plate class; configure PLATE_CLASS_IDS")
        return matched

    def _predict_yolo(
        self,
        *,
        model: Any,
        frames: list[np.ndarray],
        confidence: float,
        iou: float,
        imgsz: int,
        classes: list[int] | None = None,
    ) -> tuple[list[Any], float]:
        if not frames:
            return [], 0.0
        quantize = (
            16
            if self.settings.use_fp16 and self.settings.device.lower() != "cpu"
            else None
        )
        kwargs: dict[str, Any] = {
            "source": frames,
            "batch": len(frames),
            "conf": confidence,
            "iou": iou,
            "imgsz": imgsz,
            "device": self.settings.device,
            "quantize": quantize,
            "verbose": False,
        }
        if classes:
            kwargs["classes"] = classes
        started = time.perf_counter()
        results = list(model.predict(**kwargs))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if len(results) != len(frames):
            raise RuntimeError("YOLO result count does not match input count")
        return results, elapsed_ms

    def _predict_vehicles(self, frames: list[np.ndarray]) -> list[Any]:
        self._ensure_models()
        assert self._vehicle_detector is not None
        results, self._last_vehicle_inference_ms = self._predict_yolo(
            model=self._vehicle_detector,
            frames=frames,
            confidence=self.settings.vehicle_confidence,
            iou=self.settings.vehicle_iou,
            imgsz=self.settings.vehicle_imgsz,
            classes=list(self.settings.vehicle_class_ids),
        )
        return results

    def _predict_plates(self, vehicle_crops: list[np.ndarray]) -> list[Any]:
        self._ensure_models()
        assert self._detector is not None
        results: list[Any] = []
        self._last_plate_inference_ms = 0.0
        chunk_size = max(1, self.settings.plate_crop_batch_size)
        for start in range(0, len(vehicle_crops), chunk_size):
            chunk_results, elapsed_ms = self._predict_yolo(
                model=self._detector,
                frames=vehicle_crops[start : start + chunk_size],
                confidence=self.settings.detector_confidence,
                iou=self.settings.detector_iou,
                imgsz=self.settings.detector_imgsz,
            )
            results.extend(chunk_results)
            self._last_plate_inference_ms += elapsed_ms
        self._last_inference_ms = (
            self._last_vehicle_inference_ms + self._last_plate_inference_ms
        )
        return results

    @staticmethod
    def _read_output(output: Any) -> tuple[str, float]:
        while isinstance(output, (list, tuple)) and len(output) == 1:
            output = output[0]
        if isinstance(output, dict):
            text = output.get("text", "")
            score = output.get("score", 0.0)
        else:
            text = getattr(output, "text", str(output) if output is not None else "")
            score = getattr(output, "score", 0.0)
        try:
            confidence = float(score)
        except (TypeError, ValueError):
            confidence = 0.0
        return str(text), confidence

    def _recognize_crops(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        assert self._recognizer is not None
        if not crops:
            return []
        rgb = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops]
        try:
            outputs = self._recognizer.predict(
                rgb,
                device=self._torch_device(),
                return_scores=True,
            )
            if not isinstance(outputs, (list, tuple)) or len(outputs) != len(crops):
                raise ValueError("Recognizer did not return one result per crop")
        except Exception:
            outputs = []
            for crop in rgb:
                value = self._recognizer.predict(
                    crop,
                    device=self._torch_device(),
                    return_scores=True,
                )
                if isinstance(value, (list, tuple)):
                    value = value[0] if value else None
                outputs.append(value)
        recognized: list[tuple[str, float]] = []
        for output in outputs:
            text, score = self._read_output(output)
            recognized.append(
                (
                    normalize_plate_text(text, self.settings.output_persian_digits),
                    score,
                )
            )
        return recognized

    def process_batch(self, packets: Sequence[FramePacket]) -> list[TaskResult]:
        if not packets:
            return []
        vehicle_results = self._predict_vehicles(
            [packet.frame for packet in packets]
        )
        vehicle_records: list[
            tuple[int, list[int], float, np.ndarray]
        ] = []
        vehicle_counts = {index: 0 for index in range(len(packets))}
        padding_ratio = max(0.0, self.settings.vehicle_crop_padding_ratio)
        allowed_vehicle_ids = set(self.settings.vehicle_class_ids)

        for frame_position, (packet, result) in enumerate(
            zip(packets, vehicle_results)
        ):
            frame_height, frame_width = packet.frame.shape[:2]
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            accepted = 0
            for box in boxes:
                class_id = int(box.cls[0].item())
                vehicle_confidence = float(box.conf[0].item())
                if (
                    class_id not in allowed_vehicle_ids
                    or vehicle_confidence < self.settings.vehicle_confidence
                ):
                    continue
                x1, y1, x2, y2 = [
                    int(round(value)) for value in box.xyxy[0].tolist()
                ]
                padding_x = int(round(max(0, x2 - x1) * padding_ratio))
                padding_y = int(round(max(0, y2 - y1) * padding_ratio))
                x1 = max(0, min(x1 - padding_x, frame_width - 1))
                y1 = max(0, min(y1 - padding_y, frame_height - 1))
                x2 = max(x1 + 1, min(x2 + padding_x, frame_width))
                y2 = max(y1 + 1, min(y2 + padding_y, frame_height))
                crop = packet.frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                vehicle_records.append(
                    (
                        frame_position,
                        [x1, y1, x2, y2],
                        vehicle_confidence,
                        crop,
                    )
                )
                accepted += 1
                if accepted >= max(1, self.settings.vehicle_max_per_frame):
                    break
            vehicle_counts[frame_position] = accepted

        detector_results = self._predict_plates(
            [record[3] for record in vehicle_records]
        )
        allowed_ids = self._plate_class_ids()

        crop_records: list[
            tuple[int, list[int], float, np.ndarray, list[int], float]
        ] = []
        for vehicle_record, result in zip(vehicle_records, detector_results):
            (
                frame_position,
                vehicle_bbox,
                vehicle_confidence,
                vehicle_crop,
            ) = vehicle_record
            vehicle_x1, vehicle_y1, _, _ = vehicle_bbox
            crop_height, crop_width = vehicle_crop.shape[:2]
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0].item())
                if class_id not in allowed_ids:
                    continue
                detector_confidence = float(box.conf[0].item())
                if detector_confidence < self.settings.detector_confidence:
                    continue
                x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
                x1 = max(0, min(x1, crop_width - 1))
                y1 = max(0, min(y1, crop_height - 1))
                x2 = max(x1 + 1, min(x2, crop_width))
                y2 = max(y1 + 1, min(y2, crop_height))
                crop = vehicle_crop[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_records.append(
                    (
                        frame_position,
                        [
                            vehicle_x1 + x1,
                            vehicle_y1 + y1,
                            vehicle_x1 + x2,
                            vehicle_y1 + y2,
                        ],
                        detector_confidence,
                        crop,
                        vehicle_bbox,
                        vehicle_confidence,
                    )
                )

        recognized = self._recognize_crops([item[3] for item in crop_records])
        by_frame: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(packets))}
        for record, (plate, recognizer_confidence) in zip(crop_records, recognized):
            (
                frame_position,
                bbox,
                detector_confidence,
                _,
                vehicle_bbox,
                vehicle_confidence,
            ) = record
            by_frame[frame_position].append(
                {
                    "plate": plate,
                    "detector_confidence": round(detector_confidence, 5),
                    "recognizer_confidence": round(recognizer_confidence, 5),
                    "bbox": bbox,
                    "vehicle_bbox": vehicle_bbox,
                    "vehicle_confidence": round(vehicle_confidence, 5),
                    "characters": list(plate),
                }
            )

        results: list[TaskResult] = []
        for index, packet in enumerate(packets):
            deduplicated: dict[str, dict[str, Any]] = {}
            unreadable: list[dict[str, Any]] = []
            for prediction in by_frame[index]:
                plate = prediction["plate"]
                if not plate:
                    unreadable.append(prediction)
                    continue
                previous = deduplicated.get(plate)
                score = prediction["detector_confidence"] + prediction["recognizer_confidence"]
                previous_score = (
                    previous["detector_confidence"] + previous["recognizer_confidence"]
                    if previous is not None
                    else -1.0
                )
                if score > previous_score:
                    deduplicated[plate] = prediction
            predictions = list(deduplicated.values()) + unreadable
            results.append(
                TaskResult.success(
                    task=self.task,
                    packet=packet,
                    processing_ms=self._last_inference_ms,
                    data={
                        "plates": predictions,
                        "plate_count": len(predictions),
                        "vehicle_count": vehicle_counts[index],
                        "vehicle_inference_ms": round(
                            self._last_vehicle_inference_ms, 3
                        ),
                        "plate_inference_ms": round(
                            self._last_plate_inference_ms, 3
                        ),
                        "batch_inference_ms": round(self._last_inference_ms, 3),
                    },
                )
            )
        self._processed_batches += 1
        self._processed_frames += len(packets)
        return results

    def status(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "pipeline": "vehicle -> plate -> OCR",
            "vehicle_detector_weights": str(
                self.settings.vehicle_detector_weights
            ),
            "vehicle_detector_exists": (
                self.settings.vehicle_detector_weights.is_file()
            ),
            "vehicle_confidence": self.settings.vehicle_confidence,
            "detector_weights": str(self.settings.detector_weights),
            "detector_exists": self.settings.detector_weights.is_file(),
            "plate_confidence": self.settings.detector_confidence,
            "recognizer_model_dir": str(self.settings.recognizer_model_dir),
            "recognizer_complete": self.settings.recognizer_model_dir.is_dir()
            and not self.recognizer_missing_files(),
            "models_loaded": (
                self._vehicle_detector is not None
                and self._detector is not None
                and self._recognizer is not None
            ),
            "model_load_error": str(self._load_error) if self._load_error else None,
            "processed_batches": self._processed_batches,
            "processed_frames": self._processed_frames,
            "last_inference_ms": round(self._last_inference_ms, 3),
            "last_vehicle_inference_ms": round(
                self._last_vehicle_inference_ms, 3
            ),
            "last_plate_inference_ms": round(
                self._last_plate_inference_ms, 3
            ),
        }
