from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import settings
from app.processors.face_recognition import (
    FaceEnrollmentValidationError,
    FaceRecognitionProcessor,
)

LOGGER = logging.getLogger("uvicorn.error")


# ── Image format detection ──────────────────────────────────────────


_EXTENSION_TO_FORMAT: dict[str, str] = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".bmp": "bmp",
}

_MIME_TO_EXT: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/bmp": ".bmp",
}


def _detect_format_from_bytes(data: bytes) -> str | None:
    """Detect image format from magic bytes."""
    if data[:2] == b"\xff\xd8":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:2] == b"BM":
        return "bmp"
    return None


def _format_to_mime(fmt: str) -> str:
    return {"jpeg": "image/jpeg", "png": "image/png", "bmp": "image/bmp"}.get(fmt, "application/octet-stream")


def _format_to_ext(fmt: str) -> str:
    return {".jpg": "jpeg", ".jpeg": "jpeg", "png": "png", "bmp": "bmp"}.get(fmt, ".jpg")


# ── Validation helpers ──────────────────────────────────────────────


def _safe_filename(filename: str) -> str:
    """Normalize a filename: strip path components, keep only name."""
    import re
    # Remove any path components
    name = Path(filename).name
    # Remove null bytes and control characters
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    return name if name else "image"


def _validate_extension(filename: str) -> str | None:
    """Check extension against allowed list. Returns the normalized extension or None."""
    name = _safe_filename(filename)
    ext = Path(name).suffix.lower()
    if ext in settings.supported_image_extensions:
        return ext
    return None


def _validate_mime(content_type: str | None) -> str | None:
    """Check MIME type against allowed list. Returns the extension or None."""
    if content_type is None:
        return None
    ct = content_type.strip().lower()
    ext = _MIME_TO_EXT.get(ct)
    if ext is not None and ext in settings.supported_image_extensions:
        return ext
    return None


# ── Result types ────────────────────────────────────────────────────


@dataclass
class ImageValidationResult:
    valid: bool
    failure_code: str | None = None
    failure_message: str | None = None
    data: bytes | None = None
    detected_format: str | None = None
    width: int = 0
    height: int = 0


@dataclass
class ImageProcessResult:
    success: bool
    failure_code: str | None = None
    failure_message: str | None = None
    failure_details: dict[str, Any] | None = None
    processed_bytes: bytes | None = None
    processed_format: str | None = None  # "jpeg", "png", "bmp"
    embedding: np.ndarray | None = None
    vector_point_id: str | None = None
    detected_face_count: int = 0
    selected_face_metadata: dict[str, Any] | None = None

    @property
    def status(self) -> int:
        """Legacy-compatible vector enrollment status (1=success, 0=failure)."""
        return 1 if self.success and self.vector_point_id else 0


# ── Validation ──────────────────────────────────────────────────────


def validate_uploaded_image(data: bytes, filename: str, content_type: str | None) -> ImageValidationResult:
    """Run all safety checks on an uploaded image before face processing.

    Returns ImageValidationResult with valid=True only when all checks pass.
    """
    # 1. Nonempty
    if not data:
        return ImageValidationResult(valid=False, failure_code="empty_file", failure_message="فایل تصویر خالی است.")

    # 2. Safe filename
    safe_name = _safe_filename(filename)
    if not safe_name:
        return ImageValidationResult(valid=False, failure_code="invalid_filename", failure_message="نام فایل تصویر معتبر نیست.")

    # 3. No path traversal
    if ".." in filename or "/" in filename.lstrip("/") or "\\" in filename:
        return ImageValidationResult(valid=False, failure_code="path_traversal", failure_message="نام فایل شامل مسیر غیرمجاز است.")

    # 4. Extension check
    ext = _validate_extension(filename)
    if ext is None:
        return ImageValidationResult(
            valid=False,
            failure_code="unsupported_extension",
            failure_message=f"پسوند تصویر پشتیبانی نمی‌شود. پسوندهای مجاز: {', '.join(settings.supported_image_extensions)}",
        )

    # 5. MIME check (when provided)
    mime_ext = _validate_mime(content_type)
    if content_type and mime_ext is None:
        return ImageValidationResult(
            valid=False,
            failure_code="unsupported_mime",
            failure_message=f"نوع محتوای فایل پشتیبانی نمی‌شود: {content_type}",
        )

    # 6. Extension-MIME consistency (when both provided)
    if content_type and mime_ext and ext != mime_ext:
        return ImageValidationResult(
            valid=False,
            failure_code="mime_extension_mismatch",
            failure_message="نوع محتوای فایل با پسوند آن مطابقت ندارد.",
        )

    # 7. File size
    if len(data) > settings.max_upload_bytes_per_image:
        return ImageValidationResult(
            valid=False,
            failure_code="file_too_large",
            failure_message=f"حجم تصویر بیشتر از حد مجاز {settings.max_upload_bytes_per_image // (1024 * 1024)} مگابایت است.",
        )

    # 8. Decode and check format
    detected = _detect_format_from_bytes(data)
    if detected is None:
        return ImageValidationResult(valid=False, failure_code="corrupt_image", failure_message="فایل تصویر خراب یا نامعتبر است و قابل خواندن نیست.")

    # 9. Verify actual image decoding
    np_arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return ImageValidationResult(valid=False, failure_code="corrupt_image", failure_message="فایل تصویر خراب یا نامعتبر است و قابل خواندن نیست.")

    height, width = img.shape[:2]

    # 10. Nonzero dimensions
    if width < 1 or height < 1:
        return ImageValidationResult(valid=False, failure_code="zero_dimensions", failure_message="عرض یا ارتفاع تصویر نامعتبر است.")

    # 11. Max dimensions
    if width > settings.max_decoded_width:
        return ImageValidationResult(
            valid=False,
            failure_code="dimensions_too_large",
            failure_message=f"عرض تصویر ({width}) بیشتر از حد مجاز ({settings.max_decoded_width}) است.",
        )
    if height > settings.max_decoded_height:
        return ImageValidationResult(
            valid=False,
            failure_code="dimensions_too_large",
            failure_message=f"ارتفاع تصویر ({height}) بیشتر از حد مجاز ({settings.max_decoded_height}) است.",
        )

    # 12. Max pixels
    total_pixels = width * height
    if total_pixels > settings.max_total_decoded_pixels:
        return ImageValidationResult(
            valid=False,
            failure_code="too_many_pixels",
            failure_message=f"تعداد پیکسل‌های تصویر ({total_pixels}) بیشتر از حد مجاز ({settings.max_total_decoded_pixels}) است.",
        )

    return ImageValidationResult(
        valid=True,
        data=data,
        detected_format=detected,
        width=width,
        height=height,
    )


# ── Face processing service ─────────────────────────────────────────


class PersonnelImageProcessor:
    """Application service for processing one Personnel image.

    The router coordinates authorization, validation, service invocation,
    transaction lifecycle, and response construction. This service handles
    image decoding, face detection, optional cropping, embedding generation,
    and vector persistence.
    """

    def __init__(
        self,
        face_processor: FaceRecognitionProcessor | None,
    ) -> None:
        self._face_processor = face_processor

    def process_image(
        self,
        image_data: bytes,
        *,
        person_name: str,
        ref_img_id: str | int | None = None,
        enable_cropping: bool = False,
    ) -> ImageProcessResult:
        """Process one validated personnel image.

        When face_processor is None (mock mode), returns a mock success
        result without actual face detection or embedding.
        """
        if self._face_processor is None:
            # Mock mode — return a fake success
            fake_point_id = str(uuid.uuid4())
            return ImageProcessResult(
                success=True,
                processed_bytes=image_data,
                processed_format="jpeg",
                embedding=np.zeros((settings.face_vector_size,), dtype=np.float32),
                vector_point_id=fake_point_id,
                detected_face_count=1,
                selected_face_metadata={"mock": True},
            )

        # Decode image for face processing
        np_arr = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return ImageProcessResult(
                success=False,
                failure_code="decode_failed",
                failure_message="فایل تصویر قابل خواندن نیست یا ساختار آن خراب است.",
            )

        # Run face enrollment through the existing processor pipeline
        try:
            enroll_result = self._face_processor.enroll(
                image,
                person=person_name,
                ref_img_id=ref_img_id,
            )
        except FaceEnrollmentValidationError as exc:
            return ImageProcessResult(
                success=False,
                failure_code=exc.code,
                failure_message=str(exc),
                failure_details=exc.details,
            )
        except ValueError as exc:
            return ImageProcessResult(
                success=False,
                failure_code="enrollment_failed",
                failure_message=f"ثبت چهره انجام نشد: {exc}",
            )
        except Exception as exc:
            LOGGER.warning("Face enrollment unexpected error: %s", exc)
            return ImageProcessResult(
                success=False,
                failure_code="enrollment_error",
                failure_message=(
                    "خطای داخلی هنگام پردازش چهره رخ داد؛ وضعیت مدل‌ها و سرویس برداری را بررسی کنید. "
                    f"({type(exc).__name__})"
                ),
            )

        point_id = enroll_result.get("point_id")
        bbox = enroll_result.get("bbox")

        # Determine which image bytes to store
        if enable_cropping and settings.store_cropped_face:
            # The enroll method already aligned/cropped the face internally.
            # Re-run alignment to get the processed crop for storage.
            # We re-use the internal quality pipeline.
            landmarks = enroll_result.get("quality_metrics", {}).get("landmark_count", 0)
            if landmarks >= 5:
                faces = self._face_processor._faces(
                    self._face_processor._predict_yolo(
                        self._face_processor._face_detector,
                        [image],
                        model_path=self._face_processor.settings.face_model_path,
                        imgsz=self._face_processor.settings.face_imgsz,
                        confidence=self._face_processor.settings.face_confidence,
                        fixed_batch=self._face_processor.settings.face_engine_fixed_batch,
                    )[0]
                )
                for face in faces:
                    _, quality, _, crop, _ = self._face_processor._quality(image, face)
                    if quality > 0 and crop is not None:
                        success, encoded = cv2.imencode(".jpg", crop)
                        if success:
                            processed_bytes = encoded.tobytes()
                            processed_format = "jpeg"
                            break
                else:
                    # Fall back to original
                    processed_bytes = image_data
                    processed_format = "jpeg"
            else:
                processed_bytes = image_data
                processed_format = "jpeg"
        else:
            processed_bytes = image_data
            processed_format = "jpeg"

        selected_metadata: dict[str, Any] = {
            "bbox": bbox,
            "quality": enroll_result.get("quality"),
        }
        quality_metrics = enroll_result.get("quality_metrics", {})
        if quality_metrics:
            selected_metadata["quality_metrics"] = quality_metrics

        # Build embedding (already computed by enroll, but we need the np array)
        # We re-embed the crop to get the actual vector for storage.
        # Actually enroll already returns the point_id — we don't need to
        # store the embedding vector itself.
        embedding = np.zeros((settings.face_vector_size,), dtype=np.float32)

        return ImageProcessResult(
            success=True,
            processed_bytes=processed_bytes,
            processed_format=processed_format,
            embedding=embedding,
            vector_point_id=point_id,
            detected_face_count=1,
            selected_face_metadata=selected_metadata,
        )

    def is_mock(self) -> bool:
        return self._face_processor is None

    def delete_vector(self, point_id: str | None) -> bool:
        """Compensate a successful enrollment when image persistence fails."""
        if not point_id or self._face_processor is None:
            return True
        try:
            return bool(self._face_processor.delete_points([point_id]))
        except Exception as exc:
            LOGGER.warning("Vector enrollment rollback failed for %s: %s", point_id, exc)
            return False
