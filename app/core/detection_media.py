from __future__ import annotations

import base64
import mimetypes
import uuid
from pathlib import Path, PurePosixPath
from typing import Iterable
from urllib.parse import unquote, urlsplit

import cv2
import numpy as np
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles


MEDIA_STATUS_MISSING = "missing"
MEDIA_STATUS_WRITING = "writing"
MEDIA_STATUS_READY = "ready"
MEDIA_STATUS_FAILED = "failed"
VALID_MEDIA_STATUSES = {
    MEDIA_STATUS_MISSING,
    MEDIA_STATUS_WRITING,
    MEDIA_STATUS_READY,
    MEDIA_STATUS_FAILED,
}

# These directories contain personnel/detection evidence and must never be
# exposed by the compatibility /media static mount.
PRIVATE_DETECTION_MEDIA_DIRS = frozenset(
    {
        "human",
        # Legacy flat names remain private while older development media exists.
        "detected_faces",
        "face_thumbnails",
        "body_images",
        "full_frame_images",
        "reference_images",
        "human_snapshots",
        "whole_snapshots",
        "human_videos",
        "human_face_videos",
        "personnel_cropped_faces",
        "personnel_snapshots",
        "personnel_zip_errors",
    }
)


class InvalidMediaKey(ValueError):
    """Raised when a stored media reference escapes the configured media root."""


class RestrictedMediaStaticFiles(StaticFiles):
    """Compatibility static files that deny private detection/personnel media."""

    async def get_response(self, path: str, scope):  # type: ignore[override]
        normalized = PurePosixPath(path.replace("\\", "/"))
        first = normalized.parts[0] if normalized.parts else ""
        if first in PRIVATE_DETECTION_MEDIA_DIRS:
            raise StarletteHTTPException(status_code=404)
        return await super().get_response(path, scope)


class DetectionMediaStorage:
    """Resolve and manage detection media using root-relative storage keys.

    Database values are storage keys such as ``human/detected_faces/a.jpg``. Legacy
    ``/media/...`` values and absolute paths inside the configured root are
    accepted and normalized, but paths outside the root are rejected.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def canonical_key(self, raw: str | Path | None) -> str | None:
        if raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None

        parsed = urlsplit(text)
        if parsed.scheme or parsed.netloc:
            # file:// and remote URLs are not valid storage keys.
            raise InvalidMediaKey("Media references must be local storage keys")
        text = unquote(parsed.path).replace("\\", "/")
        if text.startswith("/media/"):
            text = text[len("/media/") :]

        candidate = Path(text)
        if candidate.is_absolute():
            resolved = candidate.expanduser().resolve()
        else:
            normalized = PurePosixPath(text)
            if any(part in {"", ".", ".."} for part in normalized.parts):
                raise InvalidMediaKey("Invalid media storage key")
            resolved = (self.root / Path(*normalized.parts)).resolve()

        if resolved != self.root and self.root not in resolved.parents:
            raise InvalidMediaKey("Media path is outside SAVED_MEDIA_PATH")
        if resolved == self.root:
            raise InvalidMediaKey("Media storage key must identify a file")
        return resolved.relative_to(self.root).as_posix()

    def resolve(
        self,
        raw: str | Path | None,
        *,
        require_file: bool = False,
    ) -> Path | None:
        key = self.canonical_key(raw)
        if key is None:
            return None
        candidate = (self.root / Path(key)).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise InvalidMediaKey("Media path is outside SAVED_MEDIA_PATH")
        if require_file and not candidate.is_file():
            return None
        return candidate

    def key_for_path(self, path: Path) -> str:
        resolved = path.expanduser().resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise InvalidMediaKey("Media path is outside SAVED_MEDIA_PATH")
        if resolved == self.root:
            raise InvalidMediaKey("Media storage key must identify a file")
        return resolved.relative_to(self.root).as_posix()

    def exists(self, raw: str | Path | None) -> bool:
        try:
            path = self.resolve(raw, require_file=True)
        except InvalidMediaKey:
            return False
        return path is not None and path.stat().st_size > 0

    def delete(self, raw: str | Path | None) -> bool:
        try:
            path = self.resolve(raw)
        except InvalidMediaKey:
            return False
        if path is None or not path.is_file():
            return False
        path.unlink(missing_ok=True)
        self._remove_empty_parents(path.parent)
        return True

    def delete_many(self, keys: Iterable[str | Path | None]) -> int:
        deleted = 0
        seen: set[str] = set()
        for raw in keys:
            try:
                key = self.canonical_key(raw)
            except InvalidMediaKey:
                continue
            if not key or key in seen:
                continue
            seen.add(key)
            deleted += int(self.delete(key))
        return deleted

    def _remove_empty_parents(self, directory: Path) -> None:
        current = directory
        while current != self.root and self.root in current.parents:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

    @staticmethod
    def content_type(path: Path) -> str:
        guessed, _ = mimetypes.guess_type(path.name)
        return guessed or "application/octet-stream"

    def finalized_video_status(self, raw: str | Path | None) -> str:
        if raw in (None, ""):
            return MEDIA_STATUS_MISSING
        try:
            path = self.resolve(raw, require_file=True)
        except InvalidMediaKey:
            return MEDIA_STATUS_FAILED
        if path is None or path.stat().st_size <= 0:
            return MEDIA_STATUS_FAILED
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                return MEDIA_STATUS_FAILED
            success, frame = capture.read()
            if not success or frame is None or frame.size == 0:
                return MEDIA_STATUS_FAILED
            return MEDIA_STATUS_READY
        finally:
            capture.release()

    @staticmethod
    def resized_thumbnail(
        image: np.ndarray,
        *,
        max_size: int = 224,
    ) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Cannot create a thumbnail from an empty image")
        height, width = image.shape[:2]
        scale = min(1.0, float(max_size) / max(height, width))
        if scale >= 1.0:
            return image.copy()
        size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    def save_jpeg(
        self,
        image: np.ndarray,
        *,
        directory: str,
        filename: str,
        quality: int,
        max_size: int | None = None,
    ) -> tuple[str, Path]:
        safe_directory = self.canonical_key(f"{directory}/placeholder")
        assert safe_directory is not None
        target_dir = (self.root / Path(safe_directory).parent).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        target = (target_dir / Path(filename).name).resolve()
        if self.root not in target.parents:
            raise InvalidMediaKey("Media path is outside SAVED_MEDIA_PATH")
        output = self.resized_thumbnail(image, max_size=max_size) if max_size else image
        if not cv2.imwrite(
            str(target),
            output,
            [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, int(quality)))],
        ):
            raise RuntimeError(f"Could not save media image: {target}")
        return self.key_for_path(target), target

    def create_face_thumbnail(
        self,
        face_key: str | None,
        *,
        filename: str | None = None,
        max_size: int = 224,
        quality: int = 72,
    ) -> str | None:
        try:
            face_path = self.resolve(face_key, require_file=True)
        except InvalidMediaKey:
            return None
        if face_path is None:
            return None
        image = cv2.imread(str(face_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return None
        target_name = filename or (
            f"{face_path.stem}_{uuid.uuid4().hex[:12]}_thumbnail.jpg"
        )
        key, _ = self.save_jpeg(
            image,
            directory="human/face_thumbnails",
            filename=target_name,
            quality=quality,
            max_size=max_size,
        )
        return key

    def thumbnail_bytes(
        self,
        thumbnail_key: str | None,
        face_key: str | None = None,
        *,
        max_size: int = 224,
        quality: int = 72,
    ) -> bytes | None:
        source_path: Path | None = None
        image: np.ndarray | None = None
        for key in (thumbnail_key, face_key):
            try:
                source_path = self.resolve(key, require_file=True)
            except InvalidMediaKey:
                source_path = None
            if source_path is not None:
                image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
                if image is not None and image.size > 0:
                    break
                source_path = None
        if source_path is None or image is None or image.size == 0:
            return None
        image = self.resized_thumbnail(image, max_size=max_size)
        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, int(quality)))],
        )
        return encoded.tobytes() if success else None

    def thumbnail_data_uri(
        self,
        thumbnail_key: str | None,
        face_key: str | None = None,
    ) -> str | None:
        try:
            data = self.thumbnail_bytes(thumbnail_key, face_key)
        except (InvalidMediaKey, OSError):
            return None
        if not data:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
