from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import UUID


SHA256_METADATA_KEY = "sha256"


class RecordingStorageError(RuntimeError):
    pass


class ObjectVerificationError(RecordingStorageError):
    pass


class ObjectConflictError(RecordingStorageError):
    pass


class ObjectNotFoundError(RecordingStorageError):
    pass


class MinioClientProtocol(Protocol):
    def bucket_exists(self, bucket_name: str) -> bool: ...

    def make_bucket(self, bucket_name: str) -> None: ...

    def fput_object(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str,
        content_type: str,
        metadata: dict[str, str],
    ) -> Any: ...

    def stat_object(self, bucket_name: str, object_name: str) -> Any: ...

    def remove_object(self, bucket_name: str, object_name: str) -> None: ...

    def presigned_get_object(
        self, bucket_name: str, object_name: str, expires: timedelta
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class RecordingStorageSettings:
    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str = "recordings"
    secure: bool = False
    region: str | None = None
    minio_retention_days: int = 30
    network_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not self.endpoint.strip():
            raise ValueError("MinIO endpoint must not be empty")
        if not self.access_key or not self.secret_key:
            raise ValueError("MinIO credentials must not be empty")
        if not self.bucket_name.strip():
            raise ValueError("MinIO bucket name must not be empty")
        if self.minio_retention_days <= 0:
            raise ValueError("MinIO retention must be positive")
        if self.network_timeout_seconds <= 0:
            raise ValueError("MinIO network timeout must be positive")

    def create_client(self) -> MinioClientProtocol:
        try:
            from minio import Minio
            import urllib3
        except ImportError as exc:
            raise RecordingStorageError(
                "The optional 'minio' package is required to create a MinIO client"
            ) from exc
        http_client = urllib3.PoolManager(
            timeout=urllib3.Timeout(connect=self.network_timeout_seconds, read=self.network_timeout_seconds),
            retries=urllib3.Retry(total=1, connect=1, read=1, redirect=0),
        )
        return Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
            region=self.region,
            http_client=http_client,
        )


@dataclass(frozen=True, slots=True)
class StoredRecording:
    job_uuid: UUID
    object_key: str
    size: int
    sha256: str
    already_present: bool


@dataclass(frozen=True, slots=True)
class SpoolSettings:
    root: Path
    failed_retention_days: int = 7
    high_water_percent: float = 90.0
    max_scan_files: int = 10_000
    max_delete_files: int = 250

    def __post_init__(self) -> None:
        if self.failed_retention_days <= 0:
            raise ValueError("Failed spool retention must be positive")
        if not 0 < self.high_water_percent <= 100:
            raise ValueError("High-water percent must be in (0, 100]")
        if self.max_scan_files <= 0 or self.max_delete_files <= 0:
            raise ValueError("Spool cleanup bounds must be positive")


@dataclass(frozen=True, slots=True)
class SpoolStatus:
    total_bytes: int
    used_bytes: int
    free_bytes: int
    used_percent: float
    high_water_exceeded: bool


@dataclass(frozen=True, slots=True)
class SpoolCleanupResult:
    scanned: int
    deleted: int
    bytes_freed: int
    scan_limit_reached: bool
    delete_limit_reached: bool
    status: SpoolStatus


def recording_object_key(job_uuid: UUID | str) -> str:
    parsed = job_uuid if isinstance(job_uuid, UUID) else UUID(str(job_uuid))
    return f"recordings/{parsed}.mp4"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class RecordingStorageService:
    def __init__(
        self,
        settings: RecordingStorageSettings,
        client: MinioClientProtocol | None = None,
        lifecycle_config_factory: Callable[[int], Any] | None = None,
    ) -> None:
        self.settings = settings
        self.client = client if client is not None else settings.create_client()
        self._lifecycle_config_factory = lifecycle_config_factory

    def initialize_private_bucket(self) -> None:
        if not self.client.bucket_exists(self.settings.bucket_name):
            self.client.make_bucket(self.settings.bucket_name)
        self._remove_public_policy()
        self._configure_expiration()
        self.backfill_legacy_retention_tags()

    def backfill_legacy_retention_tags(self, max_objects: int = 10_000) -> int:
        list_objects = getattr(self.client, "list_objects", None)
        get_tags = getattr(self.client, "get_object_tags", None)
        set_tags = getattr(self.client, "set_object_tags", None)
        if list_objects is None or get_tags is None or set_tags is None:
            return 0
        from minio.commonconfig import Tags
        updated = 0
        for index, item in enumerate(list_objects(self.settings.bucket_name, recursive=True)):
            if index >= max_objects:
                break
            object_name = str(getattr(item, "object_name", ""))
            if not object_name or object_name.endswith("/"):
                continue
            tags = get_tags(self.settings.bucket_name, object_name) or {}
            if tags.get("retention-days"):
                continue
            replacement = Tags.new_object_tags()
            replacement.update(tags)
            replacement["retention-days"] = "30"
            set_tags(self.settings.bucket_name, object_name, replacement)
            updated += 1
        return updated

    def upload_finalized(
        self, job_uuid: UUID | str, local_path: Path | str, *, metadata: dict[str, str] | None = None,
        retention_days: int = 30,
    ) -> StoredRecording:
        parsed_uuid = job_uuid if isinstance(job_uuid, UUID) else UUID(str(job_uuid))
        path = Path(local_path)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Finalized recording is not a regular file: {path}")
        size = path.stat().st_size
        checksum = sha256_file(path)
        key = recording_object_key(parsed_uuid)

        existing = self._stat_if_present(key)
        if existing is not None:
            self._verify_stat(existing, size, checksum, key, conflict=True)
            self._set_retention_tags(key, retention_days)
            return StoredRecording(parsed_uuid, key, size, checksum, True)

        self.client.fput_object(
            self.settings.bucket_name,
            key,
            str(path),
            content_type="video/mp4",
            metadata={SHA256_METADATA_KEY: checksum, "retention-days": str(retention_days), **(metadata or {})},
        )
        self.verify(key, size, checksum)
        self._set_retention_tags(key, retention_days)
        return StoredRecording(parsed_uuid, key, size, checksum, False)

    def _set_retention_tags(self, object_key: str, retention_days: int) -> None:
        if retention_days not in {7, 30, 90, 365}:
            raise ValueError("invalid recording retention period")
        setter = getattr(self.client, "set_object_tags", None)
        if setter is None:
            return
        from minio.commonconfig import Tags
        tags = Tags.new_object_tags()
        tags["retention-days"] = str(retention_days)
        setter(self.settings.bucket_name, object_key, tags)

    def upload_verified_and_remove_local(
        self, job_uuid: UUID | str, local_path: Path | str
    ) -> StoredRecording:
        path = Path(local_path)
        stored = self.upload_finalized(job_uuid, path)
        path.unlink()
        return stored

    def verify(self, object_key: str, expected_size: int, expected_sha256: str) -> None:
        stat = self._stat_if_present(object_key)
        if stat is None:
            raise ObjectNotFoundError(f"Recording object not found: {object_key}")
        self._verify_stat(stat, expected_size, expected_sha256, object_key)

    def delete(self, job_uuid: UUID | str) -> None:
        self.client.remove_object(self.settings.bucket_name, recording_object_key(job_uuid))

    def presigned_download(
        self, job_uuid: UUID | str, expires: timedelta = timedelta(minutes=15)
    ) -> str:
        if expires <= timedelta(0) or expires > timedelta(days=7):
            raise ValueError("Presigned expiry must be greater than zero and at most 7 days")
        key = recording_object_key(job_uuid)
        if self._stat_if_present(key) is None:
            raise ObjectNotFoundError(f"Recording object not found: {key}")
        return self.client.presigned_get_object(
            self.settings.bucket_name, key, expires=expires
        )

    def _stat_if_present(self, object_key: str) -> Any | None:
        try:
            return self.client.stat_object(self.settings.bucket_name, object_key)
        except Exception as exc:
            code = _error_code(exc)
            if code in {"NoSuchKey", "NoSuchObject", "NotFound", "XMinioInvalidObjectName"}:
                return None
            raise

    def _verify_stat(
        self,
        stat: Any,
        expected_size: int,
        expected_sha256: str,
        object_key: str,
        conflict: bool = False,
    ) -> None:
        actual_size = int(getattr(stat, "size", -1))
        metadata = getattr(stat, "metadata", {}) or {}
        actual_checksum = _metadata_value(metadata, SHA256_METADATA_KEY)
        if actual_size != expected_size or actual_checksum != expected_sha256:
            error_type = ObjectConflictError if conflict else ObjectVerificationError
            raise error_type(
                f"Recording object verification failed for {object_key}: "
                f"expected size/checksum {expected_size}/{expected_sha256}, "
                f"received {actual_size}/{actual_checksum or 'missing'}"
            )

    def _remove_public_policy(self) -> None:
        remove_policy = getattr(self.client, "delete_bucket_policy", None)
        if remove_policy is None:
            return
        try:
            remove_policy(self.settings.bucket_name)
        except Exception as exc:
            if _error_code(exc) not in {"NoSuchBucketPolicy", "NoSuchPolicy"}:
                raise

    def _configure_expiration(self) -> None:
        set_lifecycle = getattr(self.client, "set_bucket_lifecycle", None)
        if set_lifecycle is None:
            return
        if self._lifecycle_config_factory is not None:
            config = self._lifecycle_config_factory(self.settings.minio_retention_days)
            set_lifecycle(self.settings.bucket_name, config)
            return
        try:
            from minio.commonconfig import Filter, Tag
            from minio.lifecycleconfig import Expiration, LifecycleConfig, Rule
        except ImportError as exc:
            raise RecordingStorageError(
                "The optional 'minio' package is required to configure bucket retention"
            ) from exc
        config = LifecycleConfig(
            [
                Rule(
                    "Enabled",
                    rule_filter=Filter(tag=Tag("retention-days", str(days))),
                    rule_id=f"recording-retention-{days}",
                    expiration=Expiration(days=days),
                )
                for days in (7, 30, 90, 365)
            ]
        )
        set_lifecycle(self.settings.bucket_name, config)


class LocalSpoolLifecycle:
    def __init__(
        self,
        settings: SpoolSettings,
        disk_usage: Callable[[Path], Any] = shutil.disk_usage,
        now: Callable[[], float] | None = None,
    ) -> None:
        self.settings = settings
        self._disk_usage = disk_usage
        self._now = now if now is not None else __import__("time").time

    def status(self) -> SpoolStatus:
        usage = self._disk_usage(self.settings.root)
        used_percent = (usage.used / usage.total * 100.0) if usage.total else 100.0
        return SpoolStatus(
            total_bytes=usage.total,
            used_bytes=usage.used,
            free_bytes=usage.free,
            used_percent=used_percent,
            high_water_exceeded=used_percent >= self.settings.high_water_percent,
        )

    def cleanup_failed(self, protected_paths: set[Path] | None = None) -> SpoolCleanupResult:
        root = self.settings.root.resolve()
        protected = {path.resolve() for path in (protected_paths or set())}
        cutoff = self._now() - self.settings.failed_retention_days * 86_400
        candidates: list[tuple[float, Path, int]] = []
        scanned = 0
        scan_limit_reached = False

        for directory, _, filenames in os.walk(root, followlinks=False):
            for filename in filenames:
                if scanned >= self.settings.max_scan_files:
                    scan_limit_reached = True
                    break
                scanned += 1
                path = Path(directory, filename)
                try:
                    stat = path.lstat()
                    resolved = path.resolve()
                except (FileNotFoundError, OSError):
                    continue
                if path.is_symlink() or resolved in protected or stat.st_mtime > cutoff:
                    continue
                if resolved != root and root not in resolved.parents:
                    continue
                candidates.append((stat.st_mtime, path, stat.st_size))
            if scan_limit_reached:
                break

        deleted = 0
        bytes_freed = 0
        for _, path, size in sorted(candidates):
            if deleted >= self.settings.max_delete_files:
                break
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            deleted += 1
            bytes_freed += size

        return SpoolCleanupResult(
            scanned=scanned,
            deleted=deleted,
            bytes_freed=bytes_freed,
            scan_limit_reached=scan_limit_reached,
            delete_limit_reached=len(candidates) > deleted,
            status=self.status(),
        )


def _metadata_value(metadata: Any, key: str) -> str | None:
    normalized = key.lower()
    for candidate, value in dict(metadata).items():
        candidate_name = str(candidate).lower()
        if candidate_name in {normalized, f"x-amz-meta-{normalized}"}:
            return str(value)
    return None


def _error_code(exc: Exception) -> str | None:
    return getattr(exc, "code", None)
