from __future__ import annotations

import os
import sys
from collections import namedtuple
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from app.core.recording_storage import (
    LocalSpoolLifecycle,
    ObjectConflictError,
    ObjectVerificationError,
    RecordingStorageService,
    RecordingStorageSettings,
    SpoolSettings,
    recording_object_key,
    sha256_file,
)


JOB_UUID = UUID("12345678-1234-5678-9234-567812345678")


class MissingObjectError(Exception):
    code = "NoSuchKey"


class FakeMinioClient:
    def __init__(self) -> None:
        self.bucket = False
        self.objects: dict[str, SimpleNamespace] = {}
        self.uploads = 0
        self.removed: list[str] = []
        self.policy_deleted = False
        self.lifecycle: object | None = None

    def bucket_exists(self, bucket_name: str) -> bool:
        return self.bucket

    def make_bucket(self, bucket_name: str) -> None:
        self.bucket = True

    def delete_bucket_policy(self, bucket_name: str) -> None:
        self.policy_deleted = True

    def set_bucket_lifecycle(self, bucket_name: str, config: object) -> None:
        self.lifecycle = config

    def fput_object(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str,
        content_type: str,
        metadata: dict[str, str],
    ) -> None:
        self.uploads += 1
        path = Path(file_path)
        self.objects[object_name] = SimpleNamespace(
            size=path.stat().st_size,
            metadata={f"x-amz-meta-{key}": value for key, value in metadata.items()},
        )

    def stat_object(self, bucket_name: str, object_name: str) -> SimpleNamespace:
        try:
            return self.objects[object_name]
        except KeyError as exc:
            raise MissingObjectError from exc

    def remove_object(self, bucket_name: str, object_name: str) -> None:
        self.removed.append(object_name)

    def presigned_get_object(
        self, bucket_name: str, object_name: str, expires: timedelta
    ) -> str:
        return f"https://storage.invalid/{bucket_name}/{object_name}?seconds={expires.total_seconds():g}"


def _service(client: FakeMinioClient) -> RecordingStorageService:
    return RecordingStorageService(
        RecordingStorageSettings(
            endpoint="minio:9000", access_key="injected", secret_key="injected"
        ),
        client=client,
        lifecycle_config_factory=lambda days: {"expiration_days": days},
    )


def test_object_key_is_deterministic_and_canonical() -> None:
    assert recording_object_key(JOB_UUID) == (
        "recordings/12345678-1234-5678-9234-567812345678.mp4"
    )
    assert recording_object_key(str(JOB_UUID).upper()) == recording_object_key(JOB_UUID)


def test_upload_verifies_and_is_idempotent_by_job_uuid(tmp_path: Path) -> None:
    recording = tmp_path / "final.mp4"
    recording.write_bytes(b"finalized-mp4")
    client = FakeMinioClient()
    service = _service(client)

    first = service.upload_finalized(JOB_UUID, recording)
    second = service.upload_finalized(JOB_UUID, recording)

    assert first.already_present is False
    assert second.already_present is True
    assert first.sha256 == sha256_file(recording)
    assert client.uploads == 1
    assert recording.exists()


def test_existing_job_uuid_with_different_content_is_rejected(tmp_path: Path) -> None:
    recording = tmp_path / "final.mp4"
    recording.write_bytes(b"first")
    client = FakeMinioClient()
    service = _service(client)
    service.upload_finalized(JOB_UUID, recording)
    recording.write_bytes(b"different")

    with pytest.raises(ObjectConflictError):
        service.upload_finalized(JOB_UUID, recording)
    assert client.uploads == 1


def test_local_file_is_deleted_only_after_remote_verification(tmp_path: Path) -> None:
    recording = tmp_path / "final.mp4"
    recording.write_bytes(b"video")
    client = FakeMinioClient()
    service = _service(client)
    original_stat = client.stat_object

    def corrupt_stat(bucket_name: str, object_name: str) -> SimpleNamespace:
        stat = original_stat(bucket_name, object_name)
        return SimpleNamespace(size=stat.size + 1, metadata=stat.metadata)

    client.stat_object = corrupt_stat  # type: ignore[method-assign]
    with pytest.raises(ObjectVerificationError):
        service.upload_verified_and_remove_local(JOB_UUID, recording)
    assert recording.exists()


def test_private_bucket_presigned_download_and_delete(tmp_path: Path) -> None:
    recording = tmp_path / "final.mp4"
    recording.write_bytes(b"video")
    client = FakeMinioClient()
    service = _service(client)

    service.initialize_private_bucket()
    service.upload_finalized(JOB_UUID, recording)
    url = service.presigned_download(JOB_UUID, timedelta(minutes=5))
    service.delete(JOB_UUID)

    assert client.bucket is True
    assert client.policy_deleted is True
    assert client.lifecycle == {"expiration_days": 30}
    assert "seconds=300" in url
    assert client.removed == [recording_object_key(JOB_UUID)]


def test_spool_cleanup_honors_retention_protection_and_delete_bound(tmp_path: Path) -> None:
    old_one = tmp_path / "old-one.mp4"
    old_two = tmp_path / "old-two.mp4"
    protected = tmp_path / "protected.mp4"
    recent = tmp_path / "recent.mp4"
    for path in (old_one, old_two, protected, recent):
        path.write_bytes(path.name.encode())
    now = 2_000_000_000.0
    old_timestamp = now - 8 * 86_400
    for path in (old_one, old_two, protected):
        os.utime(path, (old_timestamp, old_timestamp))
    os.utime(recent, (now, now))

    usage = namedtuple("usage", "total used free")(100, 91, 9)
    lifecycle = LocalSpoolLifecycle(
        SpoolSettings(tmp_path, max_delete_files=1),
        disk_usage=lambda path: usage,
        now=lambda: now,
    )
    result = lifecycle.cleanup_failed({protected})

    assert result.deleted == 1
    assert result.delete_limit_reached is True
    assert protected.exists()
    assert recent.exists()
    assert sum(path.exists() for path in (old_one, old_two)) == 1
    assert result.status.high_water_exceeded is True


def test_spool_scan_is_bounded(tmp_path: Path) -> None:
    now = 2_000_000_000.0
    for index in range(3):
        path = tmp_path / f"old-{index}.mp4"
        path.write_bytes(b"x")
        os.utime(path, (now - 8 * 86_400, now - 8 * 86_400))
    usage = namedtuple("usage", "total used free")(100, 10, 90)
    lifecycle = LocalSpoolLifecycle(
        SpoolSettings(tmp_path, max_scan_files=2),
        disk_usage=lambda path: usage,
        now=lambda: now,
    )

    result = lifecycle.cleanup_failed()

    assert result.scanned == 2
    assert result.scan_limit_reached is True
    assert result.deleted == 2


def test_minio_client_uses_bounded_connect_and_read_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class Timeout:
        def __init__(self, **kwargs: object) -> None:
            captured["timeout"] = kwargs

    class Retry:
        def __init__(self, **kwargs: object) -> None:
            captured["retry"] = kwargs

    class PoolManager:
        def __init__(self, **kwargs: object) -> None:
            captured["pool"] = kwargs

    def minio_factory(endpoint: str, **kwargs: object) -> object:
        captured["endpoint"] = endpoint
        captured["minio"] = kwargs
        return object()

    monkeypatch.setitem(sys.modules, "minio", SimpleNamespace(Minio=minio_factory))
    monkeypatch.setitem(sys.modules, "urllib3", SimpleNamespace(
        Timeout=Timeout, Retry=Retry, PoolManager=PoolManager,
    ))
    settings = RecordingStorageSettings(
        endpoint="minio:9000", access_key="injected", secret_key="injected", network_timeout_seconds=4.0,
    )
    settings.create_client()

    assert captured["timeout"] == {"connect": 4.0, "read": 4.0}
    assert captured["retry"] == {"total": 1, "connect": 1, "read": 1, "redirect": 0}
    assert "http_client" in captured["minio"]  # type: ignore[operator]

pytestmark = pytest.mark.unit
