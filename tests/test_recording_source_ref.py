from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.recording_source_ref import (
    SOURCE_REF_PREFIX,
    recording_source_ref,
    redacted_recording_source,
    resolve_recording_source,
    safe_recording_source_name,
)


def test_credential_bearing_rtsp_has_stable_opaque_namespaced_reference() -> None:
    uri = "rtsp://admin:p%40ssword@camera.internal:8554/private/path?token=secret"
    reference = recording_source_ref(uri)
    assert reference == recording_source_ref(uri)
    assert reference.startswith(SOURCE_REF_PREFIX) and len(reference) == len(SOURCE_REF_PREFIX) + 64
    assert not any(value in reference for value in ("admin", "ssword", "camera.internal", "secret"))


def test_resolution_scans_authoritative_registry_and_never_accepts_redacted_uri() -> None:
    source = SimpleNamespace(source_uri="rtsp://admin:secret@camera.internal/live")
    registry = SimpleNamespace(list=lambda: [source])
    assert resolve_recording_source(registry, recording_source_ref(source.source_uri)) is source
    assert resolve_recording_source(registry, "rtsp://camera.internal/…") is None


def test_collision_is_detected_instead_of_selecting_ambiguously(monkeypatch: pytest.MonkeyPatch) -> None:
    first = SimpleNamespace(source_uri="rtsp://one")
    second = SimpleNamespace(source_uri="rtsp://two")
    registry = SimpleNamespace(list=lambda: [first, second])
    monkeypatch.setattr("app.core.recording_source_ref.recording_source_ref", lambda _uri: f"{SOURCE_REF_PREFIX}{'a' * 64}")
    with pytest.raises(RuntimeError, match="collision"):
        resolve_recording_source(registry, f"{SOURCE_REF_PREFIX}{'a' * 64}")


def test_display_value_removes_userinfo_query_and_private_path_details() -> None:
    display = redacted_recording_source("rtsp://admin:secret@[2001:db8::1]:8554/private/path?token=hidden")
    assert display == "rtsp://[2001:db8::1]:8554/…"
    assert not any(value in display for value in ("admin", "secret", "private", "token", "hidden"))


def test_credential_bearing_or_uri_derived_name_is_replaced() -> None:
    uri = "rtsp://admin:secret@camera.internal/live"
    assert safe_recording_source_name(uri, uri) == "دوربین"
    assert safe_recording_source_name("admin secret camera", uri) == "دوربین"
    assert safe_recording_source_name("دوربین ورودی", uri) == "دوربین ورودی"

pytestmark = pytest.mark.unit
