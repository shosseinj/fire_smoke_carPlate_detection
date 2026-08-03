from __future__ import annotations

import hashlib
import hmac
from urllib.parse import urlsplit

from app.core.source_registry import SourceRecord, SourceRegistry


SOURCE_REF_PREFIX = "recording-source:v1:"


def recording_source_ref(source_uri: str) -> str:
    digest = hashlib.sha256(f"recording-source\0{source_uri}".encode("utf-8")).hexdigest()
    return f"{SOURCE_REF_PREFIX}{digest}"


def resolve_recording_source(registry: SourceRegistry, source_ref: str) -> SourceRecord | None:
    if not source_ref.startswith(SOURCE_REF_PREFIX) or len(source_ref) != len(SOURCE_REF_PREFIX) + 64:
        return None
    matches = [
        source for source in registry.list()
        if hmac.compare_digest(recording_source_ref(source.source_uri), source_ref)
    ]
    if len(matches) > 1:
        raise RuntimeError("recording source reference collision")
    return matches[0] if matches else None


def redacted_recording_source(source_uri: str) -> str:
    try:
        parsed = urlsplit(source_uri)
        hostname = parsed.hostname
        if not parsed.scheme or hostname is None:
            return "منبع محافظت‌شده"
        host = f"[{hostname}]" if ":" in hostname else hostname
        port = f":{parsed.port}" if parsed.port is not None else ""
        suffix = "/…" if parsed.path and parsed.path != "/" else ""
        return f"{parsed.scheme.lower()}://{host}{port}{suffix}"
    except (TypeError, ValueError):
        return "منبع محافظت‌شده"


def safe_recording_source_name(name: str, source_uri: str) -> str:
    candidate = name.strip()
    if not candidate or candidate == source_uri or "://" in candidate:
        return "دوربین"
    try:
        parsed = urlsplit(source_uri)
        secrets = [value for value in (parsed.username, parsed.password) if value]
    except (TypeError, ValueError):
        secrets = []
    return "دوربین" if any(secret in candidate for secret in secrets) else candidate
