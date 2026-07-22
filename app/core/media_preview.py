from __future__ import annotations

import hashlib
import re
from urllib.parse import urlsplit, urlunsplit


def preview_stream_path(source_id: str) -> str:
    """Return a stable path which does not disclose the camera identifier."""
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:24]
    return f"preview-{digest}"


def preview_publish_uri(base_uri: str, source_id: str) -> str:
    parsed = urlsplit(base_uri.strip())
    if parsed.scheme.lower() != "rtsp" or not parsed.hostname:
        raise ValueError("MEDIA_PREVIEW_PUBLISH_BASE must be an RTSP URI")
    if parsed.query or parsed.fragment:
        raise ValueError("MEDIA_PREVIEW_PUBLISH_BASE cannot contain a query or fragment")
    path = "/".join(
        part for part in (parsed.path.rstrip("/"), preview_stream_path(source_id)) if part
    )
    return urlunsplit((parsed.scheme, parsed.netloc, f"/{path.lstrip('/')}", "", ""))


def redact_preview_uri(uri: str) -> str:
    parsed = urlsplit(uri)
    if parsed.username is None and parsed.password is None:
        return uri
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))


def redact_rtsp_credentials(message: str) -> str:
    """Remove RTSP user-info from a URI or a larger GStreamer message."""
    return re.sub(r"(?i)(rtsp://)[^/@\s]+@", r"\1", message)
