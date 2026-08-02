from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

LOGGER = logging.getLogger(__name__)


def live_stream_path(source_id: str, profile: str) -> str:
    if profile not in {"wall", "fullscreen"}:
        raise ValueError("profile must be wall or fullscreen")
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:24]
    return f"live-branch/{profile}/{digest}"


def _is_nvmm_caps(caps: Any) -> bool:
    return caps is not None and "video/x-raw" in caps.to_string() and "memory:NVMM" in caps.to_string()


@dataclass(slots=True)
class LiveBranch:
    source_id: str
    profile: str
    path: str
    pipeline: Any
    tee: Any
    tee_pad: Any
    elements: list[Any]
    sink: Any
    references: set[str] = field(default_factory=set)
    last_heartbeat: dict[str, float] = field(default_factory=dict)
    removing: bool = False


@dataclass(slots=True)
class LiveSource:
    source_id: str
    tee: Any
    pipeline: Any
    native_width: int | None = None
    native_height: int | None = None
    attached: bool = True


class GpuLiveBranchManager:
    """On-demand NVMM branches attached to a decoder-owned tee.

    This manager never observes buffers on the host. The existing AI branch is
    owned by DeepStreamIngestor; this class only owns request pads and its new
    GPU encoder branches.
    """

    REQUIRED_ELEMENTS = (
        "tee",
        "queue",
        "nvvideoconvert",
        "capsfilter",
        "nvv4l2h264enc",
        "h264parse",
        "rtspclientsink",
    )

    def __init__(
        self,
        *,
        publish_base: str,
        browser_base: str = "",
        enabled: bool = False,
        grace_seconds: float = 5.0,
        heartbeat_timeout_seconds: float = 15.0,
        gst_loader: Callable[[], tuple[Any, Any]] | None = None,
    ) -> None:
        self.publish_base = self._validate_base(publish_base)
        self.browser_base = browser_base.rstrip("/")
        self.enabled = bool(enabled)
        self.grace_seconds = max(0.0, float(grace_seconds))
        self.heartbeat_timeout_seconds = max(0.1, float(heartbeat_timeout_seconds))
        self.gst_loader = gst_loader
        self._gst: Any | None = None
        self._glib: Any | None = None
        self._lock = threading.RLock()
        self._sources: dict[str, LiveSource] = {}
        self._branches: dict[tuple[str, str], LiveBranch] = {}
        self._pending_removal: dict[tuple[str, str], float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False

    @staticmethod
    def _validate_base(uri: str) -> str:
        parsed = urlsplit(uri.strip())
        if parsed.scheme.lower() != "rtsp" or not parsed.hostname or parsed.query or parsed.fragment:
            raise ValueError("live branch publish base must be an RTSP URI without query or fragment")
        return uri.strip().rstrip("/")

    def _require_gst(self) -> Any:
        if self._gst is None:
            if self.gst_loader is None:
                raise RuntimeError("GStreamer runtime is not initialized")
            self._gst, self._glib = self.gst_loader()
        return self._gst

    def set_runtime(self, gst: Any, glib: Any | None = None) -> None:
        """Bind the already initialized decoder GStreamer runtime."""
        with self._lock:
            if self._gst is None:
                self._gst = gst
                self._glib = glib

    def _make(self, factory: str, name: str) -> Any:
        element = self._require_gst().ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"Required live branch element is unavailable: {factory}")
        return element

    def browser_url(self, source_id: str, profile: str) -> str:
        path = live_stream_path(source_id, profile)
        if not self.browser_base:
            return path
        return f"{self.browser_base}/{path}"

    def publish_uri(self, source_id: str, profile: str) -> str:
        return f"{self.publish_base}/{live_stream_path(source_id, profile)}"

    def attach_source(
        self,
        source_id: str,
        tee: Any,
        pipeline: Any,
        confirmed_caps: Any | None = None,
    ) -> bool:
        """Register a decoder tee after its decoded pad has negotiated NVMM."""
        if not self.enabled:
            return False
        sink_pad = tee.get_static_pad("sink")
        caps = confirmed_caps or (sink_pad.get_current_caps() if sink_pad is not None else None)
        if not _is_nvmm_caps(caps):
            LOGGER.warning("Refusing live branch attachment without confirmed NVMM: %s", source_id)
            return False
        with self._lock:
            if self._closed:
                return False
            caps_text = caps.to_string()
            width_match = re.search(r"width=(?:\(int\))?(\d+)", caps_text)
            height_match = re.search(r"height=(?:\(int\))?(\d+)", caps_text)
            self._sources[source_id] = LiveSource(
                source_id, tee, pipeline,
                int(width_match.group(1)) if width_match else None,
                int(height_match.group(1)) if height_match else None,
            )
        return True

    def detach_source(self, source_id: str) -> None:
        with self._lock:
            self._sources.pop(source_id, None)
            keys = [key for key in self._branches if key[0] == source_id]
        for key in keys:
            self._remove(key)

    def acquire(self, source_id: str, profile: str, viewer_id: str) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "source_id": source_id, "profile": profile}
        if profile not in {"wall", "fullscreen"}:
            raise ValueError("profile must be wall or fullscreen")
        key = (source_id, profile)
        now = time.monotonic()
        with self._lock:
            source = self._sources.get(source_id)
            if source is None:
                raise RuntimeError("source has no confirmed NVMM decoder tee")
            branch = self._branches.get(key)
            if branch is None:
                branch = self._build(source, profile)
                self._branches[key] = branch
            branch.references.add(viewer_id)
            branch.last_heartbeat[viewer_id] = now
            self._pending_removal.pop(key, None)
            return self._contract(branch)

    def heartbeat(self, source_id: str, profile: str, viewer_id: str) -> bool:
        with self._lock:
            branch = self._branches.get((source_id, profile))
            if branch is None or viewer_id not in branch.references:
                return False
            branch.last_heartbeat[viewer_id] = time.monotonic()
            return True

    def release(self, source_id: str, profile: str, viewer_id: str) -> bool:
        key = (source_id, profile)
        with self._lock:
            branch = self._branches.get(key)
            if branch is None:
                return False
            branch.references.discard(viewer_id)
            branch.last_heartbeat.pop(viewer_id, None)
            if not branch.references:
                self._pending_removal[key] = time.monotonic() + self.grace_seconds
            return True

    def _build(self, source: LiveSource, profile: str) -> LiveBranch:
        Gst = self._require_gst()
        suffix = hashlib.sha256(f"{source.source_id}:{profile}".encode()).hexdigest()[:12]
        queue = self._make("queue", f"live_queue_{suffix}")
        converter = self._make("nvvideoconvert", f"live_convert_{suffix}")
        capsfilter = self._make("capsfilter", f"live_caps_{suffix}")
        encoder = self._make("nvv4l2h264enc", f"live_encoder_{suffix}")
        parser = self._make("h264parse", f"live_parser_{suffix}")
        sink = self._make("rtspclientsink", f"live_sink_{suffix}")
        elements = [queue, converter, capsfilter, encoder, parser, sink]
        tee_pad = None
        try:
            if profile == "wall":
                capsfilter.set_property("caps", Gst.Caps.from_string(
                    "video/x-raw(memory:NVMM),format=NV12,width=260,height=260"
                ))
            else:
                capsfilter.set_property("caps", Gst.Caps.from_string(
                    "video/x-raw(memory:NVMM),format=NV12"
                ))
            queue.set_property("leaky", 2)
            queue.set_property("max-size-buffers", 2)
            find_property = getattr(encoder, "find_property", None)
            if find_property is None or find_property("insert-sps-pps") is not None:
                encoder.set_property("insert-sps-pps", True)
            sink.set_property("location", self.publish_uri(source.source_id, profile))
            sink_find_property = getattr(sink, "find_property", None)
            if sink_find_property is None or sink_find_property("protocols") is not None:
                sink.set_property("protocols", 4)
            if sink_find_property is None or sink_find_property("latency") is not None:
                sink.set_property("latency", 0)
            for element in elements:
                source.pipeline.add(element)
            tee_pad = source.tee.get_request_pad("src_%u")
            if tee_pad is None or not tee_pad.link(queue.get_static_pad("sink")) == Gst.PadLinkReturn.OK:
                raise RuntimeError("could not acquire/link live tee request pad")
            if not queue.link(converter) or not converter.link(capsfilter) or not capsfilter.link(encoder):
                raise RuntimeError("could not link GPU live conversion branch")
            if not encoder.link(parser):
                raise RuntimeError("could not link NVENC to H264 parser")
            if not parser.link(sink):
                raise RuntimeError("could not link H264 parser to RTSP publisher")
            for element in elements:
                element.sync_state_with_parent()
            return LiveBranch(source.source_id, profile, live_stream_path(source.source_id, profile), source.pipeline, source.tee, tee_pad, elements, sink)
        except Exception:
            if tee_pad is not None:
                try:
                    source.tee.release_request_pad(tee_pad)
                except Exception:
                    pass
            for element in elements:
                try:
                    element.set_state(Gst.State.NULL)
                    source.pipeline.remove(element)
                except Exception:
                    pass
            raise

    def _contract(self, branch: LiveBranch) -> dict[str, Any]:
        source = self._sources.get(branch.source_id)
        return {
            "enabled": True, "source_id": branch.source_id, "profile": branch.profile,
            "path": branch.path, "url": self.browser_url(branch.source_id, branch.profile),
            "references": len(branch.references),
            "width": 260 if branch.profile == "wall" else (source.native_width if source else None),
            "height": 260 if branch.profile == "wall" else (source.native_height if source else None),
        }

    def _remove(self, key: tuple[str, str]) -> None:
        with self._lock:
            branch = self._branches.pop(key, None)
            self._pending_removal.pop(key, None)
        if branch is None:
            return
        Gst = self._require_gst()
        probe_id = None
        try:
            probe_id = branch.tee_pad.add_probe(
                Gst.PadProbeType.BLOCK_DOWNSTREAM,
                lambda *_args: Gst.PadProbeReturn.OK,
            )
        except Exception:
            LOGGER.debug("Live branch pad could not be blocked before removal", exc_info=True)
        for element in branch.elements:
            try:
                element.set_state(Gst.State.NULL)
            except Exception:
                pass
        try:
            branch.tee_pad.unlink(branch.elements[0].get_static_pad("sink"))
            branch.tee.release_request_pad(branch.tee_pad)
        except Exception:
            LOGGER.debug("Live branch pad cleanup failed", exc_info=True)
        finally:
            if probe_id is not None:
                try:
                    branch.tee_pad.remove_probe(probe_id)
                except Exception:
                    pass
        for element in reversed(branch.elements):
            try:
                branch.pipeline.remove(element)
            except Exception:
                pass

    def _expire(self) -> None:
        now = time.monotonic()
        with self._lock:
            for branch in self._branches.values():
                expired = {viewer for viewer, stamp in branch.last_heartbeat.items() if now - stamp > self.heartbeat_timeout_seconds}
                branch.references.difference_update(expired)
                for viewer in expired:
                    branch.last_heartbeat.pop(viewer, None)
            keys = [
                key
                for key, deadline in self._pending_removal.items()
                if deadline <= now
                and key in self._branches
                and not self._branches[key].references
            ]
        for key in keys:
            self._remove(key)

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gpu-live-branch", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            self._expire()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            source_ids = list(self._sources)
            keys = list(self._branches)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        for key in keys:
            self._remove(key)
        for source_id in source_ids:
            self.detach_source(source_id)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"enabled": self.enabled, "backend": "gpu_nvmm_nvenc", "namespace": "live-branch", "sources": len(self._sources), "branches": {f"{source}/{profile}": self._contract(branch) for (source, profile), branch in self._branches.items()}}
