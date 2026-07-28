from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("uvicorn.error")


@dataclass(slots=True, frozen=True)
class RawStreamPacket:
    source_id: str
    codec: str
    pts_ns: int | None
    dts_ns: int | None
    duration_ns: int | None
    is_keyframe: bool
    data: bytes


class RawStreamRouter:
    """
    Receives encoded H.264/H.265 access units from DeepStream/GStreamer.

    This component must never perform decode or AI inference.
    """

    def __init__(
        self,
        *,
        project_root: Path,
        enabled: bool = True,
        recording_enabled: bool = False,
        relay_enabled: bool = False,
        clip_buffer_enabled: bool = True,
    ) -> None:
        self.project_root = project_root
        self.enabled = enabled
        self.recording_enabled = recording_enabled
        self.relay_enabled = relay_enabled
        self.clip_buffer_enabled = clip_buffer_enabled

        self._lock = threading.RLock()
        self._running = False

    def start(self) -> None:
        if not self.enabled:
            return

        with self._lock:
            if self._running:
                return

            self._running = True
            LOGGER.info("RAW_STREAM_ROUTER_STARTED")

    def close(self) -> None:
        with self._lock:
            if not self._running:
                return

            self._running = False
            LOGGER.info("RAW_STREAM_ROUTER_STOPPED")

    def publish_packet(
        self,
        *,
        source_id: str,
        codec: str,
        pts_ns: int | None,
        dts_ns: int | None,
        duration_ns: int | None,
        is_keyframe: bool,
        data: bytes,
    ) -> None:
        if not self.enabled or not self._running:
            return

        packet = RawStreamPacket(
            source_id=source_id,
            codec=codec,
            pts_ns=pts_ns,
            dts_ns=dts_ns,
            duration_ns=duration_ns,
            is_keyframe=is_keyframe,
            data=data,
        )

        if self.clip_buffer_enabled:
            self._publish_to_clip_buffer(packet)

        if self.recording_enabled:
            self._publish_to_recorder(packet)

        if self.relay_enabled:
            self._publish_to_relay(packet)

    def _publish_to_clip_buffer(self, packet: RawStreamPacket) -> None:
        # مرحله بعد: per-camera encoded GOP ring buffer
        pass

    def _publish_to_recorder(self, packet: RawStreamPacket) -> None:
        # مرحله بعد: segment writer / splitmuxsink
        pass

    def _publish_to_relay(self, packet: RawStreamPacket) -> None:
        # مرحله بعد: RTSP/WebRTC/HLS publisher
        pass

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "running": self._running,
                "recording_enabled": self.recording_enabled,
                "relay_enabled": self.relay_enabled,
                "clip_buffer_enabled": self.clip_buffer_enabled,
            }