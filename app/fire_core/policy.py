from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FireSmokePolicyConfig:
    """Admin-controlled count thresholds for one rolling detection window."""

    window_seconds: float = 3.0
    low_count: int = 5
    medium_count: int = 10
    high_count: int = 20

    def validated(self) -> "FireSmokePolicyConfig":
        if not 0.25 <= float(self.window_seconds) <= 300.0:
            raise ValueError("window_seconds must be between 0.25 and 300")
        if not 1 <= int(self.low_count) < int(self.medium_count) < int(self.high_count):
            raise ValueError("counts must satisfy 1 <= low_count < medium_count < high_count")
        return FireSmokePolicyConfig(
            window_seconds=float(self.window_seconds),
            low_count=int(self.low_count),
            medium_count=int(self.medium_count),
            high_count=int(self.high_count),
        )
