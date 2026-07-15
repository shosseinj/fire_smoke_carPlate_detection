from __future__ import annotations

import itertools
from typing import Any, Mapping, Sequence

import numpy as np

from app.core.router import TaskRouter


class ExistingExtractorBridge:
    """Adapter for an existing FastAPI frame extractor.

    It supports either a fixed source order or a dynamic source_ids list for
    extractors that stop reading disabled sources.
    """

    def __init__(
        self,
        router: TaskRouter,
        source_order: Sequence[str] | None = None,
    ) -> None:
        if source_order is not None and len(source_order) != len(set(source_order)):
            raise ValueError("source_order contains duplicate source IDs")
        self.router = router
        self.source_order = tuple(source_order) if source_order is not None else None
        self._sequence = itertools.count(1)

    def enabled_source_ids(self) -> list[str]:
        """Use this list to tell the main extractor which readers should run."""
        return self.router.registry.enabled_source_ids()

    def submit(
        self,
        frames: Sequence[np.ndarray],
        *,
        source_ids: Sequence[str] | None = None,
        frame_indexes: Sequence[int] | None = None,
        source_times_seconds: Sequence[float | None] | None = None,
        metadata: Sequence[Mapping[str, Any] | None] | None = None,
        round_sequence: int | None = None,
    ) -> dict[str, int]:
        resolved_source_ids = tuple(source_ids) if source_ids is not None else self.source_order
        if resolved_source_ids is None:
            raise ValueError("source_ids must be provided when no fixed source_order was configured")
        if len(frames) != len(resolved_source_ids):
            raise ValueError(
                f"Expected {len(resolved_source_ids)} frames, received {len(frames)}"
            )
        return self.router.submit_round(
            frames=frames,
            source_ids=resolved_source_ids,
            round_sequence=round_sequence or next(self._sequence),
            frame_indexes=frame_indexes,
            source_times_seconds=source_times_seconds,
            metadata=metadata,
        )
