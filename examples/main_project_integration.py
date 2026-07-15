"""Integration example for an existing FastAPI frame extractor."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.bridge import ExistingExtractorBridge
from app.main import runtime

# Fixed-order mode: use when the extractor always returns all configured sources.
SOURCE_ORDER = [f"camera-{index:02d}" for index in range(1, 51)]
bridge = ExistingExtractorBridge(runtime.router, SOURCE_ORDER)


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.start()
    try:
        yield
    finally:
        runtime.close()


app = FastAPI(lifespan=lifespan)


def on_fixed_order_frame_round(frames, frame_indexes=None, source_times_seconds=None):
    """For [fN_c1, fN_c2, ...] with a permanent source order."""
    return bridge.submit(
        frames,
        frame_indexes=frame_indexes,
        source_times_seconds=source_times_seconds,
    )


def on_dynamic_frame_round(frames, source_ids, frame_indexes=None):
    """For an extractor that stops disabled readers and returns active IDs."""
    return bridge.submit(
        frames,
        source_ids=source_ids,
        frame_indexes=frame_indexes,
    )


def sources_the_extractor_should_enable() -> list[str]:
    """The main extractor should synchronize its readers to this list."""
    return bridge.enabled_source_ids()


# Important:
# 1. Do not JPEG-encode frames when both modules run in the same Python process.
# 2. source_ids order must exactly match frames order for every round.
# 3. The routing API changes the registry immediately; the extractor should use
#    sources_the_extractor_should_enable() to start/stop actual readers.
