from __future__ import annotations

import json
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.runtime import Runtime
from app.schemas import FrameRoundResponse

router = APIRouter(prefix="/api/v1/frame-rounds", tags=["frame-routing"])


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime


@router.post("/jpeg", response_model=FrameRoundResponse)
async def submit_jpeg_round(
    files: Annotated[list[UploadFile], File(description="JPEG/PNG frames in source_ids order")],
    source_ids_json: Annotated[str, Form()],
    round_sequence: Annotated[int, Form()],
    frame_indexes_json: Annotated[str | None, Form()] = None,
    source_times_json: Annotated[str | None, Form()] = None,
    runtime: Runtime = Depends(get_runtime),
) -> FrameRoundResponse:
    try:
        source_ids = list(json.loads(source_ids_json))
        frame_indexes = list(json.loads(frame_indexes_json)) if frame_indexes_json else None
        source_times = list(json.loads(source_times_json)) if source_times_json else None
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"فیلد فرم JSON نامعتبر: {exc}") from exc
    if len(files) != len(source_ids):
        raise HTTPException(status_code=422, detail="تعداد فایل‌ها و source_ids_json متفاوت است")

    frames: list[np.ndarray] = []
    for upload in files:
        content = await upload.read()
        frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=415, detail=f"امکان خواندن {upload.filename} وجود ندارد")
        frames.append(frame)

    try:
        summary = runtime.router.submit_round(
            frames=frames,
            source_ids=source_ids,
            round_sequence=round_sequence,
            frame_indexes=frame_indexes,
            source_times_seconds=source_times,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return FrameRoundResponse(round_sequence=round_sequence, **summary)
