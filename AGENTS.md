# AGENTS.md — Project Context for AI Assistants

## Project Identity

- **Name:** Unified Video AI Task Router
- **Version:** 2.0.0
- **Stack:** FastAPI + PyTorch + TensorRT + ONNX + DeepStream
- **Python:** 3.10+
- **Container:** Docker with NVIDIA GPU (DeepStream base)

## What This System Does

Ingests multiple video streams (local MP4 files or RTSP cameras), routes frames to three independent batched AI inference workers, and broadcasts annotated results via REST + WebSocket to a live dashboard.

## Three AI Pipelines

### 1. Fire & Smoke Detection (`app/processors/fire_smoke.py`)

- YOLO detection → stable object tracking → severity analyzer → incident lifecycle
- Config: window_seconds, low/medium/high counts, confidence thresholds

### 2. Iranian Plate Recognition (`app/processors/plate.py`)

- Vehicle detection (YOLO) → crop → plate detection (YOLO) → crop → OCR (Hezar)
- Validates Iranian format: 2 digits + Persian letter + 5 digits
- Per-camera policy overrides via `PlateSettingsStore`

### 3. Face Recognition (`app/processors/face_recognition.py`)

- YOLO-pose human detection → ByteTrack tracking → face detection on human ROIs → quality gate (blur, yaw/pitch/roll, landmarks, eye distance) → ArcFace embedding → vector search (Qdrant or SQLite fallback) → identity tracking
- Quality settings stored in `FaceQualitySettingsStore`

## Architecture Patterns

### Micro-batching

- Workers wait up to `max_wait_ms` (~25ms) to accumulate a batch before inference.
- `LatestPerSourceBuffer` keeps only the newest frame per source per task (fair queuing).

### Lazy Loading

- Models are loaded on first `process_batch()` call via `_ensure_dependencies()`.
- App starts even when model files are missing.
- `preload()` called at startup for face processor to fail early.

### Model Fallback Chain

- Format priority: `.engine` → `.onnx` → `.pt` (configurable via `MODEL_PREFERRED_FORMAT`, `MODEL_ALLOW_ONNX_FALLBACK`, `MODEL_ALLOW_PT_FALLBACK`).
- Managed by `ModelManager` in `app/core/model_management.py`.

### Ingestion Backends

- **DeepStream** (`Dockerfile.deepstream`): GPU-accelerated decode, production.
- **OpenCV** (`Dockerfile`): CPU decode, testing/development.

## Key Files

### App Entry

- `app/main.py` — FastAPI app, router mounting, `/health`
- `app/runtime.py` — `Runtime` dataclass, `build_runtime()`, all component wiring

### Core

- `app/core/router.py` — `TaskRouter`: fans frames to per-task workers
- `app/core/worker.py` — `TaskWorker`: threaded batch processor
- `app/core/latest_buffer.py` — `LatestPerSourceBuffer`: fair frame queue
- `app/core/types.py` — `TaskName` enum, `FramePacket`, `TaskResult`
- `app/core/broadcast.py` — `AnnotatedBroadcastHub`: WebSocket + JPEG broadcast
- `app/core/result_store.py` — `ResultStore`: in-memory ring buffer
- `app/core/source_registry.py` — `SourceRegistry`: camera CRUD + SQLite
- `app/core/model_management.py` — `ModelManager` + `ModelConversionManager`

### Processors

- `app/processors/face_recognition.py` — `FaceRecognitionProcessor` (~2087 lines)
- `app/processors/fire_smoke.py` — `FireSmokeProcessor` (~519 lines)
- `app/processors/plate.py` — `PlateRecognitionProcessor` (~709 lines)
- `app/processors/base.py` — `BatchProcessor` abstract base
- `app/processors/ultralytics_loader.py` — Thread-safe YOLO imports

### API Routers

- `app/api/diagnostics.py` — `/api/v1/diagnostics/*`
- `app/api/processor_tests.py` — `/api/v1/tests/*` (per-stage pipeline tests)
- `app/api/frames.py` — `/api/v1/frame-rounds/jpeg`
- `app/api/cameras.py` — `/api/v1/cameras/*`
- `app/api/faces.py` — Face enrollment endpoints
- `app/api/humans.py` — Human tracking history
- `app/api/plate_logs.py`, `app/api/fire_smoke_logs.py` — Log queries
- `app/api/plate_settings.py` — Plate policy overrides
- `app/api/models.py` — Model catalog + conversion jobs
- `app/api/broadcast.py` — Dashboard + WebSocket streams

### Config

- `app/config.py` — `Settings` dataclass (193 lines, all env-var driven)
- `.env.example` — Documented defaults
- `docker-compose.yml` — Production deployment config

## Coding Conventions

### General

- **No comments in code** unless absolutely necessary for clarity.
- **No emojis** in code or documentation.
- **Type hints** required on all function signatures.
- **`from __future__ import annotations`** at top of every file.
- Dataclasses use `slots=True` and `frozen=True` for settings/value objects.

### Naming

- `snake_case` for variables, functions, methods, modules.
- `PascalCase` for classes.
- Private methods use leading underscore: `_ensure_dependencies()`, `_predict_yolo()`.
- Router files use plural resource names: `faces.py`, `cameras.py`, `plate_logs.py`.

### API Patterns

- Router module follows this template:
  ```python
  router = APIRouter(prefix="/api/v1/<resource>", tags=["<tag-group>"])
  def get_runtime() -> Runtime:
      from app.main import runtime
      return runtime
  ```
- Use `Annotated` for `UploadFile`, `Form`, `File`, `Depends` parameters.
- Response models use Pydantic `BaseModel` (see `app/schemas.py`).

### Imports

- Standard library, then third-party, then local (alphabetical groups).
- Lazy imports inside `get_runtime()` to avoid circular imports.

## TensorRT Notes

- Engine files are **version-specific** and **platform-specific**.
- Current production engines built with TRT 10.3 (DeepStream container).
- `trtexec` and Python `tensorrt` package must match versions.
- Use `trtexec --getPlanVersionOnly --loadEngine=file.engine` to check.
- Dynamic batch engines (min/opt/max shapes) preferred over fixed batch.
- ArcFace preprocessing: resize to 112x112, `blobFromImages` with `scalefactor=1/128`, `mean=127.5`, `swapRB=True`.

## Testing

- Test endpoints available at `/api/v1/tests/*` (Swagger @ `/docs`).
- Per-stage tests for each pipeline: human detection, face pipeline, fire/smoke, vehicle detection, plate pipeline.
- Model file existence checks per pipeline.
- No formal test suite (pytest) found — tests are done via API calls.

## Common Issues

1. **DeepStream base image has TRT 10.3** — don't install tensorrt-cu12 in Dockerfile.deepstream.
