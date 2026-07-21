# Verified Architecture

Status: initialized from repository inspection.

## Project Identity

- **Name:** Unified Video AI Task Router
- **Version:** 2.0.0 (from FastAPI app metadata)
- **Language:** Python 3.10+
- **Stack:** FastAPI, PyTorch, TensorRT 10.7, ONNX Runtime, DeepStream, OpenCV
- **Runtime:** NVIDIA GPU inside DeepStream Docker container
- **Repository root:** `C:\Users\jafari.h\Desktop\ai_project\testing\merged_video_ai_router`

## Entry Points

### Application

| Entry Point | File | Description |
|---|---|---|
| Dev server | `run.py` | `uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)` |
| Production container | `Dockerfile.deepstream` | DeepStream-based NVIDIA container |
| Development container | `Dockerfile` | Python 3.12-slim (CPU-only) |
| Compose service | `docker-compose.yml` | Service `video-ai-router`, GPU reserved |

### API Entry Point

`app/main.py` → `app = FastAPI(title="Unified Video AI Task Router", version="2.0.0")`  
`app.runtime.build_runtime()` → returns `Runtime` dataclass  

All routers are included in `app/main.py`. The `runtime` module-level variable is imported by all API routers via `from app.main import runtime`.

## Directories

| Path | Purpose |
|---|---|
| `app/` | Application package |
| `app/main.py` | FastAPI app, lifespan, router registration, health endpoint |
| `app/runtime.py` | Component construction (Runtime dataclass, build_runtime) |
| `app/config.py` | Settings dataclass (frozen, slots), env-based config |
| `app/schemas.py` | Pydantic request/response models for cameras, sources |
| `app/api/` | 14 router modules (see Endpoints section) |
| `app/core/` | Core logic (router, worker, buffers, ingestor, stores) |
| `app/processors/` | Inference processors (fire_smoke, plate, face_recognition, mock) |
| `app/fire_core/` | Fire/smoke policy configuration |
| `app/observability/` | Empty (placeholder for observability) |
| `app/web/` | Static web assets (dashboard.html) |
| `tests/` | 15 test files |
| `scripts/` | build_all_engines, smoke_test, mock_load_test, agentic tools |
| `weights/` | Model files (fire_smoke, plate_detector, vehicle_detector, face_recognition) |
| `data/` | SQLite databases, JSON source registry, Qdrant vector store |
| `saved_media/` | Snapshots, crop images, evidence videos |
| `camera_configs/` | Camera configuration files |
| `docs/` | Generated documentation |
| `models/` | Additional model data |
| `.agentic/` | Agentic workflow state |

## Services

| Service | Configuration | Default |
|---|---|---|
| FastAPI server | uvicorn on port 8000 | 0.0.0.0:8000 |
| SQLite (cameras) | `CAMERA_DB_PATH` | `data/cameras.sqlite3` |
| SQLite (plate_logs/fire/human) | `PLATE_LOG_DB_PATH` | `plate_logs.sqlite3` |
| Qdrant (face vectors) | `FACE_QDRANT_PATH` or URL | `data/qdrant` |
| DeepStream (GPU decode) | `VIDEO_INGEST_BACKEND=deepstream` | Production path |
| OpenCV (CPU decode) | `VIDEO_INGEST_BACKEND=opencv` (currently hardcoded to 'deepstream' in config.py) | Dev fallback |

## Processing Pipeline

```
Video Source (MP4/RTSP)
  │
  ▼
VideoFileIngestor / DeepStreamIngestor
  │  Decodes at VIDEO_INGEST_FPS (default 5)
  │  Preview at VIDEO_PREVIEW_FPS (default 25) for play-only
  │  Per-camera pipeline control
  ▼
TaskRouter.submit_round(frames, source_ids)
  │
  ├─► LatestBuffer (per task: fire_smoke, plate_recognition, face_recognition)
  │     Keeps newest frame per source; replaces stale waiting frames
  │
  ▼
TaskWorker (per task, background thread)
  │  Micro-batches: batch_size frames or max_wait_ms timeout
  │
  ▼
BatchProcessor (varies by task)
  │  fire_smoke: FireSmokeProcessor (YOLO-based fire/smoke detection)
  │  plate_recognition: PlateRecognitionProcessor (vehicle→plate→OCR cascade)
  │  face_recognition: FaceRecognitionProcessor (human→face→quality→embedding→search)
  │
  ▼
TaskResult → ResultStore + Broadcast + Log Stores
  │
  ├─► AnnotatedBroadcastHub (WebSocket + MJPEG streams)
  ├─► PlateLogStore (SQLite persistence)
  ├─► FireSmokeLogStore (incident severity tracking)
  └─► HumanLogStore (tracking video/snapshot evidence)
```

## Three Inference Workers

1. **fire_smoke** - YOLO detector for fire and smoke classes
2. **plate_recognition** - Vehicle detector → plate detector → Iranian plate OCR cascade
3. **face_recognition** - Human detector (YOLO-pose) → face detector → ArcFace embedding → Qdrant vector search

Model fallback: `.engine` → `.onnx` → `.pt` (when fallbacks enabled)

## Endpoints

### API Routers (14 routers registered in `app/main.py`)

| Router prefix | Tags | File |
|---|---|---|
| `/api/v1/diagnostics` | system-diagnostics | `app/api/diagnostics.py` |
| `/api/v1/tests` | processor-tests | `app/api/processor_tests.py` |
| `/api/v1/settings` | general-settings | `app/api/general_settings.py` |
| `/api/v1/models` | model-management | `app/api/models.py` |
| `/api/v1/sources` | sources | `app/api/sources.py` |
| `/api/v1/cameras` | cameras | `app/api/cameras.py` |
| `/api/v1/frame-rounds` | frame-routing | `app/api/frames.py` |
| `/api/v1/results` / `/api/v1/router` | results | `app/api/results.py` |
| `/dashboard` / `/api/v1/broadcast` | annotated-broadcast | `app/api/broadcast.py` |
| `/api/v1/plate-logs` | plate-logs | `app/api/plate_logs.py` |
| `/api/v1/plate-settings` | plate-settings | `app/api/plate_settings.py` |
| `/api/v1/plate-settings` | plate-settings | `app/api/plate_settings.py` |
| `/api/v1/fire-smoke-logs` / `/api/v1/fire-smoke` | fire-smoke | `app/api/fire_smoke_logs.py` |
| `/api/v1/faces` | face-recognition | `app/api/faces.py` |
| `/api/v1/humans` | human-tracking | `app/api/humans.py` |

### Non-API Endpoints

| Path | Description |
|---|---|
| `/` | Redirects to `/dashboard` |
| `/health` | Health and runtime counters |
| `/docs` | Swagger UI |
| `/media` | Static files (snapshots, evidence media) |

### WebSockets

| Path | Protocol | Description |
|---|---|---|
| `/api/v1/broadcast/ws` | Binary + JSON text | Multiplexed annotated camera frames + state events |
| `/api/v1/results/ws` | JSON text | Live inference results |

## UI

- **Dashboard:** `/dashboard` - Annotated camera wall with per-tile task badges
- **Swagger:** `/docs` - Full API documentation and testing

## Workers and Queues

| Worker | Type | Key properties |
|---|---|---|
| TaskWorker × 3 | Background threading | `batch_size`, `max_wait_ms` |
| ModelConversionManager | Background threading | Single queue, PT→ONNX→engine pipeline |
| FireSmokeLogStore writer | Background thread | Bounded queue for snapshot/media writes |
| HumanLogStore writer | Background thread | Bounded queue for video/snapshot evidence |

## Storage

| Storage | Path/Source | Contents |
|---|---|---|
| SQLite (cameras) | `CAMERA_DB_PATH` → `data/cameras.sqlite3` | Camera registry, model selections, plate settings |
| SQLite (logs) | `PLATE_LOG_DB_PATH` → `plate_logs.sqlite3` | Plate logs, fire/smoke logs, human/track logs |
| Qdrant | `FACE_QDRANT_PATH` → `data/qdrant` | Face embedding vectors (collection: `faces`) |
| Filesystem | `weights/` | Model artifacts (.engine, .onnx, .pt) |
| Filesystem | `saved_media/` | Snapshots, evidence crops, track videos |
| Filesystem | `data/` | Runtime data (JSON source registry, Qdrant store) |

## Observability

- `GET /health` - Status, processor mode, registered sources, router/broadcast/ingestor status, log counts
- `GET /api/v1/diagnostics/overview` - Full runtime introspection
- `GET /api/v1/diagnostics/checks` - Safe section-by-section maintenance checks
- `GET /api/v1/settings/general` - Combined configuration snapshot
- `GET /api/v1/models/settings` - Model selection snapshot
- `GET /api/v1/router/status` - Router and worker statistics

## Deployment

- **Production image:** `Dockerfile.deepstream` (NVIDIA DeepStream base)
- **Dev image:** `Dockerfile` (python:3.12-slim)
- **Orchestration:** `docker-compose.yml` (service `video-ai-router`, GPU 1)
- **Ports:** 8000 (API), SHM size 2GB
- **Volumes:** app (read-only), data, weights (read-only), saved_media
- **Environment:** 110+ env vars via `.env.example`
- **Entry:** `python run.py` (or compose CMD)

## Test Architecture

- **Framework:** pytest 8.3+
- **Test client:** httpx (via FastAPI TestClient/Starlette)
- **Test files:** 15 files in `tests/`
- **Smoke test:** `python scripts/smoke_test.py`
- **Mock load test:** `PROCESSOR_MODE=mock python scripts/mock_load_test.py`
- **Config:** `pyproject.toml` with `pythonpath = ["."]`, `testpaths = ["tests"]`
- **Dev req:** `requirements-dev.txt` (pytest, httpx)

## Known Unknowns

- Actual GPU model and driver version on deployment host
- `agent_test_failures.log` path is defined in AGENTS.md but does not exist yet
- Qdrant remote URL (`FACE_QDRANT_URL`) is not configured by default; local path is used
- DeepStream container not built yet in this environment (no NVIDIA GPU access detected)
- TensorRT engines were built on a different machine/container than current host
