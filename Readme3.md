# Unified Video AI Task Router

Version **2.0.0** — Real-time multi-camera AI video analytics server with three parallel inference pipelines.

---

## Overview

Ingests video streams (local files / RTSP), routes frames to independent batched AI workers, and broadcasts annotated results via **REST + WebSocket** to a live dashboard. Built on FastAPI with GPU-accelerated inference (TensorRT / ONNX).

---

## AI Pipelines

### 1. Fire & Smoke Detection

- **Model:** YOLO-NAS / YOLO11 (TensorRT engine or ONNX)
- **Pipeline:** Detection → Stable object tracking → Severity analyzer → Incident lifecycle
- **Configurable:** Fire/smoke confidence thresholds, severity windows, incident start/end policies

### 2. Iranian Plate Recognition

- **Models:** YOLO vehicle detector + YOLO plate detector + Hezar OCR
- **Pipeline:** Vehicle detection → Crop → Plate detection → Crop → OCR
- **Validation:** Iranian plate format (2 digits + Persian letter + 5 digits)
- **Configurable per camera:** Confidence thresholds, class IDs, crop padding

### 3. Face Recognition

- **Models:** YOLO-pose human detector + YOLO face detector + ArcFace embedder
- **Pipeline:** Human detection → ByteTrack tracking → Face detection on human ROIs → Quality gate (blur, pose, landmarks) → ArcFace embedding → Vector search (Qdrant / SQLite fallback) → Identity tracking
- **Quality gates:** Blur detection, yaw/pitch/roll limits, eye distance, landmark requirements
- **Storage:** Qdrant vector DB (remote or embedded) with SQLite fallback

---

## Architecture

```
Video Sources (MP4 / RTSP)
        |
   Video Ingestor
  (DeepStream or OpenCV)
        |
   Task Router
      /    |    \
     /     |     \
fire_smoke  plate   face_recognition
worker     worker      worker
   |         |           |
micro-batch micro-batch micro-batch
   \         |           /
    \        |          /
      Result Store + Broadcast Hub
              |
      REST API + WebSocket
              |
         Dashboard
```

### Key Design Decisions

- **Fair queuing** — Each source contributes one newest frame per task round; fast cameras cannot starve slow ones.
- **Stale-frame replacement** — Old frames are dropped when new ones arrive before inference completes.
- **Micro-batching** — Workers wait up to ~25ms to fill a batch before inference (configurable).
- **Dynamic task assignment** — Tasks can be changed per camera at runtime without restart.
- **Lazy model loading** — App starts even if models are missing; loading happens on first inference.
- **Mock mode** — `PROCESSOR_MODE=mock` runs without GPU or model files for testing.

---

## API Endpoints

Full Swagger documentation available at `/docs` when the server is running.

### System

| Method | Path                                       | Description                                   |
| ------ | ------------------------------------------ | --------------------------------------------- |
| GET    | `/health`                                  | Health check and runtime counters             |
| GET    | `/api/v1/diagnostics/overview`             | Read-only runtime snapshot                    |
| GET    | `/api/v1/diagnostics/checks`               | Organized maintenance checks (pass/warn/fail) |
| POST   | `/api/v1/diagnostics/cameras/{id}/restart` | Restart one DeepStream pipeline               |

### Processor Tests

| Method | Path                                    | Description                         |
| ------ | --------------------------------------- | ----------------------------------- |
| GET    | `/api/v1/tests/all`                     | All model checks in one call        |
| GET    | `/api/v1/tests/face/models`             | Face model files + TensorRT version |
| POST   | `/api/v1/tests/face/human-detection`    | Test human YOLO-pose only           |
| POST   | `/api/v1/tests/face/full-pipeline`      | Test full face pipeline             |
| GET    | `/api/v1/tests/fire-smoke/models`       | Fire/smoke model files              |
| POST   | `/api/v1/tests/fire-smoke/detection`    | Test fire/smoke detection           |
| GET    | `/api/v1/tests/plate/models`            | Vehicle/plate/OCR model files       |
| POST   | `/api/v1/tests/plate/vehicle-detection` | Test vehicle detection only         |
| POST   | `/api/v1/tests/plate/full-pipeline`     | Test full plate pipeline            |

### Cameras & Sources

| Method         | Path                           | Description                   |
| -------------- | ------------------------------ | ----------------------------- |
| GET/POST       | `/api/v1/cameras`              | List / create cameras         |
| GET/PUT/DELETE | `/api/v1/cameras/{id}`         | Read / update / delete camera |
| POST           | `/api/v1/cameras/assign-tasks` | Assign AI tasks to cameras    |

### Frame Routing

| Method | Path                        | Description                      |
| ------ | --------------------------- | -------------------------------- |
| POST   | `/api/v1/frame-rounds/jpeg` | Submit JPEG frames for inference |

### Results & Broadcast

| Method | Path                      | Description                     |
| ------ | ------------------------- | ------------------------------- |
| GET    | `/api/v1/results/recent`  | Recent inference results        |
| GET    | `/api/v1/results/workers` | Worker status                   |
| WS     | `/api/v1/results/ws`      | WebSocket for real-time results |

### Logs & Settings

| Method   | Path                      | Description                           |
| -------- | ------------------------- | ------------------------------------- |
| GET/POST | `/api/v1/plate-logs`      | Query / search plate records          |
| GET      | `/api/v1/fire-smoke-logs` | Fire/smoke incident logs              |
| GET      | `/api/v1/faces`           | Face enrollment & identity management |
| GET      | `/api/v1/humans`          | ByteTrack identity history & media    |
| GET      | `/api/v1/plate-settings`  | Plate detection policies              |

### Model Management

| Method | Path                             | Description                          |
| ------ | -------------------------------- | ------------------------------------ |
| GET    | `/api/v1/models/catalog`         | Model catalog browser                |
| POST   | `/api/v1/models/convert`         | Submit PT → TensorRT/ONNX conversion |
| GET    | `/api/v1/models/conversion-jobs` | Conversion job history               |

### Dashboard

| Path         | Description                |
| ------------ | -------------------------- |
| `/dashboard` | Live annotated camera wall |

---

## Configuration

All configuration is via environment variables (see `.env.example` for the full list).

### Core

| Variable               | Default      | Description                              |
| ---------------------- | ------------ | ---------------------------------------- |
| `PROCESSOR_MODE`       | `real`       | `real` or `mock` (mock runs without GPU) |
| `VIDEO_INGEST_BACKEND` | `deepstream` | `deepstream` or `opencv`                 |
| `VIDEO_INGEST_FPS`     | `5`          | Target ingestion FPS                     |
| `VIDEO_LOOP`           | `true`       | Loop video files                         |

### Model Fallback

| Variable                    | Default  | Description                        |
| --------------------------- | -------- | ---------------------------------- |
| `MODEL_PREFERRED_FORMAT`    | `engine` | Preferred model format             |
| `MODEL_ALLOW_ONNX_FALLBACK` | `true`   | Fall back to ONNX if engine fails  |
| `MODEL_ALLOW_PT_FALLBACK`   | `true`   | Fall back to PyTorch if ONNX fails |

---

## Deployment

### Docker (Production)

```bash
docker compose up --build
```

Uses `Dockerfile.deepstream` with NVIDIA DeepStream base image and GPU acceleration.

### Docker (Slim / Testing)

```bash
docker build -f Dockerfile -t video-ai-router .
```

Uses `python:3.12-slim` with TensorRT pip package.

### Environment

Copy `.env.example` to `.env` and adjust paths:

```bash
cp .env.example .env
```

---

## Model Export

TensorRT engines are platform-specific. Use the included export worker:

```bash
python -m app.model_export_worker \
  --source model.pt \
  --target model.engine \
  --format engine \
  --imgsz 640 \
  --batch 8 \
  --workspace 4 \
  --device 0 \
  --half \
  --dynamic
```

Or use `trtexec` directly for advanced options:

```bash
trtexec \
  --onnx=model.onnx \
  --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:4x3x640x640 \
  --maxShapes=images:8x3x640x640 \
  --saveEngine=model.engine
```

---

## Dependencies

- **Python 3.10+**
- **PyTorch 2.5.1** (CUDA 12.4)
- **TensorRT** (10.3 or 11.1)
- **Ultralytics 8.4.83**
- **ONNX Runtime GPU 1.22.0**
- **FastAPI 0.138.2**
- **OpenCV 4.13.0**
- **Qdrant Client 1.18.0**
- **Hezar 0.44.0** (Persian OCR)
