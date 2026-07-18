# Unified Video AI Task Router

## Python compatibility

The application supports Python 3.10 and newer. Python's `enum` module is part of the standard library; do not install the unrelated PyPI package named `enum`.

This project merges the uploaded fire/smoke detector and Iranian plate-recognition service into one FastAPI-compatible routing module. The included configuration runs eight MP4 files as looping cameras, while the router remains scalable to larger deployments.

It is built around the frame contract already used by the main project:

```text
round 1 -> [f1_c1, f1_c2, f1_c3, ...]
round 2 -> [f2_c1, f2_c2, f2_c3, ...]
round N -> [fN_c1, fN_c2, fN_c3, ...]
```

The list must always be accompanied by the matching source-ID order. The router does not guess camera identity from array position.

## What the merged project does

- Enables or disables each camera/video independently.
- Assigns zero, one, or multiple AI tasks to each source.
- Routes the same captured frame to both models when a source has both tasks.
- Creates separate real-time micro-batches for fire/smoke and plate recognition.
- Keeps only the newest waiting frame for each source to avoid growing latency.
- Uses fair source ordering so high-FPS cameras do not starve slower sources.
- Supports dynamic task changes without restarting workers or reloading the API.
- Stores recent results and broadcasts new results over WebSocket.
- Starts even when model files are missing; models are loaded lazily on first inference.
- Includes mock mode, tests, and a smoke test that run without GPU or model files.

## Configured eight-camera demo

`data/sources.json` and `examples/initial_sources.json` select eight of the provided videos:

| Source      | Video          | Tasks                             |
| ----------- | -------------- | --------------------------------- |
| `camera-01` | `smoke1.mp4`   | `fire_smoke`                      |
| `camera-02` | `smoke2.mp4`   | `fire_smoke`                      |
| `camera-03` | `sdf.mp4`      | `fire_smoke`                      |
| `camera-04` | `fg.mp4`       | `plate_recognition`               |
| `camera-05` | `yt.mp4`       | `plate_recognition`               |
| `camera-06` | `bucket11.mp4` | `plate_recognition`               |
| `camera-07` | `etry.mp4`     | `fire_smoke`, `plate_recognition` |
| `camera-08` | `test1.mp4`    | `fire_smoke`, `plate_recognition` |

For one extraction round containing all eight cameras, the router creates:

- 5 fire/smoke task submissions.
- 5 plate-recognition task submissions.
- 10 total task submissions. Cameras 7 and 8 send the same decoded frame to both models.

## Real-time batching model

Each task has its own worker:

```text
built-in video cameras or existing extractor
        |
        | one list + source IDs per extraction round
        v
    TaskRouter
      /    \
     /      \
fire/smoke   plate recognition
latest-only  latest-only
buffer       buffer
     |          |
micro-batch  micro-batch
worker       worker
     \          /
      result store
          |
  REST + WebSocket
```

Default worker settings:

```text
fire/smoke batch size: 8
plate batch size:      8
maximum batch wait:    25 ms
```

A batch is sent immediately when full. Otherwise, it is sent when the short wait window expires. When a source produces another frame while its previous frame is still waiting, the older waiting frame is replaced. This prevents an overloaded model from processing stale video seconds later.

## Integration into the existing main FastAPI project

Use direct Python calls. Do not send frames through HTTP when both modules run in the same process.

```python
from app.core.bridge import ExistingExtractorBridge
from app.runtime import build_runtime

runtime = build_runtime()
runtime.start()

source_order = [f"camera-{index:02d}" for index in range(1, 51)]
bridge = ExistingExtractorBridge(runtime.router, source_order)

# Your extractor creates this list each iteration:
frames = [f1_c1, f1_c2, f1_c3, ...]

bridge.submit(
    frames,
    frame_indexes=[frame_index_c1, frame_index_c2, ...],
    source_times_seconds=[time_c1, time_c2, ...],
)
```

For an extractor that physically stops disabled cameras and returns only active sources:

```python
active_source_ids = bridge.enabled_source_ids()
frames, returned_source_ids = extractor.extract(active_source_ids)

bridge.submit(
    frames,
    source_ids=returned_source_ids,
    frame_indexes=frame_indexes,
)
```

The `source_ids` sequence must match the `frames` sequence exactly.

A complete integration skeleton is in:

```text
examples/main_project_integration.py
```

## Built-in video camera ingestion

At application startup, enabled camera rows containing local video paths or `rtsp://`/`rtsps://` URIs are opened automatically. An RTSP URI is recognized directly, so `metadata` may be empty. Local files loop at end-of-stream; RTSP readers reconnect after a failure. Both are sampled at 5 FPS by default. The ingestor follows live registry changes: disabling or deleting a camera closes its reader, creating or enabling it opens the source, changing its URI replaces the pipeline, and changing assigned tasks affects the next submitted frame.

Two ingestion backends are available:

- `deepstream`: production Linux/NVIDIA path. `nvurisrcbin` and NVDEC create an independent pipeline for every enabled MP4 or RTSP source. One blocked camera cannot block the other sources.
- `opencv`: native development fallback, including Windows. It is retained for tests and small local-file demonstrations, not large RTSP deployments.

The controls are:

```text
VIDEO_INGESTION_ENABLED=true
VIDEO_INGEST_BACKEND=deepstream
VIDEO_INGEST_FPS=5
VIDEO_PREVIEW_FPS=25
VIDEO_LOOP=true
RTSP_TRANSPORT=tcp
RTSP_INGESTION_ENABLED=true
RTSP_OPEN_TIMEOUT_MS=20000
RTSP_READ_TIMEOUT_MS=10000
RTSP_RECONNECT_SECONDS=3
DEEPSTREAM_RTSP_LATENCY_MS=500
DEEPSTREAM_RTSP_STALL_TIMEOUT_SECONDS=30
```

With `VIDEO_INGEST_BACKEND=deepstream`, RTSP is opened by GStreamer/DeepStream and local paths are converted to file URIs automatically. Cameras with AI tasks use `VIDEO_INGEST_FPS`; play-only cameras with `tasks: []` use the higher `VIDEO_PREVIEW_FPS` without increasing inference load. TCP is the default for reliable LAN camera delivery. `RTSP_RECONNECT_SECONDS` controls the application retry delay after a failed pipeline, while `DEEPSTREAM_RTSP_STALL_TIMEOUT_SECONDS` controls how long DeepStream waits without receiving data before forcing its internal RTSP reconnection. Credentials remain in the persisted registry but are redacted from source API responses, status output, packet metadata, and connection errors. The OpenCV timeout settings apply only to the fallback backend.

Camera configuration is stored in the SQLite `cameras` table. `CAMERA_DB_PATH` defaults to `data/cameras.sqlite3`. On the first run only, when the table is empty, records are imported from `SOURCE_REGISTRY_PATH` (default `data/sources.json`). After import, SQLite is authoritative and the JSON file is not rewritten.

Each camera has `frame_width` and `frame_height` fields (both default to `640`). With the DeepStream backend, `nvvideoconvert` performs this normalization on the GPU before the frame reaches the CPU-facing app sink. Updating either value through camera CRUD cleanly rebuilds only that camera pipeline.

### Run the DeepStream service

DeepStream runs in the NVIDIA Linux container; it does not run inside the native Windows Python debugger. Build the image while internet access is available:

```powershell
docker compose build
```

After the image is built, switch to the camera LAN if required and start it without rebuilding:

```powershell
docker compose up -d --no-build
docker compose logs -f video-ai-router
```

Open `http://127.0.0.1:8000/dashboard`. No source schema changes are required: both `data/example.mp4` and `rtsp://...` values work with empty metadata. `GET /health` reports `video_ingestor.backend` as `deepstream` and exposes per-source frame, warning, and reconnect counters.

The provided Compose service now defaults `RTSP_INGESTION_ENABLED` to `true`, so enabled `rtsp://` records and local video files are opened together. To force a static-only run that never creates an RTSP pipeline, use:

```powershell
$env:RTSP_INGESTION_ENABLED="false"
docker compose up -d --no-build --force-recreate
```

To use the native fallback explicitly:

```powershell
$env:VIDEO_INGEST_BACKEND="opencv"
python run.py
```

Live ingestion counters and per-camera frame indexes are available at `GET /api/v1/router/status` and under `video_ingestor` in `GET /health`.

Local-file EOS uses a synchronized DeepStream teardown/reopen while preserving a monotonic camera frame index. This prevents the broadcast layer from rejecting frames from the second playback as stale. `VIDEO_LOOP` is passed through Compose and defaults to `true`; per-camera `loop_count` is visible in diagnostics.

## Annotated camera wall

Open the built-in frontend at:

```text
http://localhost:8000/dashboard
```

Each camera tile shows its assigned tasks. Fire, smoke, and plate bounding boxes are drawn server-side on the exact frame that produced the AI result. The dashboard receives every camera through one multiplexed binary WebSocket, avoiding the browser's six-connection HTTP/1.1 limit. Cameras assigned to both tasks are published after both results for that frame have been composed.

The dashboard polls the same source registry and broadcast-state APIs every two seconds. Changes made through Swagger at `/docs` therefore update camera names, enabled state, and task badges without a manual page refresh. The dashboard links back to Swagger, `/dashboard` is listed in Swagger, and `/` redirects to the dashboard.

Frontend broadcasting can be controlled at startup:

```text
BROADCAST_ENABLED=true
BROADCAST_JPEG_QUALITY=82
```

It can also be toggled from the dashboard or through:

```http
PUT /api/v1/broadcast/state
Content-Type: application/json

{"enabled": false}
```

Streams and snapshots are available at:

```text
WS  /api/v1/broadcast/ws
GET /api/v1/broadcast/streams/camera-01.mjpg
GET /api/v1/broadcast/snapshots/camera-01.jpg
```

The WebSocket is the dashboard transport for all cameras. Individual MJPEG endpoints remain available for external clients that need a single camera.

Positive detections print one-line `[DETECTION]` records to standard output. Empty frames are not logged.

## Persistent plate detection logs

Every recognized non-empty plate is automatically written to the SQLite table `plate_logs`. The table has exactly three fields:

| Field    | Meaning                                         |
| -------- | ----------------------------------------------- |
| `camera` | Source/camera ID that detected the plate        |
| `time`   | UTC detection timestamp from `processed_at_utc` |
| `plate`  | Recognized plate number                         |

The database defaults to `data/plate_logs.sqlite3` and can be changed with `PLATE_LOG_DB_PATH`.

External producers can add a record through:

```http
POST /api/v1/plate-logs
Content-Type: application/json

{
  "camera_id": "camera-01",
  "time": "2026-07-15T08:15:30+03:30",
  "plate": "23ن92917"
}
```

Stored records can be read with `GET /api/v1/plate-logs`. Optional query parameters are `camera_id`, `plate`, and `limit`.

## Persistent fire/smoke events

Confirmed incidents are written to `fire_smoke_logs` in the same SQLite database as plate logs. A medium incident creates one row; a later high transition updates that same incident row and snapshot instead of creating another alert. The ten-second incident-end grace prevents one-second detection dips from producing repeated rows. Each record includes the camera, UTC time, severity, fire/smoke counts and confidence, rolling-window length, and a snapshot URL. Snapshot drawing, JPEG encoding, and database writes use a bounded background queue and do not block inference. Files are served below `/media/fire_smoke_snapshots/` and stored under `SAVED_MEDIA_PATH`.

The default policy treats fewer than 5 positive processed frames in 3 seconds as a false positive, 5 as low, 10 as medium, and 20 as high. Admin changes take effect online without restarting DeepStream:

```http
GET /api/v1/fire-smoke/settings
PUT /api/v1/fire-smoke/settings
Content-Type: application/json

{
  "window_seconds": 3,
  "low_count": 5,
  "medium_count": 10,
  "high_count": 20
}
```

Read events with `GET /api/v1/fire-smoke-logs`; optional filters are `camera_id`, `severity`, and `limit`. Counts are based on frames actually submitted to AI, so configure `VIDEO_INGEST_FPS` high enough for the selected window and high threshold.

Detection confidence is filtered both in the model call and again when model output is parsed. The defaults are:

```text
FIRE_CONFIDENCE=0.30
SMOKE_CONFIDENCE=0.30
PLATE_CONFIDENCE=0.30
PLATE_OCR_CONFIDENCE=0.50
```

Plate recognition uses a GPU-batched cascade:

```text
full frames -> YOLO vehicle gate -> vehicle crops -> plate detector -> plate crops -> OCR
```

The vehicle gate uses `weights/vehicle_detector/yolo11n.pt` and COCO classes car, motorcycle, bus, and truck. Frames without an accepted vehicle never run the plate model. Plate boxes are converted back to original-frame coordinates before drawing, logging, and WebSocket delivery. Relevant controls are:

```text
VEHICLE_DETECTOR_WEIGHTS=weights/vehicle_detector/yolo11n.pt
VEHICLE_CONFIDENCE=0.35
VEHICLE_IOU=0.45
VEHICLE_IMGSZ=640
VEHICLE_MAX_PER_FRAME=12
VEHICLE_CLASS_IDS=2,3,5,7
VEHICLE_CROP_PADDING_RATIO=0.05
PLATE_CROP_BATCH_SIZE=16
MIN_VEHICLE_WIDTH_PIXELS=120
MIN_VEHICLE_HEIGHT_PIXELS=80
MIN_VEHICLE_AREA_RATIO=0.025
```

Vehicles below any configured minimum width, height, or frame-area ratio are treated as too far from the camera and rejected before plate inference. Vehicle crops are grouped into bounded GPU batches instead of invoking the plate detector once per car.

OCR output is accepted only when it has the complete Iranian layout: two digits, one Persian letter, then five digits. This is exactly seven digits and one Persian character. Partial reads and Latin-letter reads are discarded and are not logged or broadcast as recognized plates.

General plate settings are seeded from the environment into SQLite on the first run. After that, they can be changed online from Swagger and become effective on the next model batch without restarting DeepStream or reloading the models. Each camera may override any subset; fields without an override always inherit the current general value:

- `GET` / `PUT /api/v1/plate-settings/general`
- `GET` / `PATCH` / `DELETE /api/v1/plate-settings/cameras/{camera_id}`

For a camera `PATCH`, omitted fields are unchanged and a field sent as `null` has its override removed. `DELETE` removes every override for that camera.

## Swagger maintenance and model testing

Open `/docs`. The API is ordered into diagnostics, camera CRUD, frame/model testing, fire/smoke, plate settings, plate logs, results, broadcast, and compatibility sections.

- `GET /api/v1/diagnostics/overview` shows DeepStream, model scores, workers, logs, and broadcast state.
- `GET /api/v1/diagnostics/checks` runs safe section-by-section maintenance checks.
- `POST /api/v1/diagnostics/cameras/{camera_id}/restart` cleanly restarts one DeepStream source.
- `POST /api/v1/frame-rounds/jpeg` accepts test JPEG/PNG frames and routes them through each task assigned to the selected camera ID.
- `GET /api/v1/results/recent` returns the resulting fire/smoke or plate model output.

## Camera and routing API

The camera API provides full online CRUD. Changes are persisted before the API returns and are also published as secret-free `camera_changed` JSON messages over `/api/v1/broadcast/ws`. Binary messages on the same socket remain annotated JPEG frames.

An enabled camera may use any of four processing modes. Use `PUT /api/v1/cameras/{camera_id}/tasks` in Swagger with one of these arrays:

```json
[]
["fire_smoke"]
["plate_recognition"]
["fire_smoke", "plate_recognition"]
```

An empty array is play-only mode. DeepStream continues decoding and the dashboard/WebSocket continues receiving frames, but the router submits no fire/smoke, vehicle, plate, or OCR work for that camera.

### Create a camera

```http
POST /api/v1/cameras
Content-Type: application/json

{
  "camera_id": "camera-09",
  "name": "Gate camera",
  "enabled": true,
  "tasks": ["fire_smoke", "plate_recognition"],
  "source_uri": "rtsp://user:password@camera-host/live",
  "frame_width": 640,
  "frame_height": 640,
  "metadata": {"area": "gate"}
}
```

### Read, update, replace, or delete cameras

```http
GET    /api/v1/cameras
GET    /api/v1/cameras/camera-09
PATCH  /api/v1/cameras/camera-09
PUT    /api/v1/cameras/camera-09
DELETE /api/v1/cameras/camera-09
POST   /api/v1/cameras/camera-09/enable
POST   /api/v1/cameras/camera-09/disable
```

RTSP credentials are stored for ingestion but redacted from REST responses and WebSocket events.

## Backward-compatible source API

`/api/v1/sources` operates on the same SQLite camera rows and remains available to existing integrations.

### List sources

```http
GET /api/v1/sources
```

### Enable or disable a source

```http
POST /api/v1/sources/camera-03/enable
POST /api/v1/sources/camera-03/disable
```

The main extractor should regularly synchronize its active readers with:

```python
runtime.registry.enabled_source_ids()
```

### Change tasks for several sources

```http
PUT /api/v1/sources/bulk/task-assignment
Content-Type: application/json

{
  "source_ids": ["camera-07", "camera-08", "camera-09"],
  "tasks": ["fire_smoke", "plate_recognition"],
  "enabled": true
}
```

### Recent results

```http
GET /api/v1/results/recent?source_id=camera-07&task=plate_recognition&limit=100
```

### Live result stream

```text
ws://HOST:8000/api/v1/results/ws
```

### Router and batch statistics

```http
GET /api/v1/router/status
```

Statistics include pending sources, replaced stale frames, processed batches, processed frames, failures, last batch size, and last inference duration.

## HTTP frame endpoint

For separate-process testing, JPEG or PNG frames can be posted as multipart data:

```http
POST /api/v1/frame-rounds/jpeg
```

Fields:

- `files`: one file per source.
- `source_ids_json`: JSON list in file order.
- `round_sequence`: extraction-round number.
- `frame_indexes_json`: optional JSON list.
- `source_times_json`: optional JSON list.

This endpoint adds encoding, copying, network, and decoding overhead. It is not the recommended path for the main real-time pipeline.

## Model setup

The uploaded archives did not contain the actual model files. Copy them to these locations or change `.env`:

```text
weights/fire_smoke/model.engine
weights/vehicle_detector/yolo11n.pt
weights/plate_detector/model.pt
weights/plate_recognizer/model.pt
weights/plate_recognizer/model_config.yaml
weights/plate_recognizer/preprocessor/image_processor_config.yaml
```

The fire/smoke processor accepts TensorRT `.engine`, ONNX `.onnx`, or Ultralytics `.pt` models. A fixed-batch TensorRT engine is padded according to `FIRE_SMOKE_ENGINE_FIXED_BATCH`.

## Model catalog, conversion, and selection

Swagger contains a `model-management` section:

- `GET /api/v1/models/artifacts` lists `.pt`, `.engine`, and `.onnx` files. Filter by `role` or `format=pt`.
- `GET /api/v1/models/artifacts/content?path=...` downloads a catalog/export artifact by its returned URL.
- `POST /api/v1/models/conversions` queues a background PT-to-TensorRT export and, by default, also creates an ONNX fallback.
- `POST /api/v1/models/engine-exports` uploads a `.pt` directly from Swagger and queues the same background export.
- `GET /api/v1/models/conversions/{job_id}` reports progress, output artifacts, and per-format errors.
- `GET/PATCH /api/v1/models/settings` selects the fire/smoke, vehicle, and plate models.

For direct Swagger upload, open `POST /api/v1/models/engine-exports`, choose the
`.pt` file and its `role`, then execute. The response is `202 Accepted`; poll its
`status_url`. With the default `select_when_ready=true`, the completed `.engine`
(or successful ONNX fallback) becomes that role's persisted selection. Its
downloadable `selected_url` is visible under
`GET /api/v1/settings/general -> models -> resolved_models -> <role>`.

Model files are organized by role below `MODEL_ROOT_PATH`:

```text
weights/fire_smoke/
weights/vehicle_detector/
weights/plate_detector/
```

Names containing `nano`, `tiny`, `small`, `medium`, or `large` are identified in the catalog. YOLO suffixes such as `yolo11n`, `yolo11s`, `yolo11m`, and `yolo11l` are mapped to the corresponding variants. Only installed choices are returned.

Example conversion request:

```json
{
  "source_model": "vehicle_detector/yolo11n.pt",
  "output_directory": "vehicle_detector/exports",
  "imgsz": 640,
  "batch": 16,
  "workspace_gb": 4,
  "half": true,
  "dynamic": true,
  "device": "0",
  "create_onnx_fallback": true,
  "select_when_ready": true,
  "overwrite": false,
  "timeout_seconds": 300
}
```

Conversion runs in a single background queue and does not block FastAPI or DeepStream ingestion threads. Dynamic TensorRT export is the default because vehicle and plate crop batches vary at runtime.

Runtime resolution defaults to:

```text
TensorRT .engine -> ONNX .onnx -> PyTorch .pt
```

Fallback requires the files to have the same directory and stem, such as `fire_small.engine`, `fire_small.onnx`, and `fire_small.pt`. When a custom output directory is requested, the conversion job copies the source PT into that directory to keep the model family complete. Runtime inference errors also advance to the next available format. Model selection changes are persisted and applied at the next inference batch without restarting the camera pipelines.

Selecting a concrete `.engine` or `.onnx` path is an exact per-role format choice,
so different roles may use different formats at the same time. Selecting a `.pt`
path keeps the legacy `MODEL_PREFERRED_FORMAT` family resolution shown above.
Conversion job history is stored in the same SQLite database as model settings;
an export interrupted by application restart is retained and marked failed.

The `general-settings` Swagger section provides `GET/PATCH /api/v1/settings/general`. It combines model selection/export settings, plate thresholds, fire/smoke incident policy, all startup environment configuration, and the camera processing modes in one response.

Relevant startup defaults are:

```text
MODEL_ROOT_PATH=weights
MODEL_PREFERRED_FORMAT=engine
MODEL_ALLOW_ONNX_FALLBACK=true
MODEL_ALLOW_PT_FALLBACK=true
MODEL_EXPORT_IMGSZ=640
MODEL_EXPORT_BATCH_SIZE=16
MODEL_EXPORT_WORKSPACE_GB=4
MODEL_EXPORT_HALF=true
MODEL_EXPORT_DYNAMIC=true
MODEL_EXPORT_TIMEOUT_SECONDS=300
```

TensorRT engines are GPU/driver/TensorRT-specific. Build them in the deployment container on the target GPU. The DeepStream image installs `onnx`, `onnxruntime-gpu`, and `onnxslim` so ONNX export and inference do not depend on Ultralytics attempting a package installation at runtime.

The plate processor performs:

1. Batched YOLO vehicle detection across source frames.
2. Early rejection of small/distant vehicles using each camera's effective settings.
3. Batched YOLO plate detection inside the remaining vehicle crops.
4. Batched Hezar OCR when supported by the recognizer.
5. Exact Iranian-format and OCR-score validation.
6. Original-frame coordinate restoration and per-frame plate deduplication.

## Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python run.py
```

Windows PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
python run.py
```

API documentation:

```text
http://localhost:8000/docs
```

## Mock mode

To validate routing without GPU or weights:

```bash
PROCESSOR_MODE=mock python run.py
```

Windows PowerShell:

```powershell
$env:PROCESSOR_MODE="mock"
python run.py
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
python scripts/smoke_test.py
```

Expected smoke-test output:

```text
SMOKE TEST PASSED
{'received_frames': 8, 'accepted_sources': 8, 'task_submissions': 10}
```

## Scaling notes for 50+ streams

- Do not wait indefinitely for one frame from every configured camera. The previous fire project used all-or-nothing synchronization, which lets one failed stream stop the entire round.
- Keep capture queues shallow. The router already keeps one pending frame per source per task.
- Use a fixed source ID with every frame. Never depend only on list position after cameras can be enabled or disabled.
- Separate task workers prevent a slow OCR stage from blocking fire/smoke inference.
- Use TensorRT engines whose maximum or fixed batch matches the configured worker batch.
- For multiple GPUs, create one worker set per GPU and assign sources to a GPU shard before task routing.
- Measure end-to-end latency, stale replacements, inference time, and dropped capture frames. Throughput alone is not sufficient for real-time CCTV.

## Important boundary

This project owns local video-file and RTSP reading, routing, and inference. Other live-source types can still be supplied through an external extractor that obeys `enabled_source_ids()`.

```
docker run --rm -it `
  --name merged-video-ai-router `
  --gpus all `
  --entrypoint /bin/bash `
  -p 8000:8000 `
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video `
  -e VIDEO_INGESTION_ENABLED=true `
  -e PROCESSOR_MODE=real `
  -v "${PWD}\:/workspace/" `
  merged-video-ai-router:v2
```

```
docker compose build video-ai-router
docker compose up -d --force-recreate video-ai-router
```

```
docker build `                                                                                          --add-host=host.docker.internal:host-gateway                                                                                         --no-cache `
  -f Dockerfile.deepstream `
  -t merged-video-ai-router:v2 `
  .
```
