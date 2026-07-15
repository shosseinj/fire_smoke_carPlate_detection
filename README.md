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

| Source | Video | Tasks |
|---|---|---|
| `camera-01` | `smoke1.mp4` | `fire_smoke` |
| `camera-02` | `smoke2.mp4` | `fire_smoke` |
| `camera-03` | `sdf.mp4` | `fire_smoke` |
| `camera-04` | `fg.mp4` | `plate_recognition` |
| `camera-05` | `yt.mp4` | `plate_recognition` |
| `camera-06` | `bucket11.mp4` | `plate_recognition` |
| `camera-07` | `etry.mp4` | `fire_smoke`, `plate_recognition` |
| `camera-08` | `test1.mp4` | `fire_smoke`, `plate_recognition` |

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

At application startup, enabled records with `metadata.kind` set to `video_file` are opened automatically. Files loop at end-of-stream and are sampled at 5 FPS by default. The ingestor follows live registry changes: disabling a source closes its reader, enabling it reopens the file, and changing its assigned tasks affects the next submitted frame.

The controls are:

```text
VIDEO_INGESTION_ENABLED=true
VIDEO_INGEST_FPS=5
VIDEO_LOOP=true
```

Live ingestion counters and per-camera frame indexes are available at `GET /api/v1/router/status` and under `video_ingestor` in `GET /health`.

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

| Field | Meaning |
|---|---|
| `camera` | Source/camera ID that detected the plate |
| `time` | UTC detection timestamp from `processed_at_utc` |
| `plate` | Recognized plate number |

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

## Source and routing API

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
weights/plate_detector/model.pt
weights/plate_recognizer/model.pt
weights/plate_recognizer/model_config.yaml
weights/plate_recognizer/preprocessor/image_processor_config.yaml
```

The fire/smoke processor accepts a TensorRT engine or an Ultralytics `.pt` model. The default assumes a fixed batch-8 TensorRT engine and pads incomplete batches before inference.

The plate processor performs:

1. Batched YOLO plate detection across source frames.
2. Collection of all plate crops from the batch.
3. Batched Hezar OCR when supported by the recognizer.
4. Automatic per-crop fallback when the recognizer does not accept a list.
5. Per-frame plate deduplication.

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

This project now owns local video-file reading, routing, and inference for the configured demo. An external extractor is still the integration boundary for RTSP/live readers; it should obey `enabled_source_ids()` when those sources are added.
