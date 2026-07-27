# Verified Integration Map

Status: initialized from repository inspection.

## Important Flows

### 1. Frame Pipeline: Video Source → Inference → Output

```
Input                   Transformation                      Consumers                   Downstream Output
──────────────────────────────────────────────────────────────────────────────────────────────────────────

RTSP/MP4 URI            DeepStreamIngestor /                TaskRouter                  ResultStore
                        VideoFileIngestor                    │                          Broadcast (WS + MJPEG)
                        │                                   ├─ LatestBuffer (per task)  PlateLogStore
                        │  Decode @ native/sources.fps      ├─ TaskWorker (micro-batch) FireSmokeLogStore
                        │  Resize to frame_width/height     └─ BatchProcessor           HumanLogStore
                        │  Produce FramePacket                                              │
                        ▼                                   ▼                           ▼
                    FramePacket                          TaskResult                   HTTP API / WS
                    {source_id, frame,                    {task, source_id,           GET /api/v1/results/recent
                     round_sequence,                       data, error}               WS /api/v1/results/ws
                     captured_at_utc}                                                   GET /api/v1/plate-logs
                                                                                        GET /api/v1/fire-smoke-logs
                                                                                        GET /api/v1/humans/logs
```

**Failure modes:**
- Decode failure: ingestor logs warning, reconnects (RTSP) or loops (file)
- Missing source: `frames_unregistered` counter, frame dropped
- Disabled source: `frames_disabled` counter, frame dropped
- Worker queue full: `task_submission_rejections` counter, frame dropped
- Inference error: `TaskResult.error` populated, logged, not stored

### 2. Camera CRUD → Live Pipeline

```
API Endpoint            Transformation                  Runtime Effect                Evidence Path
─────────────────────────────────────────────────────────────────────────────────────────────

POST /api/v1/cameras    SourceRegistry.insert()         Ingestion pipeline starts    GET /api/v1/cameras
PATCH /api/v1/cameras/  SourceRegistry.update()         Pipeline rebuilt if URI     GET /api/v1/cameras/{id}
  {camera_id}                                           or dimensions change          Router stats
DELETE /api/v1/cameras/ SourceRegistry.remove()         Pipeline closed              GET /api/v1/cameras
  {camera_id}
POST /.../{id}/enable   SourceRecord.enabled=True       Frame routing resumes        Health endpoint
POST /.../{id}/disable  SourceRecord.enabled=False      Frame routing stops          Dashboard tile
PUT /.../{id}/tasks     SourceRecord.tasks[]            Task routing changes         Router per-task stats
```

**Listeners notified on each change:** `AnnotatedBroadcastHub`, `PlateSettingsStore`, ingestion pipeline manager.

### 3. Model Selection

```
PATCH /api/v1/models/settings     ModelManager.update()          Resolved next batch     GET /api/v1/models/settings
                                    │                             │                     GET /api/v1/settings/general
                                    ▼                             ▼
                                SQLite persist              Inference processor
                                                            (model_provider callback)
```

**Selection rules:**
- Exact `.engine`/`.onnx` path → use that format
- `.pt` path → use `MODEL_PREFERRED_FORMAT` family with fallback chain
- Fallback chain: `.engine` → `.onnx` → `.pt`
- Format requires same directory and stem

### 4. Model Conversion

```
POST /api/v1/models/conversions   ModelConversionManager.submit()    Background thread
                                    │                                   │
                                    ▼                                   ▼
                                Queue job → PT export to ONNX →     Poll GET /api/v1/models/
                                  ONNX to TensorRT engine             conversions/{job_id}
                                  (optional select_when_ready)        File appears in weights/
```

### 5. Fire/Smoke Event Pipeline

```
FireSmokeProcessor           FireSmokeLogStore.observe_result()    GET /api/v1/fire-smoke-logs
  │                            │
  ▼                            ▼
TaskResult{data:             Rolling window severity              API response
  fires, smokes}              evaluation                          Severity snapshot on dashboard
  │                           Incident persistence
  ▼                           Bounded snapshot/media write
Broadcast (real-time)         queue
```

**Severity policy:** `window_seconds` rolling count; `low_count` < `medium_count` < `high_count`. Default 3s window, counts 5/10/20. Incident end grace: 10 seconds.

### 6. Plate Recognition Pipeline

```
Full frames → YOLO vehicle detection → early rejection (size/area) →
  vehicle crops → YOLO plate detection → plate crops → Hezar OCR →
    Iranian format validation (##[letter]#####) → coordinate restoration →
      plate deduplication → PlateLogStore.insert_result() + Broadcast
```

**Vehicle gate:** COCO classes 2 (car), 3 (motorcycle), 5 (bus), 7 (truck). Min width/height/area ratio enforced.

### 7. Face Recognition Pipeline

```
Full frames → YOLO-pose human detection → ByteTrack tracking →
  face detection (YOLOv8n-face) → quality gate (landmarks, blur, pose) →
    ArcFace embedding → Qdrant vector search → recognition decision →
      HumanLogStore.observe_result() + Broadcast
```

**Quality gate:** threshold, blur, min face size, min eye distance, max yaw/pitch/roll, landmark requirement.

## Interface Contracts

### FramePacket → TaskWorker

```python
@dataclass(frozen=True, slots=True)
class FramePacket:
    source_id: str
    frame: np.ndarray          # RGB, normalized to frame_width×frame_height
    round_sequence: int
    frame_index: int
    captured_monotonic: float
    captured_at_utc: str
    source_time_seconds: float | None
    metadata: Mapping[str, Any]
```

### TaskWorker → BatchProcessor

```python
# process() receives list[np.ndarray], returns list[dict[str, Any]]
# One dict per frame in the batch, same order
```

### TaskResult → Consumers

```python
@dataclass(slots=True)
class TaskResult:
    task: TaskName
    source_id: str
    round_sequence: int
    frame_index: int
    captured_at_utc: str
    processed_at_utc: str
    processing_ms: float
    data: dict[str, Any]       # Processor-specific output
    error: str | None
```

## Section Relationships

| Section | Depends on | Used by |
|---|---|---|
| Video ingestion | SourceRegistry, Camera DB | TaskRouter |
| Task routing | LatestBuffer, SourceRegistry | All processors |
| Fire/smoke processing | ModelManager, FireSmokeLogStore | Broadcast, API |
| Plate processing | ModelManager, PlateSettingsStore, PlateLogStore | Broadcast, API |
| Face processing | ModelManager, FaceQualityStore, HumanLogStore, Qdrant | Broadcast, API |
| Dashboard | AnnotatedBroadcastHub, SourceRegistry | User UI |
| Model management | Filesystem (weights/), SQLite | All processors |
| Configuration | Settings, env vars, SQLite | All sections |
