# AGENTS.md — Unified Video AI Task Router

This file is the shared operating contract for Codex, OpenCode, and any sub-agents working in this repository.

## Project identity

- Name: Unified Video AI Task Router
- Version: 2.0.0
- Stack: FastAPI, PyTorch, TensorRT, ONNX Runtime, DeepStream, OpenCV
- Python: 3.10+
- Production runtime: NVIDIA GPU inside the DeepStream Docker image
- Main entrypoint: `python3 run.py`
- API documentation: `http://127.0.0.1:9999/docs`
- Dashboard: `http://127.0.0.1:9999/dashboard`
- Health and runtime evidence: `http://127.0.0.1:9999/health`

Authentication role management uses exactly three canonical roles: `superadmin > admin >
user`. Legacy `superuser` is normalized to `superadmin`, while legacy `operator` and
`viewer` values are normalized to `user`. The configured
default account is seeded as `superadmin`; `POST /api/v1/auth/create-user` permits
superadmin to create superadmin/admin/user and admin to create admin/user. Persian
role guidance is available at `GET /api/v1/auth/roles`, and authentication or
authorization denials return Persian details.

## Product priorities

Use this priority order unless the user explicitly overrides it for a task:

1. Correctness, safety, and stable service operation.
2. Bounded end-to-end latency and real-time behavior.
3. Sustainable throughput and fair camera scheduling.
4. Processing every historical frame.

Real-time behavior is more important than accumulating stale work. Under overload, prefer the newest useful frame per source, bounded queues, fair scheduling, and visible drop/backpressure metrics. Never allow an unbounded queue. If a task explicitly requires lossless processing, apply backpressure and explain that latency may grow and RTSP sources may still lose frames upstream when processing capacity is insufficient.

For performance-sensitive changes, report evidence rather than assumptions:

- Input FPS and processed FPS per task.
- Batch size and batch latency.
- Queue depth, capacity, frame age, blocked submissions, replacements, and drops.
- Active model artifact and runtime format: TensorRT, ONNX, or PT.
- GPU memory and utilization when available.
- `/health` before and after the change.

For every low-FPS report, run the bounded sampler before changing FPS, batching, model, or queue settings:

```bash
curl -fsS "http://127.0.0.1:9999/api/v1/diagnostics/fps?sample_seconds=5&expected_fps=25"
```

Do not calculate FPS from a single cumulative counter snapshot. Use the sampler to distinguish a configured `VIDEO_INGEST_FPS` ceiling from source/decode starvation, router submission loss, task-worker replacements/backpressure, processor failures, and broadcast bandwidth. Increase `VIDEO_INGEST_FPS` gradually only after checking per-camera submitted FPS, per-task processed FPS, replacements, pending sources, batch latency, frame age, and GPU utilization. Lower dashboard resolution can reduce network and browser decode/render cost, but it does not increase decode or model-inference capacity.

The normal detector input is fixed at `3×640×640`. TensorRT detector engines should use a dynamic batch dimension only, normally batch 1–8. Use `--dynamic-batch-only`; do not make height and width dynamic unless the task explicitly requires it.

The DeepStream ingest path converts frames with `nvvideoconvert` to BGRx and normalizes
the mapped appsink buffer to the existing 3-channel NumPy processor contract. It does
not insert a CPU `videoconvert` stage. The Compose AI-ingest default is 25 FPS, while
`VIDEO_INGEST_FPS` remains the authoritative override; use the bounded FPS diagnostic
sampler before raising it further.

When OpenCV exposes CUDA resize, DeepStream uses it for the inference view and falls
back to CPU resize without changing the NumPy frame contract. The native source frame
is retained for face evidence and downstream media.

The broadcast hub retains the most recent face/human result for a short configurable
TTL (`BROADCAST_FACE_OVERLAY_TTL_MS`, default 250 ms) so slower face inference does not
make boxes flicker off every intermediate real-time frame. This is display smoothing,
not a claim that inference ran on every displayed frame.

Fire candidate confidence and fire/smoke severity confidence floors are separate
settings. Keep severity floors above detector admission when low-confidence false
positives must not accumulate into incidents. Plate persistence is queued on a bounded
background writer so database and snapshot I/O do not block plate inference.

Face recognition quality is emitted per face (`quality_score`, metrics, validity),
while each tracked human accumulates `best_face_quality`; human history persistence
must bind PostgreSQL boolean CASE parameters as booleans, not integer `0/1` values.

When ByteTrack expires a source track, face recognition emits one additive
`disappeared_humans` entry with the last stable identity. The runtime finalizes that
track in `human_logs` and creates an idempotent `detection_logs` bridge using a
`human-track:<session>:<camera>:<track>` source event key, making the record visible
through `/api/v1/logs/filter` without changing the visible `humans` broadcast list.
When Qdrant's `FaceMatch.person` is a national code, `HumanLogStore` resolves it
against `personnel.national_code` and stores the personnel `fname lname` plus
`personnel_id`; unknown or legacy non-code identities retain their original value.

Task workers support `TASK_QUEUE_POLICY=latest_per_source` for bounded-latency CCTV or
`TASK_QUEUE_POLICY=lossless_fifo` for bounded FIFO admission with backpressure. The
real-time Compose default is latest-per-source; FIFO is an explicit file/API workload
option. FIFO
prevents worker-side replacement only while its finite capacity and timeout can absorb
the workload; `TASK_QUEUE_BLOCK_TIMEOUT_MS=0` waits instead of rejecting admission.
It still cannot guarantee lossless RTSP delivery when decode, network, or GPU
throughput is lower than the source rate.


## Architecture

- `app/main.py`: FastAPI application and lifespan.
- `app/runtime.py`: component construction, startup, shutdown, and model selection logs.
- `app/core/init_db.py`: idempotent database seeding on startup — creates seed building/section/room, default shift, default personnel from `DEFAULT_PERSONNEL_SEED_DATA`, and sample detection logs. Called from `build_runtime()`.
- `app/core/router.py`: routes frames to task-specific workers.
- `app/core/worker.py`: micro-batched task execution.
- `app/core/latest_buffer.py`: task queue policy and queue telemetry.
- `app/core/deepstream_ingestor.py`: production GPU decode and frame delivery.
- `app/core/video_ingestor.py`: OpenCV development fallback.
- `app/core/model_management.py`: persistent model catalog, selection, conversion jobs, and fallback resolution.
- `app/processors/fire_smoke.py`: fire/smoke detection, tracking, severity, and incidents.
- `app/processors/plate.py`: vehicle detection, plate detection, and Iranian plate OCR.
- `app/processors/face_recognition.py`: human detection, tracking, face detection, quality checks, ArcFace, and vector search.

The three independent inference workers are:

- `fire_smoke`
- `plate_recognition`
- `face_recognition`

Model fallback priority is `.engine → .onnx → .pt` when fallbacks are enabled. Persistent selections in the model database override configuration defaults. Confirm the active artifact through `/api/v1/models/settings` and runtime logs instead of trusting filenames alone.

## Dashboard and broadcast system

The dashboard at `/dashboard` (`app/web/dashboard.html`) has two display modes:

### 1. JPEG fallback mode (default, always active)
- Connects to the broadcast WebSocket (`/api/v1/broadcast/ws`) without `metadata_only`
- Receives binary frames: 4-byte header length (uint32 BE) + JSON header + JPEG bytes
- Displays server-annotated JPEG frames on `<img class="jpeg-fallback">` elements using `URL.createObjectURL()`
- Works without WebRTC/MediaMTX/GStreamer; the `AnnotatedBroadcastHub` renders annotations server-side
- Variable: `useJpegFallback = true` in `synchronizeDashboard()`

### 2. WebRTC preview mode (optional enhancement)
- Uses WHIP/WHEP protocol via MediaMTX server (port 8789) for real-time H264 video
- Requires `MEDIA_PREVIEW_ENABLED=true`, GStreamer, and MediaMTX
- The `MediaPreviewPublisher` remuxes H264 from sources to RTSP and publishes to MediaMTX
- The MP4 file source pipeline has known EOS issues - file-based previews may fail repeatedly

### Broadcast hub (`app/core/broadcast.py`)
- `AnnotatedBroadcastHub` renders frames asynchronously in a background thread
- Produces full-res and wall-res JPEG per source
- `publish_result()` accepts `FramePacket` + `TaskResult` and queues annotation + encoding
- `publish_passthrough()` broadcasts frames without AI overlay
- Binary WS format: `struct.pack("!I", len(header)) + header.encode() + jpeg_bytes`
- The WebSocket handler (`/api/v1/broadcast/ws`) supports `metadata_only` and `fullscreen_source` query params
- MJPEG fallback at `/api/v1/broadcast/streams/{source_id}.mjpg`
- Snapshot at `/api/v1/broadcast/snapshots/{source_id}.jpg`

### Broadcast health checklist
- Broadcast enabled and rendering: check `GET /api/v1/broadcast/state`
- Dashboard loads: check `GET /dashboard`
- Binary WS delivers JPEG frames: connect to `ws://host/api/v1/broadcast/ws` without params
- Results WS delivers overlays: connect to `ws://host/api/v1/results/ws`

The recent-detections WebSocket payload matches the legacy contract exactly. Each
detection contains only `id`, `area`, `person`, `full_name`, `confidence`,
`detection_time`, `face_image_base64`, `access_granted`, `counts_for_attendance`,
and `classification`. Entries without an encodable face image are omitted; body
snapshot fallback fields are not sent.

## Polygon zone system

### Overview

The polygon zone system detects when a tracked human enters or exits a defined polygonal region (zone) in a camera's field of view. Zones are stored as `polygon_json` on the `rooms` table, where each room can have an associated polygon defining its boundary in image pixel coordinates (e.g., `[[x1,y1],[x2,y2],...]` with at least 3 points for a valid polygon).

### Zone hierarchy

    Building → Section → Room (with polygon_json)

Cameras are assigned directly to a room through nullable `cameras.room_id`, a foreign key to `rooms.id` with `ON DELETE SET NULL`. A camera's section is derived through its assigned room.

### Default polygon fallback

When a camera's section has **no rooms with custom polygons**, a default full-frame polygon
`[[0,0],[0,640],[640,640],[640,0]]` is used automatically. This ensures zone matching
works for every camera without requiring polygon configuration. The default polygon is
tracked in-memory only (`get_default_polygon_entry_state()` on `LocationStore`), with
no database inserts.

### Human log gating on polygon transitions

For FACE_RECOGNITION results, the human log (`human_logs.observe_result`) is saved **only when
a polygon zone transition occurs** (entered or exited). If no transition occurs (e.g., a person
stays outside the zone), the human log is discarded. This is implemented by a combined
`face_polygon_observer` in `runtime.py` (`_build_face_polygon_observer()`) that replaces
the separate result_observer + location_observer for the face_recognition worker.

| Condition | Human log saved? |
|-----------|-----------------|
| Person outside polygon (no custom zones, default applies) | No (no transition) |
| Person enters polygon | Yes (`entered`) |
| Person exits polygon | Yes (`exited`) |
| Person stays inside polygon | No (heartbeat, no transition) |
| No polygon zones exist for camera's section | Default full-frame polygon is used |

### Entry/exit detection

- **Human foot point**: `((x1 + x2) / 2, y2)` computed from the human bounding box `[x1, y1, x2, y2]`. This is the center-bottom point, representing where the person's feet are.
- **Point-in-polygon**: The PNPoly ray-casting algorithm (`point_in_polygon()` in `location_store.py`) checks containment.
- **Transition tracking**: The `LocationStore` maintains an in-memory `_entry_state` dictionary keyed by `(camera_id, track_id, room_id)`. When a foot point transitions from outside → inside, a match record with `transition_type='entered'` is created. When inside → outside, `transition_type='exited'` is created.
- **Heartbeat**: When a track remains inside the same zone, matches are still recorded but with `transition_type=None` (heartbeat).

### Pipeline

1. `FaceRecognitionProcessor.process_batch()` outputs humans with bbox and track_id.
2. `TaskWorker` calls the combined `face_polygon_observer(packet, result)` for each result.
3. `_build_face_polygon_observer()` in `runtime.py` resolves the camera's `room_id` and checks that room's polygon.
4. If the assigned room has a custom polygon → calls `LocationStore.match_detection_to_room()` with the human foot point.
5. If no custom polygons → uses default full-frame polygon via `LocationStore.get_default_polygon_entry_state()`.
6. If a transition (entered/exited) occurred → calls `HumanLogStore.observe_result()` to save the human log.
7. No transition → human log is **not saved**.

Face evidence is retained independently of this polygon admission gate: the face
observer persists detected-face media for successful face results without creating
a human log, so a later ByteTrack expiry can reuse that evidence when the face is
no longer visible. Expiry finalization also upgrades an existing idempotent
`detection_logs` bridge when its `face_image` was previously empty.

### API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/rooms/` | GET | List rooms with polygon_points |
| `/rooms/{id}` | GET | Get room with polygon_points |
| `/rooms/` | POST | Create room with polygon_points |
| `/rooms/{id}` | PUT | Update room polygon_points |
| `/rooms/{id}/matches` | GET | List all polygon matches for a room |
| `/rooms/{id}/entry-exits` | GET | List only entry/exit transitions (filters by transition_type) |

The `polygon_points` field in room create/update accepts `[[x,y], [x,y], ...]` with at least 3 points.

### Polygon drawing on dashboard frames

Zone polygons are drawn on the annotated broadcast JPEG frames when `draw_zones` is enabled:

- **`draw_zones`** flag: stored in `general_settings.draw_zones` (default `True`). Controls whether zone polygons are visually overlaid on the dashboard broadcast.
- **Polygon rendering**: `AnnotatedBroadcastHub._draw_zones()` in `app/core/broadcast.py` draws semi-transparent filled polygons (25% opacity) with thick borders on each zone.
- **Color palette**: 6 rotating colors (red, green, blue, yellow, magenta, cyan) assigned in order per polygon.
- **Zone data flow**: At startup and on source/settings changes, `Runtime._refresh_all_source_zones()` resolves each camera's assigned room and pushes only that room's parsed polygon to the broadcast hub.
- **Only custom zones are drawn**: Default full-frame polygon (used for fallback entry/exit tracking) is NOT drawn — only rooms with explicit `polygon_json` on the `rooms` table appear visually.
- **Runtime toggle**: The `draw_zones` flag can be changed at runtime via `PATCH /api/v1/settings/general` with `{"display": {"draw_zones": true/false}}`. The broadcast hub responds immediately.
- **Snapshot**: The `GET /api/v1/settings/general` response includes a `"display"` section with `draw_zones` (and `draw_box`, `draw_face`, `draw_skeleton` for future use).

### Database schema

- `rooms.polygon_json` — TEXT column storing JSON array of `[x, y]` points.
- `detection_room_matches` — stores each polygon match with:
  - `transition_type` — 'entered', 'exited', or NULL (heartbeat/static match)
  - `track_id` — Integer track ID for transition tracking
  - `detection_type` — 'face_recognition', 'plate_recognition', or 'fire_smoke'
  - `detection_event_id`, `room_id`, `personnel_id`, `camera_id`, `matched_at_utc`
- `general_settings.draw_zones` — INTEGER column (default `1`), controls visual zone overlay on broadcast frames.

### Sources table and confidence-field ownership

The `sources` table (defined in `app/database.py`) stores 9 per‑source confidence‑field overrides:

| Column | Type | Description |
|--------|------|-------------|
| `source_uri` | TEXT PK | Per‑source key; the well‑known row `'__default__'` holds global defaults |
| `fps` | FLOAT nullable | Per-source playback/delivery FPS override; NULL means use source-native FPS for file/static-video sources when available |
| `fire_confidence` | FLOAT nullable | Fire‑detection confidence threshold |
| `smoke_confidence` | FLOAT nullable | Smoke‑detection confidence threshold |
| `plate_confidence` | FLOAT nullable | Plate‑detection confidence threshold |
| `plate_iou` | FLOAT nullable | Plate‑detection IoU threshold |
| `vehicle_confidence` | FLOAT nullable | Vehicle‑detection confidence threshold |
| `vehicle_iou` | FLOAT nullable | Vehicle‑detection IoU threshold |
| `face_human_confidence` | FLOAT nullable | Human‑detection confidence threshold |
| `face_detection_confidence` | FLOAT nullable | Face‑detection confidence threshold |
| `face_recognition_threshold` | FLOAT nullable | Face‑recognition similarity threshold |
| `loop` | INTEGER default 1 | Per-source file looping toggle for OpenCV/static-video ingest |
| `draw_human` | INTEGER default 1 | Per-source human overlay toggle on broadcast JPEG output |
| `draw_zone` | INTEGER default 1 | Per-source zone polygon overlay toggle on broadcast JPEG output |
| `draw_fire` | INTEGER default 1 | Per-source fire overlay toggle on broadcast JPEG output |
| `draw_smoke` | INTEGER default 1 | Per-source smoke overlay toggle on broadcast JPEG output |
| `draw_vehicle` | INTEGER default 1 | Per-source vehicle overlay toggle on broadcast JPEG output |
| `draw_plate` | INTEGER default 1 | Per-source plate overlay toggle on broadcast JPEG output |
| `updated_at_utc` | TIMESTAMP | Row update timestamp |

**Split ownership with `general_settings.operational_json`:**

- The 9 confidence fields above are stored **only** in the `sources` table as typed SQL columns. Per‑source rows override the global `'__default__'` row.
- The per-source `loop` and `draw_*` overlay flags are stored on normal source rows and flow through `SourceRegistry` / `/api/v1/sources` into ingest or broadcast behavior. They do not belong on the `'__default__'` confidence row.
- The per-source nullable `fps` field is stored on normal source rows and flows through `SourceRegistry` / `/api/v1/sources` into file/static-video ingest pacing. `NULL` preserves source-native FPS when available instead of forcing a global playback rate.
- All other `OperationalSettings` fields (`video_ingest_fps`, `rtsp_transport`, `broadcast_enabled`, `rtsp_source_count`, `video_loop`, …) remain in the `general_settings.operational_json` JSON blob.
- At read time, `Runtime.operational_settings()` and the `GET /api/v1/settings/general` snapshot merge: JSON‑blob fields first, then override the 9 fields from `source_settings.get_default()`. This keeps the JSON blob as a backward‑compatible fallback.
- The 9 source-owned confidence fields are not part of the public `/api/v1/settings/general operational` contract; use `/api/v1/sources` to read or change them. Runtime snapshots may still resolve them internally through `SourceSettingsStore`.
- `SourceSettingsStore` (in `app/core/source_settings_store.py`) provides `get_default()` → returns `OperationalSettings`, `set_default(changes)` → upserts the `__default__` row, `get(source_uri)` / `set(source_uri, …)` / `delete(source_uri)` for per‑source overrides, and `resolve(source_uri)` → merges default + per‑source.

There is no `camera_id` in the `sources` table. Per‑source settings are keyed by `source_uri` only.

`GET /api/v1/sources`, `POST /api/v1/sources`, `PATCH /api/v1/sources/{id}`, and related endpoints return the resolved 9 confidence thresholds on every `SourceResponse`. Creating or updating a source with confidence fields (`fire_confidence`, `plate_confidence`, …) persists per‑source overrides to the `sources` table via `SourceSettingsStore.set()`. When `source_uri` is missing or no override exists the `__default__` global values are returned.

### `cameras` table (source identity)

The `cameras` table (defined in `app/database.py`) now uses **`source_uri` as its primary key** (TEXT, NOT NULL). The `camera_id` column and `SourceRecord.source_id` field have been removed. Every source/camera is identified by its `source_uri`:

- `SourceRecord.source_uri` is now the sole identity field (was `source_id`)
- `CameraResponse` no longer has `camera_id` — use `source_uri` instead
- `SourceResponse` no longer has `source_id` — use `source_uri` instead  
- `CameraCreate` and `SourceCreate` take `source_uri` as the required identifier
- `SourceRegistry` methods (`get`, `require`, `update`, `delete`) key by `source_uri`
- The `sources` table and `cameras` table share the same `source_uri` key, enabling unified per-source settings

Migration `20260725_0016` handles the schema change: NULL `source_uri` values are backfilled from the old `camera_id`, then `camera_id` is dropped and `source_uri` becomes the primary key.

### Multi-agent workflow

Use multiple agents when the task contains independent investigation, implementation, testing, or review work. One primary agent must act as coordinator.

### Coordinator responsibilities

The coordinator must:

1. Read this file and inspect the current worktree before delegation.
2. Restate the requested outcome and identify the real-time acceptance criteria.
3. Split work into bounded, non-overlapping responsibilities.
4. Assign explicit file or module ownership to each implementation agent.
5. Tell every agent that other agents share the worktree and that user changes must not be reverted.
6. Keep at most one agent editing a given file at a time.
7. Integrate the results, review the combined diff, and resolve conflicts.
8. Own the final testing phase and retry policy.
9. Report measured results, remaining risks, and any manual follow-up.

### Recommended roles

- Explorer: read-only code tracing, runtime diagnosis, and evidence collection.
- Backend worker: APIs, runtime wiring, databases, and model management.
- Video/GPU worker: DeepStream, OpenCV, TensorRT, batching, and performance-sensitive paths.
- Test worker: independent regression tests, live smoke tests, and failure reproduction.
- Reviewer: correctness, concurrency, shutdown behavior, bounded memory, and real-time impact.

Combine roles when the change is small. If the platform cannot create sub-agents, the primary agent must execute the same roles sequentially.

### Delegation rules

- Do not delegate vague tasks such as “fix the project.”
- Give each agent a concrete question, outcome, and owned files.
- Agents must not reset, clean, overwrite, or revert unrelated worktree changes.
- Agents must inspect existing code before editing and preserve established architecture.
- Agents must communicate discovered constraints before making cross-module changes.
- Parallel agents must not edit the same files unless the coordinator explicitly serializes their work.
- The coordinator must verify delegated claims in the actual repository or runtime.
- Use read-only explorers in parallel where possible; serialize overlapping implementation work.

## Alembic migrations

Alembic manages all database schema migrations. The migration chain is a single linear branch with no forks.

### Migration files

- Location: `alembic/versions/`
- Naming: `YYYYMMDD_NNN_short_description.py`
- Each migration file contains exactly one `revision` string and one `down_revision` pointing to its immediate predecessor.

### Linear branch rule

- Every revision ID must be unique across all files in `alembic/versions/`.
- Every `down_revision` must reference exactly one preceding migration (or `None` for the base).
- Never create two migration files with the same `revision` value — this creates a divergent branch and breaks `alembic upgrade head`.
- If two migrations are created for the same schema change target, merge them into a single file before committing.

### Adding a new migration

```bash
alembic revision -m "short description of the change"
```

This generates the next sequential file in `alembic/versions/`. Edit the file to add `upgrade()` and `downgrade()` operations. The migration is applied only after running the command below.

### Applying migrations

Before starting the API after a new migration is added:

```bash
alembic upgrade head
```

To verify the current head matches `ALEMBIC_HEAD_REVISION` in `app/database.py`:

```bash
alembic current
```

### Checking the migration chain

```bash
alembic history --verbose
```

The output must show a single linear chain with no branch labels or fork points. If branching is detected, merge the divergent files into one before proceeding.

### Verifying schema at startup

`DatabaseVerifier.verify_schema()` in `app/database.py` checks:
1. All tables in `app/database.py` `metadata` exist in PostgreSQL.
2. The `alembic_version` row matches `ALEMBIC_HEAD_REVISION`.

If the check fails the API refuses to start and instructs the operator to run `alembic upgrade head`.

## Development workflow

For every requested change:

1. Inspect `git status`, relevant source, configuration, tests, and live state when applicable.
2. Establish a baseline for behavior or reproduce the failure.
3. Define a minimal implementation plan and real-time acceptance criteria.
4. Delegate independent work with explicit ownership when multi-agent work is useful.
5. Implement the smallest coherent change.
6. Add or update tests for the changed behavior.
7. Review the combined diff for scope, secrets, concurrency hazards, and performance regressions.
8. Run the mandatory testing phase.
9. If tests fail, follow the three-retry repair policy.
10. Finish only after tests pass, or record the unresolved failure for manual work.

Do not install or upgrade packages unless the user explicitly authorizes it. Use the repository’s existing host environment, container, models, and tools. Never solve a version mismatch by silently changing TensorRT, CUDA, PyTorch, or DeepStream packages.

## Mandatory testing phase

Every development task ends with testing. Documentation-only changes require at least formatting, link/path, and consistency checks. Code changes require tests proportional to risk.

Run these stages in order when applicable:

### Stage 1: static validation

```bash
python -m py_compile <changed-python-files>
git diff --check
docker compose config --quiet
```

Only run commands relevant to the changed files. Do not let unrelated pre-existing worktree whitespace hide validation of the files owned by the task.

### Stage 2: focused tests

Run the smallest tests that directly cover the change, for example:

```bash
python -m pytest tests/test_latest_buffer.py tests/test_router.py -q
python -m pytest tests/test_model_management.py -q
python -m pytest tests/test_face_recognition.py -q
```

### Stage 3: regression suite

```bash
python -m pytest -q
```

Compare failures with the baseline. A pre-existing failure must be reported clearly and must not be claimed as caused or fixed without evidence.

### Stage 4: runtime validation

Runtime, ingestion, model, API, WebSocket, concurrency, or performance changes require live validation in the actual target container when available:

```bash
curl -fsS http://127.0.0.1:9999/health
curl -fsS http://127.0.0.1:9999/api/v1/models/settings
docker logs --tail 200 merged-video-ai-router
```

For TensorRT changes, validate deserialization, execution-context creation, input profiles, and at least one real inference at minimum and maximum supported batch sizes. Engines must be built on the device and TensorRT runtime that will execute them.

For real-time pipeline changes, sample runtime counters long enough to show frame progression, queue behavior, failure counts, batch latency, and frame age. Backend startup alone is not sufficient proof.

## Three-retry repair policy

The initial testing phase is attempt zero. If any required test fails, Codex or OpenCode must diagnose and try to repair the issue automatically. A maximum of three repair retries is allowed.

For each retry:

1. Capture the exact failing command and the relevant error.
2. Identify a concrete root-cause hypothesis.
3. Apply a scoped fix to production code or valid test expectations.
4. Rerun the failed test first.
5. Rerun the focused regression tests after the failed test passes.
6. Rerun broader validation if the fix affects shared behavior.

Retry definitions:

- Retry 1: first diagnosis, fix, and retest.
- Retry 2: revised diagnosis, fix, and retest.
- Retry 3: final diagnosis, fix, and retest.

Do not repeat the same command three times without changing the diagnosis or implementation. Do not weaken, delete, skip, or mark tests as expected failures merely to obtain a green result. Update a test only when the intended contract genuinely changed, and state that contract change.

If all required tests pass during any retry, continue to the remaining testing stages and finish normally.

## Unresolved failure log

If the third repair retry still fails, stop automatic repair and append an entry to the repository-root text file:

```text
agent_test_failures.log
```

The file is append-only. Never erase or rewrite earlier entries. Create it only when an unresolved failure occurs. Do not include secrets, camera credentials, API keys, tokens, private URLs, or full sensitive payloads.

Each entry must use this format:

```text
================================================================================
Timestamp UTC: <ISO-8601 timestamp>
Agent: <Codex or OpenCode and model if known>
Task: <short requested outcome>
Status: FAILED_AFTER_3_RETRIES
Changed files: <comma-separated paths>
Baseline: <baseline result or not available>
Failed command: <exact command>
Final error: <concise error excerpt>
Retry 1: <hypothesis, change, result>
Retry 2: <hypothesis, change, result>
Retry 3: <hypothesis, change, result>
Current diagnosis: <best-supported root cause>
Manual next step: <specific action for the user>
================================================================================
```

After logging, the agent must:

- Tell the user that automatic repair stopped after three retries.
- Link or name `agent_test_failures.log`.
- Summarize what works and what remains broken.
- Leave diagnostic changes only when they are safe and useful.
- Never claim the development task is complete.

## Definition of done

A development task is complete only when:

- The requested behavior is implemented in the correct repository and runtime boundary.
- Relevant tests were added or updated.
- Static checks and focused tests pass.
- The regression suite passes, or pre-existing failures are proven and reported.
- Required live validation passes in the target environment.
- Real-time impact is measured for hot-path changes.
- No secrets or private camera URLs were exposed.
- The combined diff was reviewed and unrelated user changes were preserved.
- No unresolved required test remains. If one remains after three retries, it is logged and the task is handed back as incomplete.

## Coding conventions

- Add `from __future__ import annotations` to every Python file.
- Type-hint every function signature.
- Use `snake_case` for functions, variables, and modules.
- Use `PascalCase` for classes.
- Prefix private methods with `_`.
- Use `slots=True` for dataclasses and `frozen=True` for settings and value objects.
- Keep imports grouped as standard library, third-party, and local.
- Avoid comments in code unless they are necessary to explain a non-obvious invariant.
- Prefer minimum coherent diffs over unrelated refactors.
- Preserve Persian text, RTL behavior, and UTF-8 when editing user-facing Persian surfaces.

API routers use this pattern:

```python
router = APIRouter(prefix="/api/v1/<resource>", tags=["<tag-group>"])


def get_runtime() -> Runtime:
    from app.main import runtime

    return runtime
```

Use `Annotated` for FastAPI `UploadFile`, `Form`, `File`, and `Depends` parameters. Use Pydantic models for request and response contracts.

## Smoke-test / processor-tests convention

Every new app or feature module MUST add a smoke-test endpoint in `app/api/processor_tests.py` to
validate that the module is wired correctly end-to-end. This is mandatory regardless of whether
the module uses a real model pipeline or only a store layer.

- For store-only modules (e.g. Personnel), add a `POST /api/v1/tests/<module>/smoke` endpoint that
  exercises CRUD, search, image/asset operations, and import/export. The test must work in both
  `PROCESSOR_MODE=mock` and `PROCESSOR_MODE=real`.
- For pipeline modules (e.g. FaceRecognition, Plate, FireSmoke), add both `GET /api/v1/tests/<module>/models`
  (model file existence check) and `POST /api/v1/tests/<module>/full-pipeline` (upload-image pipeline test).
- Register the new test endpoint in the `GET /api/v1/tests/all` all-in-one status check.
- Add corresponding unit/integration tests under `tests/` that call the smoke-test endpoint and
  verify every step passes.

## Runtime and TensorRT constraints

- TensorRT engines are version-, OS-, GPU-, and platform-specific.
- `trt107` artifacts must be built and executed with the Python TensorRT 10.7 runtime.
- Do not assume system `trtexec` and Python `tensorrt` have the same version; verify both.
- Build engines inside the same container and on the same device class used for inference.
- Detector profiles should normally be `(-1, 3, 640, 640)` with batch 1–8 and fixed spatial dimensions.
- ArcFace profiles should normally be `(-1, 3, 112, 112)` with batch 1–64 and fixed spatial dimensions.
- Use `scripts/build_all_engines.py` to rebuild all device-specific engines.
- Persistent selections under `/api/v1/models/settings` can override `.env` and `app/config.py` defaults.
- Do not install `tensorrt-cu12` into the DeepStream image to repair an engine mismatch.

## Common commands

Development and tests:

```bash
python3 run.py
PROCESSOR_MODE=mock python3 run.py
python -m pytest -q
python3 scripts/smoke_test.py
PROCESSOR_MODE=mock python3 scripts/mock_load_test.py
```

Production container:

```bash
docker compose build
docker compose up -d --no-build
docker compose logs -f video-ai-router
```

Build all TensorRT engines inside the target container:

```bash
python3 scripts/build_all_engines.py --batch 8 --workspace 4 --device 0
```

Build a detector with dynamic batch and fixed spatial dimensions:

```bash
python3 -m app.model_export_worker \
  --source <input.pt> \
  --target <output.engine> \
  --format engine \
  --imgsz 640 \
  --batch 8 \
  --workspace 4 \
  --device 0 \
  --half \
  --dynamic-batch-only
```

Inspect live state:

```bash
curl -fsS http://127.0.0.1:9999/health
curl -fsS http://127.0.0.1:9999/api/v1/models/settings
docker logs --tail 200 merged-video-ai-router
```

## Performance tuning

### Configuration environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_INGEST_FPS` | Python 10.0; Compose 5.0 | Per-source ceiling for cameras with AI tasks; resolved container environment wins |
| `VIDEO_PREVIEW_FPS` | Python 10.0; Compose 25.0 | Per-source ceiling for play-only preview cameras |
| `FIRE_SMOKE_MAX_WAIT_MS` | 50.0 | Max wait time to fill fire/smoke batch |
| `PLATE_MAX_WAIT_MS` | 50.0 | Max wait time to fill plate batch |
| `FACE_MAX_WAIT_MS` | 50.0 | Max wait time to fill face batch |
| `WORKER_THREADS` | 1 | Number of threads sharing each task processor; values above 1 require target-runtime thread-safety proof |
| `SKIP_TASKLESS_SOURCES` | false | Legacy compatibility setting; enabled sources always reach router broadcast, while task assignments control AI worker submission |

### Performance optimization strategies

1. **Measure first**: sample `/api/v1/diagnostics/fps` over the same bounded window before and after a tuning change.
2. **Resolve configuration precedence**: check `docker compose config`; service `environment` values override `env_file` and Python defaults.
3. **Tune one limit at a time**: raise `VIDEO_INGEST_FPS` gradually only when decoders supply enough frames and worker replacements remain zero or acceptable.
4. **Protect ordering and state**: do not increase `WORKER_THREADS` until TensorRT contexts, trackers, processors, callbacks, and stores are validated as thread-safe and per-source result ordering is preserved.
5. **Preserve source-selected delivery**: `/api/v1/sources` owns frontend stream selection through `enabled`; `tasks` selects AI workers. Every enabled frame is published to the bounded JPEG broadcast path before optional AI results upgrade it, so model latency or failure does not remove the video.

### CPU Bottlenecks and Optimizations

**Primary CPU Bottlenecks Identified:**

1. **Broadcast annotation (CRITICAL)**: The `AnnotatedBroadcastHub` was blocking the main processing pipeline with CPU-intensive operations:
   - Frame copying: 3 copies per frame (original, annotation, overlay)
   - Drawing operations: `cv2.rectangle()`, `cv2.putText()` for each detection
   - JPEG encoding: `cv2.imencode()` for full and wall resolution
   - Estimated time: 20-50ms per frame per camera

2. **DeepStream frame extraction**: `buffer.extract_dup()` and `.copy()` operations copy frames from GPU to CPU memory (5-15ms per frame)

3. **Frame resizing**: `cv2.resize()` in DeepStream ingestor (1-3ms per frame)

**Optimizations Implemented:**

1. **Async Broadcast Rendering** (`app/core/broadcast.py`):
   - Added `async_render=True` parameter (default enabled)
   - Rendering now happens in a background thread
   - `publish_result()` returns in <1ms instead of 20-50ms
   - **Result: 23.6x speedup** for frame submission

2. **GPU JPEG Encoding** (`app/core/broadcast.py`):
   - Added `use_gpu_jpeg=True` parameter (default enabled)
   - Uses `nvjpegenc` GStreamer element when available (DeepStream Docker)
   - Falls back to CPU encoding when GPU not available
   - **Result: Automatic GPU acceleration in production**

3. **Frame Copy Reduction**:
   - Reduced from 3 copies to 1 copy per frame in broadcast
   - Used in-place operations for header overlay
   - Still need 1 copy for annotation (unavoidable)

4. **Batch Processing Optimization**:
   - Increased `*_MAX_WAIT_MS` to 50ms for better batching
   - Allows accumulating 8 frames for efficient GPU inference

**Measured Performance Impact:**

| Operation | Before | After | Improvement |
|-----------|--------|-------|-------------|
| publish_result() | 2-5ms | 0.18ms | **23.6x faster** |
| Frame render | 20-50ms | 3.89ms | **5-13x faster** |
| Total per frame | 27-70ms | 4-5ms | **6-14x faster** |

**GPU Acceleration Path:**

For production deployment in DeepStream Docker:
1. `nvjpegenc` automatically used for GPU JPEG encoding
2. `nvdsosd` can be added for GPU-based annotation (future optimization)
3. Frames stay on GPU longer, reducing CPU overhead

**Remaining Bottlenecks (Lower Priority):**

1. **DeepStream frame extraction**: GPU→CPU copy is unavoidable but can be optimized with zero-copy or GPU preprocessing
2. **Model inference**: GPU-bound, not CPU-bound
3. **Database operations**: Async and non-blocking

### Scenario: 8 cameras at 5 FPS each

Eight cameras admitted at 5 FPS yield 40 aggregate ingest frames per second, but task load depends on each camera's assignments. This arithmetic is not proof of processing capacity. Report measured per-camera router FPS, per-camera/task processed FPS, replacements, batch latency, GPU utilization, and frame age.

### Monitoring FPS

Check real-time performance with a bounded sample:
```bash
curl -fsS "http://127.0.0.1:9999/api/v1/diagnostics/fps?sample_seconds=5&expected_fps=25"
```

Key metrics:
- `appsink_sample_fps`, `rate_limited_fps`, `admitted_fps`, and `router_frame_fps` per camera
- `processed_fps` and `latest_frame_replacement_fps` per camera/task
- `last_batch_ms` as end-to-end worker batch time; processor `last_inference_ms` is the closer model-only signal
- `pending_sources`, frame age, failures, and full-versus-wall encoded bandwidth

## Security and worktree safety

- Never expose RTSP credentials, API keys, model-service tokens, or private URLs.
- Redact secrets from logs, test output, examples, summaries, and failure entries.
- Inspect `git status` before editing.
- Treat existing modifications and untracked files as user-owned unless proven otherwise.
- Never use destructive Git or filesystem commands without explicit authorization.
- Do not rewrite unrelated code to make tests pass.
- Keep model weights, TensorRT engines, runtime databases, generated media, and local secrets out of commits unless explicitly requested.

<!-- OPENCODE-REALTIME-WORKFLOW:START -->
# Real-Time Project Agent Rules

## Core interpretation rule

Treat a user request as desired behavior, not as a complete file list. Infer and implement every necessary change across configuration, schemas, endpoints, WebSockets, UI, services, workers, queues, processing pipelines, model invocation, persistence, event payloads, logging, metrics, deployment, documentation, and tests.

Never leave a feature locally implemented but unreachable. When an option changes externally controlled behavior, determine and update the correct external interface even when the user did not explicitly mention an endpoint or UI control. When no external interface is required, document the verified reason.

## Maintenance-phase behavior

This project is in maintenance and fine-tuning. Assume existing behavior should remain stable unless the user explicitly changes its contract. Treat each request as a behavioral outcome and inspect the complete connected lifecycle before editing code.

Apply the following proportionally to the risk and reach of each maintenance request. A small local fix needs a targeted audit, not a project-wide inventory:

1. Trace the affected entity or value from every input through validation, persistence, processing, output, and cleanup.
2. Build a concise impact map covering API contracts, UI/client calls, stores, database constraints, files, model or vector systems, caches, background workers, events, logs, tests, and deployment state that actually participate in the behavior.
3. Identify resources owned by the affected entity. Create, update, and delete operations must keep those resources consistent and must not leave orphaned state.
4. Search targeted names, types, routes, or helpers for plausible analogous locations. When analogues exist, classify them in the impact map as `AFFECTED`, `NOT_AFFECTED`, or `DEFERRED_WITH_REASON`.
5. Apply a shared fix to analogous locations only when they enforce the same verified invariant and the change is safe. Do not perform speculative project-wide refactors.
6. Prefer one established validator, service, cleanup helper, or contract definition over copied implementations. Remove duplication only when the replacement is behaviorally equivalent and covered by tests.
7. Verify direct behavior, connected side effects, compatibility, and at least one negative or failure path. A successful local function call is not enough when downstream state exists.
8. Preserve unrelated working behavior and user-owned worktree changes. Maintenance work is not permission to redesign a stable section.

### Entity lifecycle and ownership

Before changing CRUD behavior, identify all state owned by or referencing the entity. Deletion is complete only when the intended database records and all owned external artifacts are removed or deliberately retained according to a verified rule.

For example, deleting Personnel may require coordinated handling of:

- the personnel database row and dependent image rows;
- Qdrant face vectors and identity aliases;
- original files under `personnel_snapshots`;
- aligned files under `personnel_cropped_faces`;
- cached results or background work that can recreate stale state;
- foreign-key references, detection history, attendance history, and audit records that must be preserved, nulled, or deleted according to the existing contract;
- API, WebSocket, and UI state that must stop exposing the deleted entity.

Never infer that all related history should be deleted. Distinguish owned artifacts from historical or audit records, verify existing retention behavior, and test both cleanup and preservation.

### Contract and file-type consistency

External contracts must describe and validate the real payload, not merely accept a loosely typed value.

- Image endpoints use multipart `UploadFile` fields with binary image OpenAPI metadata and centralized runtime validation selected from MIME, extension, magic-byte, size, and decode checks according to the endpoint's established contract. Do not require a filename extension when that contract permits extensionless uploads.
- Excel import endpoints advertise and validate Excel workbook types and reject unrelated or malformed files.
- ZIP import endpoints advertise and validate ZIP archives, inspect archive safety, and reject unrelated or malformed files.
- File metadata in OpenAPI is a client hint; server-side validation remains mandatory.
- When one upload contract is corrected, audit analogous upload endpoints for the same mismatch. Change them only when the same defect is confirmed, and add focused contract tests for each affected endpoint.

Apply the same reasoning to identifiers, enums, timestamps, optional foreign keys, pagination shapes, response models, and permission checks. Do not report every database integrity failure as the same business error; map verified constraints to accurate, actionable messages.

### Connectivity and regression checks

For a change in one section, test the connected boundaries that can be affected. Examples include create/list/search consistency, model enrollment followed by deletion, file creation followed by rollback and cleanup, database mutation followed by API/WebSocket visibility, and configuration changes followed by actual container startup.

If a shared change affects multiple modules, add one focused test per distinct contract plus an integration test for the connected flow. Record pre-existing failures separately and do not weaken tests to hide them.

### Phased TODO execution

Every maintenance task must be managed through a visible phased TODO list. Create it after the initial repository and worktree inspection and before application-code edits. Keep exactly one phase `IN_PROGRESS` at a time and update the list whenever a phase finishes, fails, or materially changes.

Use these statuses:

- `TODO`: not started;
- `IN_PROGRESS`: active work;
- `DONE`: acceptance evidence collected;
- `BLOCKED`: cannot continue safely without user input or an external dependency.

Use this default phase structure, combining low-risk phases only when the task is genuinely small:

1. `Baseline and reproduction`: confirm the actual failure, current contract, worktree state, and target runtime.
2. `Impact and ownership map`: trace connected inputs, stores, external resources, outputs, analogous locations, risks, and acceptance tests.
3. `Direct implementation`: implement the smallest coherent fix for the requested behavior.
4. `Connected consistency`: repair verified downstream effects and affected analogous contracts without speculative expansion.
5. `Validation`: run static, focused, integration, regression, runtime, and deployment checks in the required order as applicable.
6. `Knowledge and handoff`: update durable project knowledge when necessary, review the final diff, and report evidence, remaining risks, rollback, and any deferred items.

A phase is not `DONE` because code was written. Mark it `DONE` only after its stated evidence or acceptance check passes. If new evidence invalidates an earlier phase, reopen that phase and update the TODO list instead of silently continuing.

The TODO list is an execution control, not a ceremonial plan. Keep it concise, reference concrete modules or contracts, and do not create redundant subtasks that repeat the same investigation or validation.

### Reuse and redundancy control

Do not solve the same invariant repeatedly in route handlers. Before adding logic, search for an existing service, validator, serializer, cleanup helper, or test fixture. Extend the authoritative implementation when safe, then migrate affected callers deliberately. Avoid parallel legacy/current implementations unless compatibility is verified and explicitly required.

When similar code cannot be unified safely, keep the implementations separate and record the reason. Three similar lines are preferable to a risky abstraction; repeated business rules that can drift require a shared owner.

### Maintenance knowledge updates

At each investigation, implementation, and validation stage, consider whether a newly verified fact is reusable project knowledge. Follow the `Knowledge maintenance` rules below and update `AGENTS.md` or the appropriate `.agentic` file only when the fact is stable and useful for future maintenance, such as an ownership relationship, integration constraint, required validation, authoritative helper, runtime command, or known failure mode.

Do not update knowledge files merely to narrate the current task. Avoid duplicate rules, consolidate overlapping guidance, and never record guesses, transient logs, credentials, private URLs, or one-off debugging details.

## Required workflow before code

1. Inspect the current repository and find the actual execution path.
2. Identify the most similar existing behavior and trace its complete lifecycle.
3. Produce a concise impact map: definitions, inputs, transformations, consumers, outputs, tests, deployment, and risks.
4. Define acceptance tests and affected evaluation sections.
5. Implement the smallest coherent end-to-end change.
6. Run focused, integration, regression, runtime, and deployment checks as applicable.
7. Update verified project knowledge and machine-readable state.
8. Report evidence, assumptions, failures, untested areas, risks, and rollback steps.

## Excel-catalog migration

Previous-project options are provided through an Excel catalog, not through a previous source repository. Migration is prompt-scoped and incremental.

Only the apps, routes, options, or behaviors selected in the current user prompt may be implemented. Required dependency and cross-layer changes for those selected items are allowed and mandatory. Every unselected catalog item must remain `PENDING`. Do not automatically continue to another group.

The Excel catalog is a partial specification. For each selected item, use evidence in this order:

1. current user prompt;
2. selected Excel fields;
3. verified current-project patterns;
4. explicitly recorded safe assumptions.

Never invent exact previous-project behavior or claim behavioral parity when previous implementation evidence was not supplied.

For each selected catalog item:

- resolve the exact app, route, option, or feature;
- extract supplied purpose, method, input/output, default, validation, permissions, runtime effect, and notes;
- inspect the current project for the correct architectural destination and similar implementations;
- classify specification coverage as `CONFIRMED_FROM_INPUT`, `IMPLEMENTED_BY_TARGET_CONVENTION`, `PARTIALLY_SPECIFIED`, `BLOCKED_NEEDS_DETAILS`, or `NOT_EVALUATED`;
- propagate the selected behavior through every required layer;
- test omitted, default, valid, boundary, invalid, permission, runtime, downstream, compatibility, and performance behavior.

When missing details can be handled safely through a strong existing target-project convention, record the assumption and proceed. When a security-sensitive or correctness-critical requirement cannot be inferred safely, mark only that item `BLOCKED_NEEDS_DETAILS`.

## Project-wide evaluation

Maintain `.agentic/evaluation/section-registry.json` for all important sections that actually exist. Each critical section must have suitable static, unit, component, integration, regression, end-to-end, runtime, performance, or deployment checks.

Use these statuses honestly:

- `PASS`
- `PASS_WITH_WARNINGS`
- `FAIL`
- `BLOCKED`
- `NOT_CONFIGURED`
- `NOT_TESTED`
- `NOT_APPLICABLE`
- `NOT_RUNTIME_VALIDATED`

Do not invent thresholds. Derive them from explicit requirements, verified current-project baselines, existing tests, production evidence supplied by the user, or explicit user direction.

## Real-time completion gate

A build, container start, or successful unit test is not enough. When the project supports a representative runtime, validate the actual flow from input stream through processing and detection/OCR to event generation, persistence, API/WebSocket output, and downstream consumer.

Measure available indicators such as input FPS, processed FPS, dropped frames, latency, queue depth, CPU, GPU, memory, errors, reconnects, and delivery time.

Do not claim `PASS` for a critical real-time feature that was not runtime validated. Use `NOT_RUNTIME_VALIDATED` and explain why.

## Knowledge maintenance

Update `AGENTS.md` or `.agentic` knowledge files only with reusable, important, verified facts: commands, architecture relationships, required services, integration rules, evaluation requirements, deployment constraints, known failure conditions, and validated performance limits.

Do not store guesses, temporary debugging notes, credentials, machine-specific secrets, or unverified assumptions as project facts. Task-specific assumptions belong in migration state and reports.

## Source type and allocation

Cameras have a `source_type` column (`rtsp` or `static_video`, default `rtsp`) with a CHECK constraint (`ck_cameras_source_type`).

- `SourceRecord.source_type` — stored as a field on the record, default `RTSP`
- `SourceRegistry.list_by_type(source_type)` — list sources of one type
- `SourceRegistry.active_sources()` — list all enabled sources
- API schemas (`CameraCreate`, `CameraUpdate`, `CameraReplace`, `CameraResponse`) all include `source_type` with validation

Each ingestor filters by `source_type_filter` parameter (default `RTSP`):
- `VideoFileIngestor._process_once_unlocked()` only processes records matching `self.source_type_filter`
- `DeepStreamIngestor._active_records()` only returns records matching `self.source_type_filter`

Allocation caps:
- `OperationalSettings.rtsp_source_count` (default 128) — max concurrent RTSP sources
- `OperationalSettings.static_video_source_count` (default 16) — max concurrent static video sources
- Each ingestor enforces `max_sources` before opening new sources
- `apply_operational_settings()` pushes caps to both ingestors at runtime

Two ingestor instances in `Runtime`:
- `video_ingestor` — RTSP sources (DeepStream or OpenCV)
- `static_video_ingestor` — always `StaticVideoFileIngestor` (OpenCV-based, loop=False, 30 FPS default)
- `_restart_ingestor_source(camera_id)` routes to the correct ingestor based on `source_type`

`StaticVideoFileIngestor` is a `VideoFileIngestor` subclass defaulting to `source_type_filter=static_video`, `loop=False`, `max_sources=16`.

### Static video upload API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/static-videos/upload` | POST | Upload video file, return `source_uri` path |
| `/api/v1/static-videos` | POST | Upload + create camera in one call |
| `/api/v1/static-videos` | GET | List all static video camera sources |

Uploaded files are saved under `STATIC_VIDEO_UPLOAD_PATH` (default `saved_media/static_videos/`) with a UUID prefix to prevent name collisions. The returned `source_uri` is an absolute filesystem path that becomes the camera's `source_uri`. The `StaticVideoFileIngestor` automatically picks up new cameras on its next processing loop.

### Camera vs Source API separation

Cameras and sources share the same underlying `SourceRecord`/`SourceRegistry`/DB, but have different API surfaces:

| Concern | Camera API (`/api/v1/cameras`) | Source API (`/api/v1/sources`) |
|---------|-------------------------------|-------------------------------|
| `enabled`/`tasks` | **NOT exposed** — camera is about device identity | **Exposed** — operational state belongs here |
| `POST /{id}/enable`, `/disable` | Not available | Available |
| `PUT /bulk/task-assignment` | Not available | Available |
| `GET /preview-config` | Not available (removed) | **Available** — returns `enabled`/`tasks` per source |
| `enable`/`disable` in create/update schemas | Not in `CameraCreate`/`CameraUpdate`/`CameraReplace`/`CameraResponse` | In `SourceCreate`/`SourceUpdate`/`SourceResponse` |

The dashboard fetches `/api/v1/sources/preview-config` to get source operational state (enabled + tasks) for card rendering.

## Context efficiency

Use specialized subagents and on-demand skills. Keep the main context focused on decisions, interfaces, evidence, and unresolved risks. Do not send the entire repository to every subagent. Work one coherent feature or option group at a time.
<!-- OPENCODE-REALTIME-WORKFLOW:END -->
