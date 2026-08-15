# Redis Detection Events Implementation Plan

## Goal

Publish lightweight human, fire/smoke, and plate detection events to Redis without
saving snapshots or event clips on the live inference path. A separate worker will
locate the corresponding live-video segments in MinIO, extract media, and finalize
the database log.

## Target Flow

```text
live frame -> inference/tracking -> bounded event queue -> Redis Stream
live recording -> segment upload -> MinIO -> recording segment manifest
Redis consumer -> match event to segments -> extract media -> database log
```

The live/GPU path must never wait for Redis, MinIO, video decoding, or database I/O.

## Redis Streams

Use Redis Streams with consumer groups, not Pub/Sub:

```text
detection:human:v1
detection:fire:v1
detection:plate:v1
recording:segments:v1
detection:dead-letter:v1
```

Start with one stream per event type for all cameras. Do not create a stream per
camera unless measured load requires partitioning.

## Event Contracts

### Human Observation

Publish one event when one tracked person observation ends on one camera.

```json
{
  "schema_version": 1,
  "event_id": "01JABC123",
  "event_name": "human",
  "event_type": "track_ended",
  "camera_id": "camera-17",
  "room_id": 5,
  "tracking_session_id": "session-123",
  "track_id": 42,
  "personnel_id": 125,
  "ref_img_id": "personnel_125_image_1",
  "name": "Ali Ahmadi",
  "recognition_status": "recognized",
  "recognition_confidence": 0.94,
  "first_seen_at_utc": "2026-08-15T10:00:00Z",
  "last_seen_at_utc": "2026-08-15T10:00:08Z",
  "best_frame_at_utc": "2026-08-15T10:00:04Z",
  "best_frame_index": 18542,
  "bounding_box": [100, 80, 350, 700],
  "bounding_box_format": "xyxy",
  "frame_width": 1920,
  "frame_height": 1080,
  "snapshot_quality": 0.91,
  "clip_start_at_utc": "2026-08-15T09:59:57Z",
  "clip_end_at_utc": "2026-08-15T10:00:11Z",
  "counts_for_attendance": true,
  "created_at_utc": "2026-08-15T10:00:08Z"
}
```

For an unknown person, `personnel_id` and `ref_img_id` are null, `name` is
`Unknown`, and `recognition_status` is `unknown`.

Each camera publishes an independent observation. Recognized observations can be
grouped later by `personnel_id`; unknown observations must remain independent
unless cross-camera person re-identification is implemented.

### Fire/Smoke Observation

```json
{
  "schema_version": 1,
  "event_id": "01JFIRE123",
  "event_name": "fire_smoke",
  "event_type": "incident_ended",
  "incident_id": "incident-44",
  "camera_id": "camera-3",
  "room_id": 7,
  "hazard_type": "fire",
  "severity": "high",
  "fire_count": 3,
  "smoke_count": 0,
  "confidence": 0.96,
  "first_seen_at_utc": "2026-08-15T10:10:00Z",
  "last_seen_at_utc": "2026-08-15T10:10:12Z",
  "best_frame_at_utc": "2026-08-15T10:10:06Z",
  "bounding_boxes": [[110, 90, 540, 620]],
  "bounding_box_format": "xyxy",
  "frame_width": 1920,
  "frame_height": 1080,
  "clip_start_at_utc": "2026-08-15T10:09:57Z",
  "clip_end_at_utc": "2026-08-15T10:10:15Z",
  "created_at_utc": "2026-08-15T10:10:12Z"
}
```

Fire/smoke events do not contain `ref_img_id`.

### Plate Observation

```json
{
  "schema_version": 1,
  "event_id": "01JPLATE123",
  "event_name": "plate",
  "event_type": "plate_detected",
  "camera_id": "camera-8",
  "room_id": 2,
  "plate_id": 31,
  "plate_number": "12A34567",
  "raw_plate_text": "12A34567",
  "recognition_confidence": 0.92,
  "detected_at_utc": "2026-08-15T10:20:04Z",
  "best_frame_at_utc": "2026-08-15T10:20:04Z",
  "frame_index": 9921,
  "bounding_box": [720, 610, 940, 690],
  "bounding_box_format": "xyxy",
  "frame_width": 1920,
  "frame_height": 1080,
  "clip_start_at_utc": "2026-08-15T10:20:01Z",
  "clip_end_at_utc": "2026-08-15T10:20:07Z",
  "created_at_utc": "2026-08-15T10:20:04Z"
}
```

Plate events use nullable `plate_id` for registered plates and do not contain
`ref_img_id`.

### Recording Segment Manifest

Publish this only after a segment is available in MinIO:

```json
{
  "schema_version": 1,
  "segment_id": "camera-17:20260815T100000Z",
  "camera_id": "camera-17",
  "bucket": "recordings",
  "object_key": "camera-17/2026/08/15/10/segment-0001.mp4",
  "started_at_utc": "2026-08-15T10:00:00Z",
  "ended_at_utc": "2026-08-15T10:01:00Z",
  "frame_width": 1920,
  "frame_height": 1080,
  "fps": 25,
  "status": "ready"
}
```

## Implementation Phases

### Phase 1: Observe Existing Flow

Scope:

- Trace frame production through each detection store.
- Trace live recording completion and MinIO upload.
- Trace the existing Redis recording scheduler separately from detection events.
- Capture evidence for the first boundary that prevents event-driven persistence.

Verification:

- Document the actual call paths and process boundaries.
- Confirm whether recording timestamps use source capture time or wall-clock time.
- Confirm MinIO object naming and segment duration.
- Make no application-code changes.

Exit criteria:

- The complete current flow is understood and the first change point is identified.

Phase 1 findings:

- Detection dispatch is `ingestor -> TaskRouter.submit_round -> FramePacket ->
  TaskWorker -> processor`; `result_store.publish` and result/location observers
  are subsequent sequential operations, not an observer chain downstream of
  publish. Observer-side frame copying, state handling, and location matching
  execute synchronously on the task-worker thread within the detection process
  and are the current blocking boundary. Plate, fire/smoke, and human media-log
  persistence then crosses bounded queues to asynchronous writer threads rather
  than introducing Redis I/O into ingestion, processors, or `TaskWorker`.
- The plate observer copies the frame and queues it to `plate-log-writer`, which
  writes the filesystem and database records using `captured_at_utc`. The
  fire/smoke observer buffers and annotates evidence, then queues filesystem and
  database work to its writer; persisted timing currently uses
  `processed_at_utc`, and incident evidence is updated while the incident is
  active. The human path passes through the runtime polygon/location observer
  and `HumanLogStore` state/evidence before queueing work to
  `human-media-writer`; a disappeared track is its semantic completion boundary.
- There are two independent recording systems. Continuous recording is the live
  branch `splitmuxsink` pipeline plus an upload thread. It timestamps fragment
  open/close with wall-clock UTC, not frame capture timestamps, and targets
  120-second segments by default. Keyframe-aligned splitting can vary the actual
  duration, and branch lifetime can produce a shorter final fragment. This path
  uses the fullscreen-profile branch, but enabled source attachment automatically
  acquires a durable continuous reference; no user needs to open fullscreen.
  Recording still runs only while references keep the branch active.
- Continuous files are named locally
  `camera-{camera_id}-{session}-%05d.mp4`; MinIO objects use
  `continuous/{camera_id}/{YYYY/MM/DD}/{filename}` with a
  `{object_name}.json` sidecar. The separate scheduled path is
  `API/DB -> Redis scheduler sorted set/strings/locks -> coordinator -> executor
  -> storage/MinIO` and stores `recordings/{recording-job-uuid}.mp4`. Its existing
  Redis keys and scheduling behavior are independent of future detection
  Streams.
- The first future publication points are completed semantic outcomes: human
  disappeared-track, fire/smoke `incident_ended`, and successful plate OCR
  candidates. A future publisher should enqueue metadata at those boundaries,
  not perform Redis I/O directly in ingestion, processors, or `TaskWorker`.

### Phase 2: Define and Validate Event Schemas

Scope:

- Add typed schemas for human, fire/smoke, plate, and recording-segment events.
- Validate required IDs, UTC timestamps, bounding boxes, dimensions, and confidence.
- Keep `ref_img_id` only on human events.
- Add `schema_version` to every event.

Verification:

- Unit tests accept valid recognized and unknown human events.
- Unit tests reject invalid time ranges, dimensions, and bounding boxes.
- JSON round-trip tests preserve every field.

Exit criteria:

- Event contracts are stable before Redis integration begins.

### Phase 3: Add a Non-Blocking Redis Publisher

Scope:

- Reuse the existing Redis dependency and configuration mechanism.
- Add a bounded in-process queue between live callbacks and Redis I/O.
- Perform `XADD` from a background thread/task, never from the GPU callback.
- Add configurable queue capacity and Redis stream names.
- Track published, queued, dropped, retried, and failed counts.

Verification:

- Simulated slow/unavailable Redis does not block the live callback.
- Queue overflow follows an explicit policy and increments metrics.
- Events are visible in the expected Redis Stream.

Exit criteria:

- Publishing is non-blocking and observable.

### Phase 4: Publish Human Observation Events

Scope:

- Maintain first-seen and last-seen timestamps for each camera track.
- Preserve the best frame timestamp, frame index, source bounding box, dimensions,
  and quality without retaining the frame for persistence.
- Publish one `track_ended` event per camera observation.
- Include nullable `personnel_id` and `ref_img_id`.
- Do not group unknown people across cameras.

Verification:

- Known and unknown tracks produce the expected event exactly once.
- Two cameras observing the same known person produce two independent events.
- No snapshot, video, MinIO, or database I/O occurs in the live callback.

Exit criteria:

- Human metadata reaches Redis with enough information to recover media later.

### Phase 5: Publish Fire/Smoke and Plate Events

Scope:

- Publish fire/smoke incident events using incident time ranges and best evidence.
- Publish plate events using detection time and best plate bounding box.
- Keep existing inference and severity/recognition behavior unchanged.
- Exclude `ref_img_id` from both event types.

Verification:

- Fire, smoke, and combined incidents serialize correctly.
- Registered and unregistered plates serialize correctly.
- Event publication does not alter current inference results.

Exit criteria:

- All detection types publish their versioned metadata contracts.

### Phase 6: Publish and Index Recording Segments

Scope:

- Publish a segment manifest only after successful MinIO upload.
- Persist a searchable segment index by camera and UTC time range.
- Use PostgreSQL for durable indexing unless existing architecture provides an
  equally durable index.
- Make segment publication idempotent by `segment_id`.

Verification:

- Every uploaded segment has one durable manifest.
- Queries locate all segments overlapping a requested event interval.
- Segment-boundary and multi-segment intervals are covered by tests.

Exit criteria:

- Detection events can reliably resolve their source recordings.

### Phase 7: Implement the Media and Log Worker

Scope:

- Create consumer groups for detection streams.
- Resolve overlapping recording segments by camera and UTC interval.
- Download or stream the required MinIO objects.
- Extract the best-frame snapshot using timestamp and bounding box.
- Trim and concatenate event clips crossing segment boundaries.
- Upload derived media with deterministic keys based on `event_id`.
- Insert or update the database log with `media_pending`, `ready`, or `failed` state.
- Acknowledge Redis only after durable completion.

Verification:

- Worker restart does not duplicate database rows or media objects.
- Snapshot extraction scales bounding boxes if recording dimensions differ.
- Missing but expected segments trigger retry rather than permanent failure.
- One failed event does not block unrelated cameras.

Exit criteria:

- Redis metadata produces finalized database logs and derived MinIO media.

### Phase 8: Add Recovery and Operations

Scope:

- Recover abandoned pending entries with `XAUTOCLAIM`.
- Add bounded retries with backoff.
- Move terminal failures to `detection:dead-letter:v1`.
- Expose stream lag, oldest pending age, retries, failures, and processing latency.
- Configure Redis persistence and stream-retention limits.

Verification:

- Kill a worker during processing and verify another worker completes the event.
- Stop MinIO temporarily and verify recovery after restoration.
- Confirm dead-letter entries contain the event and failure reason.

Exit criteria:

- Failures are recoverable and visible to operators.

### Phase 9: Controlled Cutover

Scope:

- Run event-driven persistence alongside existing persistence for comparison.
- Compare event counts, identities, timestamps, and media availability.
- Disable direct local snapshot/video persistence only after parity is proven.
- Preserve a configuration switch for controlled rollout and rollback.

Verification:

- No missing or duplicate logs during the comparison window.
- Existing APIs continue returning their expected media URLs and fields.
- Disabling legacy media writes does not change inference behavior.

Exit criteria:

- Redis/MinIO worker persistence is the verified primary path.

### Phase 10: Load and Failure Testing

Scope:

- Test at least 30 concurrent camera sources.
- Measure inference latency, event queue depth, Redis lag, worker throughput,
  MinIO throughput, and database write latency.
- Test Redis, MinIO, database, and worker outages independently.

Verification:

- Live/GPU processing remains within its existing latency budget.
- No unbounded memory, stream, spool, or pending-entry growth occurs.
- Worker capacity can be increased without changing publishers.

Exit criteria:

- Capacity and recovery behavior are documented with measured results.

## Reliability Rules

- Delivery is at least once; consumers must be idempotent.
- `event_id` must have a database uniqueness constraint.
- Derived MinIO object keys must be deterministic from `event_id`.
- Redis entries are acknowledged only after durable completion.
- Capture timestamps, not processing timestamps, locate recording frames.
- Source recordings must outlive the maximum event processing and retry window.
- Events contain metadata only; never include frames, videos, or base64 media.
- Any live-path queue must be bounded and expose dropped-event metrics.

## Per-Phase Change Procedure

Before each implementation phase:

1. Initialize `.workflow/phase.json` with only the intended files.
2. Confirm the branch and inspect existing worktree changes.
3. Make one small change at a time.

After every code change run:

```text
git diff --stat
git diff
python tools/check_changes.py
```

Stop immediately if `check_changes.py` reports an unexpected file. Run targeted
tests after each small change and an end-to-end test before completing a phase.

## Completion Criteria

- Human, fire/smoke, and plate events are published without blocking live inference.
- Recording manifests reliably map camera/time ranges to MinIO objects.
- Workers derive snapshots and clips and finalize database records idempotently.
- Multiple observations of one recognized person across cameras remain available.
- Unknown people are not incorrectly correlated across cameras.
- Existing APIs and inference behavior remain unchanged.
- The system is verified under at least 30 concurrent cameras and dependency outages.
