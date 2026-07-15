# Merge Audit

## Fire/smoke project findings

The uploaded fire/smoke service already accepted a list shaped like `[fN_c1, fN_c2, ...]` and used batched Ultralytics inference. However:

- The extractor rejected more than eight enabled sources.
- The detector rejected more than eight camera configurations.
- It required exact permanent camera order.
- It waited for one frame from every source before returning a batch.
- One disconnected or slow source could therefore delay the full group.
- Source extraction, task selection, detection, incident logic, and API lifecycle were tightly coupled.
- A fixed batch-8 TensorRT engine was assumed.

Merged resolution:

- The new router accepts any registered source count.
- Fire/smoke model calls are micro-batched in groups of at most eight by default.
- Per-source tracker, risk, and incident state are created dynamically.
- No exact global camera order is required inside the detector.
- The original stable tracking and severity modules were retained under `app/fire_core`.
- Fixed-batch TensorRT padding is preserved.

## Iranian plate-recognition project findings

The uploaded LPR service correctly used a YOLO plate detector followed by a local Hezar recognizer. However:

- Video processing wrote every frame to a temporary JPEG before inference.
- Recognition was called one frame at a time.
- The model path was designed around upload/folder endpoints, not a shared live frame router.
- There was no camera enable/disable or per-camera task assignment.
- Preview-window calls were mixed into server-side processing.
- Fire/smoke and plate tasks could not share the same captured frame.

Merged resolution:

- In-memory NumPy frames are processed directly.
- YOLO detection runs on frame batches.
- Plate crops from the batch are collected and sent to Hezar together when supported.
- OCR falls back safely to individual crops if the installed Hezar model does not support list input.
- No OpenCV GUI window is opened by the server.
- Plate processing is an independent worker and can run simultaneously with fire/smoke processing.

## Main architecture decision

The router uses one latest-only fair buffer per task, not one unbounded FIFO per camera.

Why:

- With 50 cameras at 25 FPS, input can reach 1,250 frames per second.
- When inference capacity is lower than capture rate, an unbounded FIFO increases latency indefinitely.
- CCTV normally benefits more from the newest frame than from processing every stale frame.
- Keeping one waiting frame per source bounds memory and latency.

The processed frame itself is never modified by the router. The same NumPy array can be referenced by multiple task queues and copied only when a model or downstream operation requires it.

## Preserved functionality

- Fire and smoke class filtering.
- Candidate confidence thresholds.
- Stable object tracking.
- Rolling severity analysis.
- Incident start, high-severity alert, and incident end events.
- Iranian plate text normalization.
- Automatic plate-class discovery.
- Local Hezar model validation.
- Plate result deduplication.
- TensorRT fixed-batch padding.

## Deliberately separated functionality

The uploaded fire project included asynchronous incident-video storage. That storage code was tightly bound to its old all-camera synchronized extractor and was not moved into the inference worker. In the merged architecture, fire incident events are returned in task results. The main project can attach its existing frame/video recorder to those events without coupling media persistence to model scheduling.

This separation prevents disk I/O and MP4 finalization from blocking inference. It is also safer for a system with 50+ streams.
