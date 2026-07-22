1. Frame Submission Rate Limiting
   The DeepStream ingestor limits frame submission to target_fps (default 25 FPS):

- deepstream_ingestor.py:671: submit_interval = 1.0 / max(self.target_fps, self.preview_fps)
- deepstream_ingestor.py:690: Loop waits min(0.05, submit_interval) between iterations

2. Frame Dropping in DeepStream

- deepstream_ingestor.py:291-293: Frames arriving faster than 1.0 / delivery_target_fps are dropped
- LatestPerSourceBuffer: Only keeps the newest frame per source, dropping stale ones

3. Batch Processing Delays

- Workers wait up to max_wait_ms (default 25ms) to fill a batch
- If frames arrive slowly, batches are processed incomplete
- Each worker processes batches sequentially

4. Model Inference Time

- Fire/Smoke, Plate, and Face recognition each run separate models
- TensorRT engines have fixed batch sizes (e.g., batch 8)
- Inference time per batch adds latency
  How to Increase FPS
  Quick Configuration Changes

1. Increase target FPS:

# In .env or environment

VIDEO_INGEST_FPS=30
VIDEO_PREVIEW_FPS=30 2. Reduce batch wait time:
FIRE_SMOKE_MAX_WAIT_MS=10
PLATE_MAX_WAIT_MS=10
FACE_MAX_WAIT_MS=10 3. Increase batch sizes (if GPU memory allows):
FIRE_SMOKE_BATCH_SIZE=16
PLATE_BATCH_SIZE=16
FACE_BATCH_SIZE=16
Architecture-Level Optimizations

1. Reduce frame processing per source:

- Skip frames for sources without active tasks
- Process every Nth frame instead of every frame

2. Optimize DeepStream pipeline:

- Reduce max-size-buffers in queue (currently 1)
- Enable hardware-accelerated scaling
- Use nvvideoconvert for all transformations

3. Model optimization:

- Use TensorRT engines with dynamic batching
- Enable FP16 precision (PLATE_USE_FP16=True)
- Reduce input resolution (FIRE_SMOKE_IMGSZ=480)

4. Parallel processing:

- Run multiple worker threads per task type
- Use GPU streams for concurrent inference
  Monitoring FPS
  Check current FPS via health endpoint:
  curl -fsS http://127.0.0.1:9999/health
  curl -fsS http://127.0.0.1:9999/api/v1/diagnostics/overview
