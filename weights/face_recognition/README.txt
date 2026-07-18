The active face pipeline uses Linux TensorRT 10.3 engines built inside the
target DeepStream container:

  linux_trt10/yolo26s-pose_batch8_linux.engine
  linux_trt10/yolov8n-face_batch8_linux.engine
  linux_trt10/arcface_fp16_dynamic_b64_linux.engine

The detector plans are fixed at batch 8 and 640x640. ArcFace accepts dynamic
batches from 1 through 64 and is optimized for batch 8.

Override paths with FACE_HUMAN_MODEL, FACE_DETECTOR_MODEL, and
FACE_EMBEDDING_MODEL. Do not select the older top-level engine files: they were
built for a different platform and cannot be deserialized by Linux TensorRT.
