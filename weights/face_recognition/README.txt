The active detector models currently use PyTorch:

  yolo26s-pose_batch8.pt           human pose detector
  yolov8n-face_batch8.pt           face detector with 5 landmarks

The previous project has no ArcFace PT checkpoint, so embeddings temporarily use:

  arcface_fp16.onnx                512-dimensional ArcFace

Override filenames with FACE_HUMAN_MODEL, FACE_DETECTOR_MODEL, and
FACE_EMBEDDING_MODEL. The Windows-created engines in the old project cannot be
loaded by Linux TensorRT because plan files are platform-specific. Rebuild any
future .engine files inside the target Linux container before selecting them.
