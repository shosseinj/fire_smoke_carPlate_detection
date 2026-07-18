The active optimized fire/smoke model is:
  linux_trt10/best_nano_111_dynamic_b8_640_linux.engine

It was built for Linux, TensorRT 10.3, RTX 4090, FP16, 640x640, and dynamic
batch sizes 1-8. Keep FIRE_SMOKE_ENGINE_FIXED_BATCH=0 so partial live batches
are not padded to eight frames.

TensorRT plans are platform-specific. Rebuild the engine after changing the
GPU, TensorRT major/minor version, or operating system. The original .pt and
ONNX artifacts remain available for fallback and rebuilding.
