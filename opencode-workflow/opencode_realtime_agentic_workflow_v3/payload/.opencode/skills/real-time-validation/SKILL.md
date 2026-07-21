---
name: real-time-validation
description: Validate representative real-time execution, deployment connectivity, performance, stability, failure recovery, and downstream result delivery
compatibility: opencode
metadata:
  domain: real-time-systems
  workflow: runtime-validation
---
# Real-time validation

## Representative path

Validate the real project path, for example:

```text
camera/video/RTSP -> decoding -> preprocessing -> detector/model -> tracking/OCR -> event/result -> queue/database -> API/WebSocket -> UI/downstream consumer
```

Discover the actual path rather than assuming this exact architecture.

## Runtime checks

Check startup, model/assets, required services, configuration, networking, input connection, continuous frame movement, feature invocation, downstream delivery, responsiveness, error handling, restart, disconnection and recovery, queue growth, memory growth, and resource cleanup.

## Metrics

Collect available input FPS, processed FPS, dropped frames, average/p95/max latency, detection/OCR latency, queue depth, CPU, GPU, GPU memory, system memory, error count/rate, reconnect count, and event delivery time.

Do not invent limits. Compare against verified baselines or requirements.

## Deployment gate

A successful build, image creation, container start, or health response alone does not prove the feature. Validate a representative input and expected output.

Use `NOT_RUNTIME_VALIDATED` when the environment, data, model, hardware, credentials, or service dependency prevents representative proof.
