# Testing and Evaluation Operating Model

The section registry is the source of truth for what is evaluated. OpenCode must populate it from actual repository evidence.

## Section command format

```json
{
  "id": "smoke-detection",
  "name": "Smoke Detection",
  "enabled": true,
  "critical": true,
  "runtime_required": true,
  "owner_paths": ["src/detectors/smoke"],
  "dependencies": ["frame-preprocessing", "model-loading"],
  "consumers": ["event-generation", "api-output"],
  "commands": [
    {
      "name": "Focused tests",
      "level": "unit",
      "command": "pytest -q tests/unit/test_smoke.py",
      "cwd": ".",
      "timeout_seconds": 300,
      "required": true
    }
  ],
  "metrics": ["inference_latency_ms", "processed_fps"],
  "acceptance": {
    "max_inference_latency_ms": null,
    "min_processed_fps": null
  }
}
```

The example command and paths must be replaced with commands that actually exist.

## Evaluation rules

- A missing required command means `NOT_CONFIGURED`.
- A nonzero required command means `FAIL`.
- An unavailable dependency means `BLOCKED`.
- A critical runtime section without representative runtime proof means `NOT_RUNTIME_VALIDATED`.
- Unknown thresholds remain null.
- Reports include stdout and stderr evidence and are saved under `.agentic/reports/`.

## Regression matrix

Use `.agentic/evaluation/regression-matrix.json` to tell the agent which connected sections must be evaluated when one section changes.
