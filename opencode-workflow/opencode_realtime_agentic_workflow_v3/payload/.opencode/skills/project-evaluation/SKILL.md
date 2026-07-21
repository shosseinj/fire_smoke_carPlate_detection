---
name: project-evaluation
description: Build and run a machine-readable section evaluation system for static, unit, integration, regression, end-to-end, runtime, performance, and deployment checks
compatibility: opencode
metadata:
  domain: software-quality
  workflow: section-evaluation
---
# Project evaluation

## Section registry

Maintain `.agentic/evaluation/section-registry.json`. Register only sections that actually exist. Typical examples include startup, configuration, video/RTSP ingestion, preprocessing, detectors, tracking, OCR, event generation, storage, REST, WebSocket, workers, queues, UI, authentication, logging, metrics, Docker, GPU/model loading, health, and recovery.

Each section should include owner paths, dependencies, consumers, criticality, evaluation commands, metrics, and acceptance criteria.

## Evaluation levels

Use the levels supported by the project:

- static validation;
- unit tests;
- component tests;
- integration tests;
- end-to-end tests;
- regression tests;
- representative real-time runtime tests;
- performance tests;
- deployment tests.

## Status model

Use `PASS`, `PASS_WITH_WARNINGS`, `FAIL`, `BLOCKED`, `NOT_CONFIGURED`, `NOT_TESTED`, `NOT_APPLICABLE`, or `NOT_RUNTIME_VALIDATED`.

Critical sections prevent overall PASS when failed, blocked, unconfigured, or not runtime validated where runtime proof is required.

## Threshold integrity

Do not invent thresholds. Use explicit requirements, existing tests, a verified baseline, production evidence supplied by the user, or explicit user direction. Record unknown values as null.

## Runner

Use `python scripts/agentic_eval.py --all` or `--section <id>`. Preserve generated evidence under `.agentic/reports/`.
