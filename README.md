# Multi-Task Video AI Router

This repository combines fire/smoke analytics and Iranian license-plate recognition within a common multi-camera processing service. It was developed to study task routing, GPU micro-batching, camera lifecycle management, and real-time result delivery when different cameras require different AI pipelines.

## Architecture

Each source can be assigned zero, one, or multiple analytics tasks. The router preserves source identity and forwards the same decoded frame to each required task worker.

```text
cameras / videos
      -> ingestion
      -> task router
        -> fire/smoke worker
        -> plate-recognition worker
      -> result store
      -> REST / WebSocket / dashboard
```

Workers use latest-frame buffering and bounded micro-batches to reduce queue growth when inference is slower than the incoming stream.

## Ingestion

Two backends are supported:

- **DeepStream** for NVIDIA/Linux deployments and GPU-oriented multi-camera ingestion;
- **OpenCV** as a development fallback for local files and small tests.

The source registry supports both local videos and RTSP streams, dynamic enable/disable state, task changes, and per-camera configuration.

## Fire/Smoke Events

Fire/smoke detections are aggregated over a configurable rolling window before incident severity is assigned. Confirmed incidents can be persisted with timestamps, counts, confidence information, and evidence snapshots.

## Plate Recognition

The plate path uses a cascade:

```text
frame
  -> vehicle detector
  -> vehicle crops
  -> plate detector
  -> plate crops
  -> OCR
```

Filtering and batching are used to avoid running later stages on frames or objects that do not satisfy earlier conditions.

## Engineering Focus

The repository demonstrates integration of heterogeneous vision tasks rather than proposing a new detection architecture. The main contribution is the runtime organization: source-aware routing, asynchronous workers, dynamic task configuration, monitoring, and result persistence.


## Goal

The service provides a shared runtime for assigning different vision tasks to multiple sources while keeping capture, batching, inference, persistence, and client delivery independently manageable.

## Installation

For local development:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python run.py
```

For the containerized stack, review the environment and mounted model paths before running `docker compose up -d --build`. DeepStream execution additionally requires Linux, compatible NVIDIA drivers, the NVIDIA Container Toolkit, and the model engines expected by the configuration.

## Working with the Repository

The FastAPI application and task workers are under `app/`; migrations are under `alembic/`; operational scripts are under `scripts/`. Start Qdrant and the configured database before enabling recognition tasks. Use `python scripts/smoke_test.py` after backend changes. The tracked `app/.env` contains credential-like configuration and must not be reused for deployment.
