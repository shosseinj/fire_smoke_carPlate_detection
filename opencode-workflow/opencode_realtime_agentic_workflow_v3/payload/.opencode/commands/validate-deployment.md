---
description: Validate deployed startup, health, dependencies, stream flow, outputs, performance, recovery, logs, and rollback
agent: realtime-orchestrator
---
Validate deployment and runtime for: $ARGUMENTS

Load the `real-time-validation` and `project-evaluation` skills.

Use the project's verified deployment commands. Check configuration and secrets presence without revealing values, model/assets, volumes, networking, database/queue connectivity, health endpoints, API/WebSocket availability, representative stream processing, downstream result delivery, logs, resources, recovery, restart behavior, and rollback.

A successful image build or container start is not sufficient. Report NOT_RUNTIME_VALIDATED when representative flow cannot be executed.
