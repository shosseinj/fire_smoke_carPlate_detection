---
description: Read-only designer for demand-driven NVMM wall/fullscreen branches
mode: subagent
---

Read only. Do not edit application code.

Using the project inventory, define exact interfaces for:

- `BroadcastDemandRegistry`
- `BroadcastSessionService`
- `GpuBroadcastBranchManager`
- `WallBranchSpec`
- `FullscreenBranchSpec`
- MediaMTX path generation
- frontend session API payloads

The design must guarantee:

- zero active broadcast branch with zero demand;
- one shared wall stream per source/profile;
- one shared fullscreen stream per source;
- native fullscreen resolution/FPS;
- NVMM-only pixel path;
- independent failure boundaries from AI;
- idempotent attach/detach;
- heartbeat expiry and stop grace period.

Write `.agentic/broadcast/interface-contract.md` with concrete Python signatures and state transitions.
