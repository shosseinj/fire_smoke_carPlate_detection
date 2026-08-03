# OpenCode Multi-Agent Workflow: On-Demand GPU Video Wall

This package is an **agentic implementation workflow**, not a replacement project.
Copy its contents into the root of `fire_smoke_carPlate_detection-main`, then run the included OpenCode command.

## Goal

Add a completely separate, on-demand GPU broadcast branch:

- The current AI, inference, recording, API, and JPEG paths remain behaviorally unchanged.
- With no viewers, no wall/fullscreen encoder branch runs.
- Opening the video-wall creates shared low-resolution GPU streams for visible cameras.
- Clicking a camera creates a shared original-resolution fullscreen stream.
- Closing/disconnecting viewers releases demand and stops unused branches after a grace period.
- Broadcast frames remain in NVMM/GPU memory. No Python frame mapping, NumPy, OpenCV, JPEG, or CPU scaling is permitted.

## Install

PowerShell:

```powershell
Expand-Archive .\opencode_on_demand_gpu_broadcast_workflow.zip -DestinationPath .\workflow_tmp
& .\workflow_tmp\opencode_on_demand_gpu_broadcast_workflow\INSTALL_IN_EXISTING_PROJECT.ps1
```

The PowerShell installer merges the workflow into existing directories without deleting
or replacing the existing `.opencode` tree. It validates with
`C:/Users/jafari.h/Desktop/ai_project/.venv/Scripts/python.exe`.

Bash:

```bash
unzip opencode_on_demand_gpu_broadcast_workflow.zip
cp -a opencode_on_demand_gpu_broadcast_workflow/. .
python3 scripts/validate_broadcast_workflow.py
```

## Run with OpenCode

From the project root:

```text
/implement-on-demand-gpu-broadcast
```

If slash commands are not available in your OpenCode build, paste the contents of:

```text
.opencode/commands/implement-on-demand-gpu-broadcast.md
```

into a new OpenCode session.

## Multi-agent layout

- `broadcast-orchestrator`: owns scope, worktree, task board, integration, and final evidence.
- `pipeline-analyst`: maps current GStreamer/DeepStream source construction without editing.
- `broadcast-architect`: produces exact dynamic-branch interfaces and lifecycle design.
- `gpu-pipeline-implementer`: implements isolated GStreamer branch modules.
- `session-api-implementer`: implements demand/ref-count/heartbeat API modules.
- `frontend-stream-implementer`: implements wall and fullscreen WHEP playback.
- `broadcast-test-designer`: writes contract, lifecycle, and static guardrail tests.
- `isolation-reviewer`: rejects any modification that changes the current AI path.
- `runtime-validator`: validates NVMM/NVENC, FPS, resolution, teardown, and isolation.

Agents may analyze in parallel, but code integration is serialized by the orchestrator.

## Important constraint

“Do not change the current project” is implemented as **no behavioral change to existing paths**. A minimal additive hook may be required at the decoded-NVMM source boundary to attach a dormant `tee` or request pad. It must be behind `GPU_BROADCAST_ENABLED=false`, preserve the original AI link, and pass baseline tests when disabled.
