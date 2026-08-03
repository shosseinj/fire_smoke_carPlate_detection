# On-demand GPU broadcast implementation contract

Apply these rules whenever the on-demand video-wall feature is being implemented.

## Non-negotiable isolation

1. Do not rewrite, replace, or reorganize the current AI pipeline.
2. Do not change detector input size, source FPS settings, task routing, inference queues, frame packet contracts, recording, snapshots, existing WebSocket contracts, or current JPEG fallback behavior.
3. Add new code in isolated modules under `app/broadcast_gpu/` and new tests under `tests/broadcast_gpu/`.
4. Existing files may receive only minimal additive wiring, imports, route registration, configuration defaults, and lifecycle calls.
5. All new behavior is disabled by default with `GPU_BROADCAST_ENABLED=false`.
6. With the flag disabled, baseline startup, routes, tests, logs, and pipeline topology must remain equivalent.
7. Never map the broadcast branch to Python or CPU memory. Forbidden elements and APIs include `appsink`, `videoconvert`, OpenCV, NumPy frame conversion, JPEG encoding, and `Gst.Buffer.map` for pixel access.
8. Wall scaling must use NVIDIA elements and `video/x-raw(memory:NVMM)`.
9. Fullscreen must preserve native source width, height, aspect ratio, timestamps, and source FPS.
10. No viewer means no wall/fullscreen converter, encoder, parser, or publisher branch in PLAYING state.

## Runtime model

- A permanent lightweight attachment point is allowed after NVIDIA decode and before AI resize/caps.
- Wall and fullscreen bins are created dynamically on demand.
- One shared wall publisher and one shared fullscreen publisher may exist per source/profile.
- Demand is reference-counted across browser sessions.
- Sessions use explicit release plus heartbeat expiration.
- Stop unused branches after a configurable grace period.
- Wall profiles are bounded presets: 320x180, 480x270, 640x360.
- Fullscreen has no width/height caps.
- MediaMTX WHEP is the browser playback path.
- Detection metadata remains separate from video and may be drawn in a frontend overlay canvas.

## Agent rules

- Use a separate git worktree or feature branch before editing.
- Start with inventory and baseline tests.
- Read-only analysis agents must not edit files.
- Implementers own disjoint file sets.
- Every implementation task starts with a failing test.
- The orchestrator performs final integration and conflict resolution.
- The isolation reviewer must approve before runtime validation.
- Do not claim zero-copy based only on code inspection; collect runtime evidence when DeepStream hardware is available.
