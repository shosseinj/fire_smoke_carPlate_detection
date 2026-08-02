# On-Demand GPU Broadcast Design

## Objective

Add an optional, isolated live-video subsystem that publishes low-resolution video-wall tiles and native-resolution fullscreen streams only while frontend demand exists.

## Boundaries

The existing AI pipeline remains the owner of source decode, inference delivery, detections, recording, and current fallback broadcasting. The new subsystem attaches after NVIDIA decode and before any AI resize/caps. Its bins have independent queues and failure handling.

## Data flow

```text
RTSP / static file
  -> existing NVIDIA decode
  -> lightweight attachment point
       -> existing AI branch (unchanged)
       -> dynamic wall bin (only with wall demand)
       -> dynamic fullscreen bin (only with fullscreen demand)
```

Wall bin:

```text
queue -> nvvideoconvert -> NVMM profile caps -> nvv4l2h264enc -> h264parse -> rtspclientsink -> MediaMTX -> WHEP
```

Fullscreen bin:

```text
queue -> NVIDIA conversion only if encoder requires it -> NVMM caps without size/FPS -> nvv4l2h264enc -> h264parse -> rtspclientsink -> MediaMTX -> WHEP
```

## Demand model

A browser wall session declares visible cameras and one profile. A fullscreen session declares one camera. The registry reference-counts `(camera, mode, profile)` and emits start on transition `0 -> 1`, and delayed stop on transition `1 -> 0`. A new demand during the grace period cancels stop.

## Frontend behavior

Opening the wall creates low-resolution sessions only for visible tiles. Clicking a tile creates a native fullscreen session. The browser releases fullscreen on close and wall sessions on page exit. Heartbeats recover from unclean disconnects.

## Failure isolation

Broadcast branch errors are logged and torn down without propagating EOS/ERROR to the existing AI branch. Queues are bounded. The branch manager performs pad blocking and idempotent detach. MediaMTX outage must not stop source decode or inference.

## Feature flag and rollback

`GPU_BROADCAST_ENABLED=false` is the default. Rollback is setting the flag false and restarting. New modules may remain installed without execution.
