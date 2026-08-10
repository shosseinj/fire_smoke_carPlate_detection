# Project State

## Branch
`live`

## Phase goal
Fix and implement asynchronous saving of live frames/video through Redis to MinIO.

## Constraints
- `ai-branch` must not change.
- Existing unrelated endpoints must not change.
- Existing AI/inference behavior must not change.
- Live/GPU processing must not be blocked by MinIO/network/disk I/O.
- Diagnose the previous implementation before replacing it.

## Current status
DIAGNOSIS NOT STARTED

## Expected data flow

```text
live/GPU frame path
      |
      | fast enqueue/publish only
      v
    Redis
      |
      v
async worker / consumer
      |
      v
video/frame creation
      |
      v
    MinIO
```

## Current hypothesis
Unknown. Do not guess. Find the first broken boundary using logs and targeted checks.
