# Decode-Tee GPU Live-Branch Final Report

Status: BLOCKED — implementation and static tests are present, but the required real browser and rebuilt GPU runtime evidence could not be collected.

## AI branch baseline and isolation

Baseline captured by `live-branch-ai-guardian` from commit `beb9b3f`: `nvurisrcbin -> identity(pacer) -> queue -> nvvideoconvert -> BGRx 640x640 caps -> appsink`, then the existing NumPy/router/task-worker path. The tee now feeds the existing queue, preserving the downstream AI chain. Existing baseline tests included `tests/test_latest_buffer.py` (3 passed); PostgreSQL-dependent inference tests were blocked by unavailable `host.docker.internal`.

## Decode tee attachment

`app/core/deepstream_ingestor.py` rejects decoded pads unless negotiated caps contain `video/x-raw` and `memory:NVMM`, then links the decoder to a tee. The tee request pad feeds the unchanged pacer/queue AI chain. Runtime negotiated caps were not captured after rebuilding the changed image: BLOCKED.

## Wall chain and 260x260 caps

The on-demand branch uses `queue -> nvvideoconvert -> video/x-raw(memory:NVMM),format=NV12,width=260,height=260 -> nvv4l2h264enc -> h264parse -> rtph264pay -> rtspclientsink`. Static fake-GStreamer caps test passed; real NVMM/NVENC negotiation: BLOCKED.

## Fullscreen chain and native caps

Fullscreen uses NVMM NV12 caps without width/height resize fields. Native negotiated dimensions/FPS/aspect ratio were not runtime-proven: BLOCKED.

## GPU-only guardrail

Static guardrail passed for the live branch. Real graph validation and GPU telemetry: BLOCKED. No appsink, OpenCV, NumPy, PIL, JPEG, software videoconvert, or software encoder is used by the live manager.

## API and frontend integration

Implemented distinct `/api/v1/live-branch/{wall|fullscreen}/{acquire|heartbeat|release}` routes, `See live branch`, 260x260 wall CSS, fullscreen switching, WHEP negotiation, heartbeat, release, unload cleanup, and visible errors. Static focused tests passed.

## Playwright browser evidence

BLOCKED. Real Playwright was attempted with `LIVE_BRANCH_E2E=1`; Chromium was not installed (`Executable doesn't exist ... ms-playwright ... chromium`). No mocked playback was accepted and no video success is claimed.

## MediaMTX and WHEP evidence

MediaMTX was running and legacy preview WHEP OPTIONS returned 204, but no live-branch path was published by the running service. Live-branch WHEP evidence: BLOCKED.

## Lifecycle and cleanup

Reference reuse, heartbeat, grace release, rollback, request-pad release, source detach, and shutdown cleanup are covered by manager unit tests. Real request-pad/MediaMTX cleanup correlation: BLOCKED.

## Static-video evidence

BLOCKED — no real static-video live source was available.

## RTSP evidence

Baseline DeepStream AI processing continued with zero failed batches, but source warnings/reconnects were present. New live-branch attach/reconnect evidence: BLOCKED.

## Automated tests

`python -m pytest tests/test_live_branch.py tests/test_live_branch_frontend_contract.py -q`: 9 passed.
`python -m pytest` broader runtime-dependent coverage: BLOCKED by PostgreSQL/runtime dependencies.
`python -m py_compile` changed Python files: passed.
`docker compose config --quiet`: passed.
`git diff --check`: passed.

## Runtime blockers

The currently running container predates the worktree changes and had `LIVE_BRANCH_ENABLED=false`; its OpenAPI returned 404 for live-branch routes. A rebuild/restart with the changed worktree and `LIVE_BRANCH_ENABLED=true`, an active source, installed Chromium, and real WHEP publication are required.

## Commits

`d3ec5d2` (`Add GPU decode tee live branch`) and `824b37a` (`Correct live branch topology and WHEP host URL`) are local implementation commits. No push or merge performed.

## Merge recommendation

HOLD/REJECTED pending real GPU/NVMM/NVENC lifecycle evidence and real Playwright wall/fullscreen playback with advancing `currentTime`.

## LIVE BRANCH ACCESS

Dashboard:
http://127.0.0.1:9999/dashboard (endpoint exists; real live-branch playback BLOCKED)

Open low-resolution wall:
Open the Dashboard and click `See live branch`; URL remains http://127.0.0.1:9999/dashboard#live-branch. Real tested wall opening: BLOCKED because Chromium was unavailable and the running container exposed no live-branch route.

Open high-resolution fullscreen:
From a live wall tile, click the tile; native fullscreen URL is browser DOM fullscreen, not a stable URL. Real tested fullscreen: BLOCKED.

Example 260x260 stream:
BLOCKED — no real WHEP URL was tested or published.

Example native stream:
BLOCKED — no real WHEP URL was tested or published.

Source API:
GET http://127.0.0.1:9999/api/v1/sources/preview-config

Wall acquire API:
POST http://127.0.0.1:9999/api/v1/live-branch/wall/acquire

Fullscreen acquire API:
POST http://127.0.0.1:9999/api/v1/live-branch/fullscreen/acquire
