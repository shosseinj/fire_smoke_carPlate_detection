# Decode-Tee GPU Live-Branch Final Report

Status: BLOCKED — implementation and static tests are present, but the required real browser and rebuilt GPU runtime evidence could not be collected.

## AI branch baseline and isolation

Baseline captured by `live-branch-ai-guardian` from commit `beb9b3f`: `nvurisrcbin -> identity(pacer) -> queue -> nvvideoconvert -> BGRx 640x640 caps -> appsink`, then the existing NumPy/router/task-worker path. The tee now feeds the existing queue, preserving the downstream AI chain. Existing baseline tests included `tests/test_latest_buffer.py` (3 passed); PostgreSQL-dependent inference tests were blocked by unavailable `host.docker.internal`.

## Decode tee attachment

`app/core/deepstream_ingestor.py` rejects decoded pads unless negotiated caps contain `video/x-raw` and `memory:NVMM`, then links the decoder to a tee. The tee request pad feeds the unchanged pacer/queue AI chain. Runtime negotiated caps were not captured after rebuilding the changed image: BLOCKED.

## Wall chain and 320x320 caps

The on-demand branch uses `queue -> nvvideoconvert -> video/x-raw(memory:NVMM),format=NV12,width=320,height=320 -> nvv4l2h264enc -> h264parse -> rtspclientsink`. Static fake-GStreamer caps test passed; real NVMM/NVENC negotiation: BLOCKED.

## Fullscreen chain and native caps

Fullscreen uses NVMM NV12 caps without width/height resize fields. Native negotiated dimensions/FPS/aspect ratio were not runtime-proven: BLOCKED.

## GPU-only guardrail

Static guardrail passed for the live branch. Real graph validation and GPU telemetry: BLOCKED. No appsink, OpenCV, NumPy, PIL, JPEG, software videoconvert, or software encoder is used by the live manager.

## API and frontend integration

Implemented distinct `/api/v1/live-branch/{wall|fullscreen}/{acquire|heartbeat|release}` routes, `See live branch`, 320x320 wall CSS, fullscreen switching, WHEP negotiation, heartbeat, release, unload cleanup, and visible FPS/resolution overlays. Static focused tests passed.

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

## Active source discovery follow-up

The real `GET /api/v1/broadcast-gpu/sources` request initially returned 404 because no broadcast-gpu router existed and the dashboard was using the preview source snapshot instead. The fix adds a registry-backed route and makes the dashboard fetch it on synchronization. The route reads `runtime.registry.list()` per request, returns only enabled records with `source_id`, `source_uri`, `enabled`, `active`, and `source_type`, and therefore refreshes immediately after source registration without a restart.

The production `POST /api/v1/sources` path now accepts an existing local static video when `source_type=static_video` and the path exists. `file:///workspace/data/1.mp4` was confirmed inside the container and registered as source ID `9`. File-URI resolution was added to both video ingestors. The source discovery endpoint returned 9 sources including that static record after registration.

Manual wall acquire for the static source returned 503 because the decoder had not produced a confirmed NVMM tee attachment. This is the next failing boundary; no GPU/AI path success is claimed.

## NVMM tee and fullscreen route follow-up

The decoder pad callback now defers live registration until fixed negotiated NVMM caps are available, while linking the pre-existing AI tee branch unchanged. Runtime source 9 subsequently acquired the wall profile with HTTP 200 and MediaMTX published `live-branch/wall/1f17fc4f663d61bd827c3ee0`; release and grace cleanup removed the branch. The fullscreen profile route is the registered profile route `/api/v1/live-branch/{profile}/acquire`; after rebuilding `video-ai-router`, `POST /api/v1/live-branch/fullscreen/acquire` for source 9 returned HTTP 200 with `live-branch/fullscreen/1f17fc4f663d61bd827c3ee0` and WHEP port 8789. Browser playback remains separately unvalidated.

## WHEP publication race follow-up

MediaMTX could return `404 no stream is available` when the browser posted WHEP immediately after acquire, before `rtspclientsink` finished publishing. Acquire now waits briefly for asynchronous publication and the dashboard retries transient WHEP 404 responses. Runtime source 3 validation returned acquire HTTP 200, WHEP `OPTIONS` HTTP 204 for `live-branch/wall/7dbb9cc0448f2fa13ff76c78/whep`, and release HTTP 200.

## Wall-to-fullscreen playback follow-up

The dashboard wall is exactly 320x320. Fullscreen acquires the separate native profile and replaces the wall session. Browser heartbeat `409` responses after a runtime restart now trigger wall session reacquisition instead of leaving stale dead tiles. The dashboard displays measured FPS and video resolution above each live video. A fresh source-3 wall validation returned HTTP 200, 320x320 dimensions, WHEP OPTIONS 204, and release HTTP 200. Actual advancing browser playback remains dependent on the real Playwright browser environment.

Fullscreen permission failures were caused by calling `requestFullscreen()` after awaited network/release operations, which loses browser user activation. The dashboard now requests fullscreen synchronously from the tile click and performs branch release/acquire afterward.

## Commits

`d3ec5d2`, `824b37a`, and the source-discovery follow-up are local implementation commits. No push or merge performed.

## Merge recommendation

HOLD/REJECTED pending real GPU/NVMM/NVENC lifecycle evidence and real Playwright wall/fullscreen playback with advancing `currentTime`.

## LIVE BRANCH ACCESS

Dashboard:
http://127.0.0.1:9999/dashboard (endpoint exists; real live-branch playback BLOCKED)

Open low-resolution wall:
Open the Dashboard and click `See live branch`; URL remains http://127.0.0.1:9999/dashboard#live-branch. Real tested wall opening: BLOCKED because Chromium was unavailable and the running container exposed no live-branch route.

Open high-resolution fullscreen:
From a live wall tile, click the tile; native fullscreen URL is browser DOM fullscreen, not a stable URL. Real tested fullscreen: BLOCKED.

Example 320x320 stream:
BLOCKED — no real WHEP URL was tested or published.

Example native stream:
BLOCKED — no real WHEP URL was tested or published.

Source API:
GET http://127.0.0.1:9999/api/v1/sources/preview-config

Wall acquire API:
POST http://127.0.0.1:9999/api/v1/live-branch/wall/acquire

Fullscreen acquire API:
POST http://127.0.0.1:9999/api/v1/live-branch/fullscreen/acquire
