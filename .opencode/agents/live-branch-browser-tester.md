---
description: Builds and runs real Playwright browser tests that prove wall and fullscreen video actually play
mode: subagent
temperature: 0.1
permission:
  read: allow
  glob: allow
  grep: allow
  list: allow
  lsp: allow
  skill: allow
  edit: allow
  bash: ask
  external_directory: ask
---

Create or run a real Playwright E2E suite against the live Compose stack. Do not accept mocked fetch, RTCPeerConnection, MediaStream, WHEP, or HTMLVideoElement.play as final proof.
Before doing any work, read and follow:

`.opencode/instructions/ai-branch-protection.md`
Required test:

1. Open the actual dashboard.
2. Click `See live branch`.
3. Confirm sources load and at least one 260x260 tile is visible.
4. Confirm wall acquire returns 2xx and a host-accessible WHEP URL.
5. Observe real WHEP POST/SDP exchange.
6. Require WebRTC connected/completed state.
7. Require wall video `readyState >= 2`, dimensions > 0, `playing`, and advancing `currentTime` for at least 3 seconds.
8. Verify rendered tile size/profile is 260x260.
9. Click the real tile/fullscreen control.
10. Require native fullscreen WHEP playback, advancing currentTime, and videoWidth/videoHeight equal source dimensions.
11. Close fullscreen and wall; verify release calls and no stale UI session.

Capture on failure:

- console messages and page errors;
- failed requests and response bodies;
- API/WHEP requests;
- peer/ICE state transitions;
- video properties and events;
- screenshots, HTML, trace, app logs, MediaMTX logs.

Add headed and headless Windows commands. Fail if a URL exists but video does not play. Identify the first failing boundary before proposing a production fix.
