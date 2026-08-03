---
description: Implements the new dashboard live-branch button, 260x260 wall, native fullscreen, APIs, WHEP client, heartbeat, and release lifecycle
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
  external_directory: deny
---

Implement only the API/frontend integration assigned by the orchestrator.
Before doing any work, read and follow:

`.opencode/instructions/ai-branch-protection.md`
Required UX:

- Add a clearly visible dashboard button labeled `See live branch`.
- The button opens a new live-branch view/section using a different route or state from the old preview.
- Low-resolution wall tiles render at 260x260 and preserve correct camera identity.
- Clicking a tile opens native-resolution fullscreen for that source.
- Use HTML `<video autoplay muted playsinline>` and real WHEP/WebRTC.
- Do not use JPEG, canvas frame polling, or legacy WebSocket images for the new branch.
- Display visible errors for source load, acquire, WHEP negotiation, disconnected stream, and playback rejection.
- Implement heartbeat, explicit release, unload cleanup, fullscreen camera switching, and stale-session handling.

API behavior:

- New endpoints/path namespace must be distinct from the current preview path.
- Routes remain present when disabled and return a clear disabled response.
- Acquire responses include session ID, source ID, profile, stream path, and browser-safe WHEP URL.
- Do not return Docker-only hostnames to the browser.

Add contract/component tests, but do not mock browser playback in the final E2E proof. Coordinate exact schemas with backend implementation.
