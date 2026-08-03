---
description: Implements on-demand video wall and fullscreen WHEP playback without disturbing JPEG fallback
mode: subagent
---

Own new frontend files for GPU streaming and the smallest additive dashboard wiring approved by the orchestrator.

Requirements:

- Preserve existing JPEG fallback and current dashboard behavior when feature flag is false.
- On wall page activation, request sessions only for visible camera tiles.
- Use returned WHEP URLs in `<video>` elements.
- Clicking a tile requests native fullscreen and pauses/releases that browser's selected wall demand when appropriate.
- Closing fullscreen releases it and restores wall demand.
- Send heartbeat while active.
- Release on navigation, visibility changes, and unload where possible; heartbeat expiry is the safety net.
- Never draw video pixels to canvas. A transparent canvas may display metadata overlays only.
- Handle reconnect without creating duplicate sessions.

Write browser-contract tests before implementation.
