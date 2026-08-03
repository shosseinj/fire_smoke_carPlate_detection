# Decode-Tee GPU Live-Branch Specification

## Scope

Implement a new live-view path from the existing NVIDIA decode output to the frontend.

## Immutable AI branch

The existing AI branch must retain its current:

- elements and ordering after the split;
- caps and dimensions;
- FPS/timestamp policy;
- batching/mux/inference behavior;
- callbacks and outputs;
- reconnect and error behavior.

## New branch

Attachment point: first verified decoded NVMM output before any AI-specific resize/caps conversion.

### Wall profile

- exact output: 260x260;
- GPU resize only;
- NVMM maintained through encoder input;
- source identity preserved;
- on-demand and shared.

### Fullscreen profile

- original negotiated width and height;
- original FPS and aspect ratio;
- no resize;
- on-demand and shared.

### Delivery

- separate route/path namespace from current preview;
- MediaMTX WebRTC/WHEP;
- host-accessible URLs returned to browser;
- internal service address used for publishing.

### Dashboard

- add `See live branch` button;
- show low-resolution wall;
- click tile to open native fullscreen;
- visible errors and clean lifecycle.

## Acceptance gates

1. AI guardian passes before/after comparison.
2. Static guardrail finds no forbidden CPU pixel operation in the live path.
3. Negotiated caps prove NVMM at critical points.
4. Real Playwright wall playback passes.
5. Real Playwright fullscreen playback passes at source dimensions.
6. Lifecycle and cleanup tests pass.
7. Runtime validator correlates branch demand with MediaMTX and NVENC.
8. Final reviewer approves.
