# OpenCode Multi-Agent Workflow: Decode-Tee GPU Live Branch

This workflow adds a **new on-demand GPU live branch after NVIDIA decode** while keeping the existing AI branch unchanged.

Target behavior:

- Existing AI branch remains functionally and structurally unchanged.
- A decode-output `tee` provides a separate live branch.
- Video wall output is **260x260**, GPU-resized, NVMM-only.
- Fullscreen output is the source's **original resolution and FPS**, with no resize.
- The new live branch uses a **different frontend/API/stream URL namespace** from the existing preview path.
- Dashboard gains a **See live branch** button.
- Wall/fullscreen streams start on demand and stop after release/timeout.
- Real browser playback is validated using Playwright; URL generation alone is not accepted.

## Install

From the target project root:

```powershell
.\opencode_decode_tee_live_branch_workflow\INSTALL_IN_EXISTING_PROJECT.ps1
```

The installer merges `.opencode` and `.agentic` files without deleting existing project configuration.

## Run

```powershell
opencode --auto
```

Then run:

```text
/implement-decode-tee-live-branch
```

For a final independent audit:

```text
/audit-decode-tee-live-branch
```

For browser playback testing:

```text
/test-live-branch-dashboard
```

## Required final output

The orchestrator must print real tested values for:

- dashboard URL;
- low-resolution 260x260 video-wall URL;
- high-resolution fullscreen access flow or URL;
- example low-resolution WHEP/WebRTC URL;
- example native-resolution WHEP/WebRTC URL;
- source-listing and session API URLs.

It must not guess or report generated-but-unplayed URLs as working.
