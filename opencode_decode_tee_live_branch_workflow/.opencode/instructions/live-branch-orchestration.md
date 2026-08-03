# Decode-Tee GPU Live-Branch Orchestration

For tasks involving the new live branch:

1. Preserve the existing AI branch. Do not refactor, reorder, resize, re-cap, replace, or reroute it unless a failing compatibility test proves an unavoidable defect and the user approves.
2. Establish an AI-branch baseline before edits: element chain, caps, FPS policy, source identifiers, inference outputs, and focused tests.
3. Attach the new branch at the NVIDIA decoder's NVMM output through a `tee` or equivalent safe GPU-native split.
4. Keep the live path GPU-only: no appsink, OpenCV, NumPy pixels, PIL, JPEG frames, software videoconvert/videoscale, software encoder, buffer mapping, or host-memory copies.
5. Video-wall profile is exactly 260x260 unless the user changes the requirement.
6. Fullscreen profile is native source dimensions and source FPS, without resize caps.
7. Use a new route/path namespace distinct from the current preview or legacy broadcast path.
8. Add a dashboard button whose visible label is `See live branch` unless the existing UI localization system requires a translated equivalent plus the English semantic identifier.
9. Branches must be on demand, reference-counted, heartbeat-cleaned, and safely removed.
10. A generated WHEP URL is not proof. Browser playback must reach connected/playing with advancing `currentTime`.
11. Use real project APIs and runtime services. Do not make production success depend on mocks.
12. Never push or merge. Make local logical commits only when validation evidence exists.
