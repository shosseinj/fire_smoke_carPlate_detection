# Real live-branch Playwright test

This suite uses the real dashboard, RTCPeerConnection, MediaMTX WHEP endpoint,
and HTMLVideoElement. It does not stub fetch, WebRTC, WHEP, or `play()`.

From PowerShell (after installing the repository's approved Playwright test
dependencies and browser once):

```powershell
$env:LIVE_BRANCH_E2E='1'; $env:DASHBOARD_URL='http://127.0.0.1:9999'
npx playwright test tests/playwright/live_branch.spec.ts --headed --workers=1
```

Headless:

```powershell
$env:LIVE_BRANCH_E2E='1'; $env:DASHBOARD_URL='http://127.0.0.1:9999'
npx playwright test tests/playwright/live_branch.spec.ts --workers=1
```

The test retains trace, screenshot, video, console/page-error, failed-request,
API/WHEP request, and browser media evidence on failure. It must fail when an
acquired URL exists but the video is not ready, playing, and advancing.
