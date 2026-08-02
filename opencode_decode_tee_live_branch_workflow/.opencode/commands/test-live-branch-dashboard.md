---
description: Run real browser playback validation for the new live-branch dashboard wall and fullscreen
agent: live-branch-browser-tester
subtask: false
---
Run or create the real Playwright test for the new live branch.

The test must open the dashboard, click `See live branch`, prove a 260x260 wall video is playing with advancing currentTime, open fullscreen, prove native-resolution video is playing, then release sessions. Capture trace, screenshots, browser errors, WHEP requests, SDP, WebRTC states, video dimensions, and logs.

Do not mock playback and do not treat URL generation as success.
