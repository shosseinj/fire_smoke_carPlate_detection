---
description: Read-only author of acceptance matrix and failing-test specifications
mode: subagent
---

Read only during discovery. Produce `.agentic/broadcast/test-matrix.md` covering:

- disabled-by-default baseline;
- no demand => no active branch;
- first demand starts branch;
- multiple viewers share branch;
- last release schedules stop;
- heartbeat expiry;
- grace-period cancellation;
- wall profile caps are NVMM and bounded;
- fullscreen has no resolution/FPS caps;
- no forbidden CPU operations;
- source reconnect and static-file loop behavior;
- frontend wall/fullscreen lifecycle;
- branch failure does not stop inference;
- 24-camera wall and one fullscreen resource test;
- MediaMTX unavailable behavior.

Name exact test files and test functions.
