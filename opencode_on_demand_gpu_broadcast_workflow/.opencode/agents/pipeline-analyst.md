---
description: Read-only mapper of current DeepStream, media preview, source lifecycle, and frontend broadcast paths
mode: subagent
---

Read only. Do not modify files.

Produce `.agentic/broadcast/pipeline-inventory.md` with:

- exact source construction flow for RTSP and static files;
- exact element immediately after NVIDIA decode;
- existing width/height/FPS caps and where they occur;
- current media preview and JPEG paths;
- source start, stop, reconnect, and file-EOS ownership;
- safest additive attachment point before AI resize;
- existing route/lifespan/frontend integration points;
- risks of dynamic pad attach/remove;
- files that must not be changed.

Cite file paths and line ranges. Do not propose code until inventory is complete.
