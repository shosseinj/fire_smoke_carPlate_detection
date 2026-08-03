---
description: Implements reference-counted wall/fullscreen demand sessions and API contracts
mode: subagent
---

Own only:

- `app/broadcast_gpu/demand.py`
- `app/broadcast_gpu/sessions.py`
- `app/broadcast_gpu/schemas.py`
- `app/broadcast_gpu/api.py`
- matching `tests/broadcast_gpu/` files

Implement:

- create wall session for visible camera IDs and one wall profile;
- create fullscreen session for one camera;
- heartbeat;
- explicit release;
- expiry sweep;
- reference counts;
- delayed stop cancellation when demand returns;
- idempotent duplicate release;
- bounded session count and camera count validation.

This layer must not import OpenCV, NumPy, or current AI processors. It communicates with the branch manager through an abstract protocol so tests use a fake manager.
