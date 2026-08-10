---
description: Diagnose the broken Redis to MinIO video save path without editing
agent: plan
---

Diagnose the existing broken asynchronous video/frame saving implementation. DO NOT edit code.

Follow the data path in order:
1. live/GPU frame production
2. enqueue/publish call
3. Redis key/list/stream/pubsub/job creation
4. worker/consumer startup and message consumption
5. frame decoding / video assembly / writer lifecycle
6. MinIO client configuration, bucket, object name, upload call
7. success/failure acknowledgement and cleanup

For each boundary, show concrete evidence from code/logs/config and mark PASS, FAIL, or UNKNOWN.
Find the FIRST failing boundary; do not jump directly to rewriting.
Also run `python tools/check_changes.py` and report any unexpected Git changes.
End with exactly one recommended next diagnostic or implementation step.
