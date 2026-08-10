---
description: Implement only the next approved Redis-MinIO fix with Git checks
agent: build
---

Read all project workflow instruction files first.
Before editing, run `python tools/check_changes.py`.

Implement ONLY the next confirmed fix for the first failing Redis/MinIO boundary.
Do not refactor unrelated code.
Do not change other endpoints, AI/inference code, or live/GPU behavior.
Keep persistence asynchronous and off the critical live path.

After the edit:
1. run the smallest relevant test/check
2. run `git diff --stat`
3. run `git diff`
4. run `python tools/check_changes.py`

If an unexpected/protected file changed, stop and report it.
Report: root cause, exact files changed, test result, Git protection result, and one next step.
