---
description: Review, safely commit, and optionally push the current Git changes.
agent: build
---

Follow the `change-manager` skill for this request.

Inspect the current Git worktree and report the changed files, diff summary, and any possible secrets first. Do not commit or push without explicit user confirmation. Preserve unrelated user changes, never use destructive Git commands or force-push, and use the user's arguments as additional context:

$ARGUMENTS
