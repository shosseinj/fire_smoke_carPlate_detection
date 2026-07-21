---
description: Independently audit a recent feature or migration for missed propagation points and unreachable behavior
agent: integration-reviewer
subtask: true
---
Audit the implementation described by: $ARGUMENTS

Compare the request, repository architecture, current diff, interfaces, runtime path, tests, and evaluation records. Find missing propagation, incompatible defaults, stale contracts, unreachable features, and insufficient regression or runtime evidence.

Do not edit files. Return severity-ordered findings with paths and required corrections.
