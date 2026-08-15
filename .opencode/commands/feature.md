---
description: Safely implement a new feature with minimal regression risk
agent: safe-lead
---
Implement this feature conservatively: $ARGUMENTS

Required workflow:
1. Use @explorer to inspect the existing implementation and determine the minimum safe change surface.
2. Summarize the intended files/behavior to change before implementation.
3. Use @implementer for the smallest backward-compatible patch.
4. If any required modification affects another working section, shared behavior, public API, dependency, schema, or default, stop before that change, explain why it is necessary, and wait for my explicit approval.
5. Use @reviewer after implementation.
6. Run targeted tests/validation only. Do not repair unrelated failures.
7. Finish with changed files, validation results, and remaining risks.
