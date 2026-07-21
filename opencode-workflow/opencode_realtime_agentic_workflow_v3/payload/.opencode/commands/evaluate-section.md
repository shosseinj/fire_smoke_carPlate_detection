---
description: Evaluate one project section and its dependency and consumer regression surface
agent: realtime-orchestrator
---
Evaluate section: $ARGUMENTS

Load the `project-evaluation` skill and `real-time-validation` when relevant.

Find the section in `.agentic/evaluation/section-registry.json`, verify its commands, identify its dependencies and consumers, and run its checks plus the required regression surface.

Use:

```text
python scripts/agentic_eval.py --section "$ARGUMENTS"
```

Report evidence, metrics, thresholds, failures, blocked checks, downstream effects, and whether runtime validation was representative.
