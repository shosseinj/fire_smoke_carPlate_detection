---
description: Configure missing checks, run project-wide section evaluation, and report critical gaps and evidence
agent: realtime-orchestrator
---
Evaluate the entire project. Additional scope: $ARGUMENTS

Load the `project-evaluation` and `real-time-validation` skills.

Inspect `.agentic/evaluation/section-registry.json`. Add or correct evaluation commands based on the actual repository. Do not invent tools or thresholds.

Run:

```text
python scripts/agentic_eval.py --all
```

Also perform representative runtime and deployment validation when supported. Review the generated JSON and Markdown reports. Summarize section statuses, critical failures, blocked and unconfigured checks, regression coverage, runtime evidence, performance comparison, risks, and prioritized fixes.
