---
description: Show current live video-save phase status without editing code
agent: plan
---

Read AGENTS.md, PROJECT_STATE.md, PROTECTED_COMPONENTS.md, DECISIONS.md and `.workflow/phase.json` if present.
Run `git status --short`, `git branch --show-current`, and `python tools/check_changes.py` if phase state exists.
Do not edit files.
Report only: current branch, current phase, allowed paths, unexpected changes, and recommended next step.
