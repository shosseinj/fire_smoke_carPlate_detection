#!/usr/bin/env python3
from pathlib import Path
import json
import sys

root = Path(__file__).resolve().parent
required = [
    "payload/AGENTS.md",
    "payload/.opencode/agents/realtime-orchestrator.md",
    "payload/.opencode/agents/catalog-option-analyzer.md",
    "payload/.opencode/commands/migrate-legacy-options.md",
    "payload/.opencode/commands/migrate-selected-options.md",
    "payload/.opencode/commands/migration-status.md",
    "payload/.opencode/agents/scope-controller.md",
    "payload/.opencode/skills/catalog-option-migration/SKILL.md",
    "payload/.opencode/skills/project-evaluation/SKILL.md",
    "payload/.agentic/evaluation/section-registry.json",
    "payload/.agentic/migration/legacy-endpoint-catalog.json",
    "payload/.agentic/migration/current-scope.json",
    "payload/.agentic/migration/migration-history.json",
    "payload/scripts/agentic_eval.py",
    "payload/scripts/import_option_catalog.py",
    "installer/install.py",
    "installer/catalog_import.py",
    "reference/app_names_and_urls.xlsx",
    "RUN_PROMPT.md",
    "STEP_BY_STEP_PROMPT.md",
]
errors = []
for rel in required:
    if not (root / rel).exists():
        errors.append("missing " + rel)
for path in (root / "payload").rglob("*.json"):
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid json {path}: {exc}")
if errors:
    print("\n".join(errors))
    sys.exit(1)
print("Package validation PASS")
