#!/usr/bin/env python3
"""Validate that the OpenCode real-time Excel-catalog workflow is installed correctly."""
from __future__ import annotations

import json
import re
from pathlib import Path

REQUIRED = [
    "AGENTS.md",
    ".opencode/agents/realtime-orchestrator.md",
    ".opencode/agents/impact-analyzer.md",
    ".opencode/agents/scope-controller.md",
    ".opencode/agents/catalog-option-analyzer.md",
    ".opencode/commands/bootstrap-agentic.md",
    ".opencode/commands/migrate-legacy-options.md",
    ".opencode/commands/migrate-selected-options.md",
    ".opencode/commands/migration-status.md",
    ".opencode/commands/implement-feature.md",
    ".opencode/commands/evaluate-project.md",
    ".opencode/skills/change-propagation/SKILL.md",
    ".opencode/skills/catalog-option-migration/SKILL.md",
    ".opencode/skills/project-evaluation/SKILL.md",
    ".opencode/skills/real-time-validation/SKILL.md",
    ".agentic/evaluation/section-registry.json",
    ".agentic/migration/legacy-option-map.json",
    ".agentic/migration/legacy-endpoint-catalog.json",
    ".agentic/migration/current-scope.json",
    ".agentic/migration/migration-history.json",
    ".agentic/migration/source/previous-project-options.xlsx",
    "scripts/agentic_eval.py",
    "scripts/import_option_catalog.py",
]


def check_json(path: Path, errors: list[str]) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception as exc:
        errors.append(f"Invalid JSON {path}: {exc}")
        return None


def check_frontmatter(path: Path, errors: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        errors.append(f"Missing YAML frontmatter: {path}")


def main() -> int:
    root = Path.cwd().resolve()
    errors: list[str] = []
    warnings: list[str] = []

    for rel in REQUIRED:
        if not (root / rel).exists():
            errors.append(f"Missing required file: {rel}")

    for path in (root / ".agentic").rglob("*.json") if (root / ".agentic").exists() else []:
        check_json(path, errors)

    opencode_json = root / "opencode.json"
    if opencode_json.exists():
        config = check_json(opencode_json, errors)
        if isinstance(config, dict):
            refs = config.get("references")
            if isinstance(refs, dict) and "legacy-project" in refs:
                warnings.append("Old legacy-project reference remains in opencode.json; this Excel-catalog workflow does not use it")
    elif (root / "opencode.jsonc").exists():
        if (root / "opencode.agentic.snippet.json").exists():
            warnings.append("opencode.jsonc requires manual merge of opencode.agentic.snippet.json")
        else:
            warnings.append("opencode.jsonc exists; verify project-orchestration instruction manually")
    else:
        warnings.append("No opencode.json or opencode.jsonc found")

    for directory in [root / ".opencode" / "agents", root / ".opencode" / "commands"]:
        if directory.exists():
            for path in directory.glob("*.md"):
                check_frontmatter(path, errors)

    skills_dir = root / ".opencode" / "skills"
    if skills_dir.exists():
        for path in skills_dir.glob("*/SKILL.md"):
            check_frontmatter(path, errors)
            text = path.read_text(encoding="utf-8")
            match = re.search(r"^name:\s*([^\n]+)$", text, re.MULTILINE)
            if not match or match.group(1).strip() != path.parent.name:
                errors.append(f"Skill name must match directory: {path}")

    catalog_path = root / ".agentic" / "migration" / "legacy-endpoint-catalog.json"
    if catalog_path.exists():
        catalog = check_json(catalog_path, errors)
        if catalog:
            items = catalog.get("items") or catalog.get("endpoints") or []
            ids = [item.get("id") for item in items if isinstance(item, dict)]
            if len(ids) != len(set(ids)):
                errors.append("Duplicate item IDs in legacy-endpoint-catalog.json")
            expected = catalog.get("source", {}).get("item_count")
            if expected is None:
                expected = catalog.get("source", {}).get("endpoint_count")
            if expected is not None and expected != len(items):
                errors.append("Item count does not match legacy-endpoint-catalog.json source metadata")
            if catalog.get("migration_mode") != "PROMPT_SCOPED_INCREMENTAL_CATALOG_ONLY":
                warnings.append("Catalog migration mode is not the Excel-catalog mode; reimport the options workbook")
            if not items:
                warnings.append("Options catalog is empty")

    registry = root / ".agentic" / "evaluation" / "section-registry.json"
    if registry.exists():
        data = check_json(registry, errors)
        if data:
            sections = data.get("sections", [])
            ids = [s.get("id") for s in sections if isinstance(s, dict)]
            if len(ids) != len(set(ids)):
                errors.append("Duplicate section IDs in section-registry.json")
            if not data.get("generated_from_verified_repository_inspection"):
                warnings.append("Section registry is still a bootstrap template; run /bootstrap-agentic")

    print("OpenCode Excel-catalog agentic workflow validation")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print(f"FAILED with {len(errors)} error(s) and {len(warnings)} warning(s)")
        return 1
    print(f"PASS with {len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
