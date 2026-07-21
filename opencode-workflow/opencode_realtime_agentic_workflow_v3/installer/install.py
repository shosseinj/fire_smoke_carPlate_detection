#!/usr/bin/env python3
"""Install the OpenCode real-time workflow using an Excel option catalog."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from catalog_import import build_catalog

VERSION = "1.2.0"
AGENTS_START = "<!-- OPENCODE-REALTIME-WORKFLOW:START -->"
AGENTS_END = "<!-- OPENCODE-REALTIME-WORKFLOW:END -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="Target new-project directory")
    parser.add_argument(
        "--options-excel",
        help="Excel file containing previous-project options/endpoints. Defaults to the workbook bundled with this package.",
    )
    parser.add_argument("--sheet", help="Optional worksheet name in the Excel file")
    parser.add_argument("--force", action="store_true", help="Replace conflicting workflow-owned files after backup")
    parser.add_argument(
        "--replace-catalog-progress",
        action="store_true",
        help="Reset matching catalog-item progress instead of preserving it during import",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_payload_tree(payload: Path, target: Path, force: bool, backup_root: Path) -> tuple[list[str], list[str], list[str]]:
    copied: list[str] = []
    skipped: list[str] = []
    replaced: list[str] = []
    special = {
        "AGENTS.md",
        "opencode.agentic.template.json",
        ".agentic/migration/legacy-endpoint-catalog.json",
    }
    for source in sorted(payload.rglob("*")):
        if not source.is_file():
            continue
        rel = source.relative_to(payload)
        if rel.as_posix() in special:
            continue
        destination = target / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() == source.read_bytes():
                skipped.append(f"{rel} (already current)")
                continue
            if not force:
                skipped.append(f"{rel} (conflict preserved; rerun with --force)")
                continue
            backup_destination = backup_root / rel
            backup_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup_destination)
            shutil.copy2(source, destination)
            replaced.append(str(rel))
        else:
            shutil.copy2(source, destination)
            copied.append(str(rel))
    return copied, skipped, replaced


def merge_agents(payload_agents: Path, target_agents: Path) -> str:
    block = payload_agents.read_text(encoding="utf-8").strip()
    if not block.startswith(AGENTS_START) or not block.endswith(AGENTS_END):
        raise ValueError("Payload AGENTS.md is missing installation markers")
    if not target_agents.exists():
        target_agents.write_text(block + "\n", encoding="utf-8")
        return "created"
    current = target_agents.read_text(encoding="utf-8")
    if AGENTS_START in current and AGENTS_END in current:
        before, rest = current.split(AGENTS_START, 1)
        _, after = rest.split(AGENTS_END, 1)
        merged = before.rstrip() + "\n\n" + block + after
        target_agents.write_text(merged.rstrip() + "\n", encoding="utf-8")
        return "updated marked block"
    target_agents.write_text(current.rstrip() + "\n\n" + block + "\n", encoding="utf-8")
    return "appended marked block"


def merge_unique_list(existing: Any, additions: list[str]) -> list[str]:
    result: list[str] = []
    if isinstance(existing, list):
        result.extend(str(item) for item in existing)
    for item in additions:
        if item not in result:
            result.append(item)
    return result


def configure_opencode(payload_template: Path, target: Path) -> str:
    template = load_json(payload_template)
    json_path = target / "opencode.json"
    jsonc_path = target / "opencode.jsonc"

    if jsonc_path.exists() and not json_path.exists():
        write_json(target / "opencode.agentic.snippet.json", template)
        return (
            "opencode.jsonc detected; wrote opencode.agentic.snippet.json for safe manual merge. "
            "Remove any old legacy-project reference manually."
        )

    if json_path.exists():
        try:
            config = load_json(json_path)
        except Exception as exc:
            write_json(target / "opencode.agentic.snippet.json", template)
            return f"could not parse opencode.json ({exc}); wrote manual merge snippet"
    else:
        config = {"$schema": "https://opencode.ai/config.json"}

    config["instructions"] = merge_unique_list(config.get("instructions"), template.get("instructions", []))

    template_watcher = template.get("watcher", {})
    if isinstance(template_watcher, dict):
        watcher = config.setdefault("watcher", {})
        if isinstance(watcher, dict):
            watcher["ignore"] = merge_unique_list(watcher.get("ignore"), template_watcher.get("ignore", []))

    removed_legacy = False
    references = config.get("references")
    if isinstance(references, dict) and "legacy-project" in references:
        references.pop("legacy-project", None)
        removed_legacy = True
        if not references:
            config.pop("references", None)

    write_json(json_path, config)
    suffix = "; removed old legacy-project reference" if removed_legacy else ""
    return "created or merged opencode.json" + suffix


def append_gitignore(target: Path) -> str:
    entries = [".agentic/reports/", ".agentic/tmp/"]
    path = target / ".gitignore"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = {line.strip() for line in existing.splitlines()}
    missing = [entry for entry in entries if entry not in lines]
    if not missing:
        return "already configured"
    addition = "\n# Agentic evaluation outputs\n" + "\n".join(missing) + "\n"
    path.write_text(existing.rstrip() + addition, encoding="utf-8")
    return "updated"


def main() -> int:
    args = parse_args()
    package_root = Path(__file__).resolve().parent.parent
    payload = package_root / "payload"
    target = Path(args.target).expanduser().resolve()
    excel = (
        Path(args.options_excel).expanduser().resolve()
        if args.options_excel
        else (package_root / "reference" / "app_names_and_urls.xlsx").resolve()
    )

    if not target.exists() or not target.is_dir():
        print(f"ERROR: target directory does not exist: {target}", file=sys.stderr)
        return 2
    if not excel.exists() or not excel.is_file():
        print(f"ERROR: options Excel file does not exist: {excel}", file=sys.stderr)
        return 2
    if excel.suffix.lower() != ".xlsx":
        print("ERROR: --options-excel must be an .xlsx file", file=sys.stderr)
        return 2
    if not payload.exists():
        print(f"ERROR: package payload is missing: {payload}", file=sys.stderr)
        return 2

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_root = target / ".agentic" / "install-backups" / timestamp

    copied, skipped, replaced = copy_payload_tree(payload, target, args.force, backup_root)
    agents_result = merge_agents(payload / "AGENTS.md", target / "AGENTS.md")
    opencode_result = configure_opencode(payload / "opencode.agentic.template.json", target)
    gitignore_result = append_gitignore(target)

    catalog_path = target / ".agentic" / "migration" / "legacy-endpoint-catalog.json"
    source_copy = target / ".agentic" / "migration" / "source" / "previous-project-options.xlsx"
    try:
        catalog = build_catalog(
            excel,
            catalog_path,
            source_copy,
            requested_sheet=args.sheet,
            preserve_existing=not args.replace_catalog_progress,
        )
    except Exception as exc:
        print(f"ERROR: failed to import options Excel: {exc}", file=sys.stderr)
        return 2

    installation = {
        "workflow": "opencode-realtime-agentic-workflow",
        "version": VERSION,
        "installed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "target": str(target),
        "options_excel": str(excel),
        "installed_options_copy": str(source_copy),
        "catalog_item_count": catalog["source"]["item_count"],
        "force": bool(args.force),
        "copied_files": copied,
        "replaced_files": replaced,
        "skipped_files": skipped,
    }
    write_json(target / ".agentic" / "installation.json", installation)

    print("OpenCode real-time agentic workflow installed.")
    print(f"Target: {target}")
    print(f"Options Excel: {excel}")
    print(f"Imported catalog items: {catalog['source']['item_count']}")
    print(f"AGENTS.md: {agents_result}")
    print(f"OpenCode config: {opencode_result}")
    print(f".gitignore: {gitignore_result}")
    print(f"Copied: {len(copied)}, replaced: {len(replaced)}, skipped: {len(skipped)}")
    if skipped:
        conflicts = [item for item in skipped if "conflict preserved" in item]
        if conflicts:
            print("Preserved conflicts:")
            for item in conflicts:
                print(f"  - {item}")
    print("\nNext commands:")
    print(f"  cd {target}")
    print("  python scripts/validate_agentic_workflow.py")
    print("  opencode")
    print("  /bootstrap-agentic")
    print("  /migrate-selected-options <selected rows, routes, or options plus any behavior details>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
