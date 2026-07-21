#!/usr/bin/env python3
"""Run machine-readable project section evaluations.

The runner intentionally uses only the Python standard library. OpenCode should
populate .agentic/evaluation/section-registry.json with verified project commands.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

VALID_STATUSES = {
    "PASS",
    "PASS_WITH_WARNINGS",
    "FAIL",
    "BLOCKED",
    "NOT_CONFIGURED",
    "NOT_TESTED",
    "NOT_APPLICABLE",
    "NOT_RUNTIME_VALIDATED",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Evaluate all enabled sections")
    group.add_argument("--section", help="Evaluate one section ID")
    group.add_argument("--list", action="store_true", help="List registered sections")
    parser.add_argument("--registry", default=".agentic/evaluation/section-registry.json")
    parser.add_argument("--report-dir", default=".agentic/reports")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Registry not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return value


def command_text(command: Any) -> str:
    if isinstance(command, str):
        return command
    if isinstance(command, list):
        return " ".join(shlex.quote(str(x)) for x in command)
    raise ValueError("Command must be a string or list")


def run_command(project_root: Path, section_id: str, item: dict[str, Any]) -> dict[str, Any]:
    name = str(item.get("name") or item.get("level") or "command")
    command = item.get("command")
    required = bool(item.get("required", True))
    timeout = int(item.get("timeout_seconds", 300))
    cwd_raw = str(item.get("cwd", "."))
    cwd = (project_root / cwd_raw).resolve()

    if not command:
        return {
            "name": name,
            "status": "NOT_CONFIGURED",
            "required": required,
            "reason": "Missing command",
        }
    if not cwd.exists() or not cwd.is_dir():
        return {
            "name": name,
            "status": "BLOCKED",
            "required": required,
            "reason": f"Working directory does not exist: {cwd}",
        }

    cmd_text = command_text(command)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            cmd_text,
            cwd=cwd,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=os.environ.copy(),
        )
        duration = time.monotonic() - started
        status = "PASS" if completed.returncode == 0 else "FAIL"
        return {
            "name": name,
            "level": item.get("level"),
            "command": cmd_text,
            "cwd": str(cwd),
            "required": required,
            "status": status,
            "returncode": completed.returncode,
            "duration_seconds": round(duration, 3),
            "stdout": completed.stdout[-20000:],
            "stderr": completed.stderr[-20000:],
        }
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - started
        return {
            "name": name,
            "level": item.get("level"),
            "command": cmd_text,
            "cwd": str(cwd),
            "required": required,
            "status": "FAIL",
            "returncode": None,
            "duration_seconds": round(duration, 3),
            "reason": f"Timed out after {timeout} seconds",
            "stdout": (exc.stdout or "")[-20000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-20000:] if isinstance(exc.stderr, str) else "",
        }
    except OSError as exc:
        return {
            "name": name,
            "level": item.get("level"),
            "command": cmd_text,
            "cwd": str(cwd),
            "required": required,
            "status": "BLOCKED",
            "reason": str(exc),
        }


def section_status(section: dict[str, Any], results: list[dict[str, Any]]) -> str:
    if not section.get("enabled", True):
        return "NOT_APPLICABLE"
    if not results:
        return "NOT_CONFIGURED"

    required = [r for r in results if r.get("required", True)]
    statuses = {r.get("status") for r in required}
    if "FAIL" in statuses:
        return "FAIL"
    if "BLOCKED" in statuses:
        return "BLOCKED"
    if "NOT_CONFIGURED" in statuses:
        return "NOT_CONFIGURED"
    if section.get("runtime_required"):
        runtime_results = [r for r in results if r.get("level") in {"runtime", "performance", "deployment"}]
        if not runtime_results or any(r.get("status") != "PASS" for r in runtime_results):
            return "NOT_RUNTIME_VALIDATED"
    optional_failures = [r for r in results if not r.get("required", True) and r.get("status") != "PASS"]
    return "PASS_WITH_WARNINGS" if optional_failures else "PASS"


def overall_status(section_results: list[dict[str, Any]]) -> str:
    critical = [r for r in section_results if r.get("critical")]
    considered = critical or section_results
    statuses = {r.get("status") for r in considered}
    if "FAIL" in statuses:
        return "FAIL"
    if "BLOCKED" in statuses:
        return "BLOCKED"
    if "NOT_RUNTIME_VALIDATED" in statuses:
        return "NOT_RUNTIME_VALIDATED"
    if "NOT_CONFIGURED" in statuses:
        return "NOT_CONFIGURED"
    if "PASS_WITH_WARNINGS" in statuses:
        return "PASS_WITH_WARNINGS"
    if statuses == {"NOT_APPLICABLE"}:
        return "NOT_APPLICABLE"
    return "PASS"


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Agentic Evaluation Report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Overall status: **{report['overall_status']}**",
        f"- Platform: `{report['environment']['platform']}`",
        f"- Python: `{report['environment']['python']}`",
        "",
        "## Section results",
        "",
        "| Section | Critical | Runtime required | Status | Commands |",
        "|---|---:|---:|---|---:|",
    ]
    for section in report["sections"]:
        lines.append(
            f"| {section['id']} | {str(section['critical']).lower()} | "
            f"{str(section['runtime_required']).lower()} | **{section['status']}** | {len(section['commands'])} |"
        )
    for section in report["sections"]:
        lines.extend(["", f"## {section['id']}: {section['status']}", ""])
        if section.get("notes"):
            lines.append(section["notes"])
            lines.append("")
        for result in section["commands"]:
            lines.append(f"### {result.get('name', 'command')} — {result.get('status')}")
            lines.append("")
            if result.get("command"):
                lines.append(f"`{result['command']}`")
                lines.append("")
            if result.get("reason"):
                lines.append(f"Reason: {result['reason']}")
                lines.append("")
            if result.get("stderr"):
                lines.extend(["<details><summary>stderr</summary>", "", "```text", result["stderr"], "```", "", "</details>", ""])
            if result.get("stdout"):
                lines.extend(["<details><summary>stdout</summary>", "", "```text", result["stdout"], "```", "", "</details>", ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    project_root = Path.cwd().resolve()
    registry_path = (project_root / args.registry).resolve()
    registry = load_json(registry_path)
    sections = registry.get("sections", [])
    if not isinstance(sections, list):
        raise RuntimeError("Registry field 'sections' must be a list")

    if args.list:
        for section in sections:
            print(f"{section.get('id')}\tcritical={bool(section.get('critical'))}\tenabled={section.get('enabled', True)}\t{section.get('name', '')}")
        return 0

    selected = [s for s in sections if s.get("enabled", True)]
    if args.section:
        selected = [s for s in sections if s.get("id") == args.section]
        if not selected:
            print(f"ERROR: unknown section: {args.section}", file=sys.stderr)
            return 2

    section_results: list[dict[str, Any]] = []
    for section in selected:
        commands = section.get("commands", [])
        if not isinstance(commands, list):
            commands = []
        results: list[dict[str, Any]] = []
        for item in commands:
            if isinstance(item, str):
                item = {"name": item, "command": item}
            if not isinstance(item, dict):
                results.append({"name": "invalid command", "status": "BLOCKED", "required": True, "reason": "Command entry must be object or string"})
                continue
            result = run_command(project_root, str(section.get("id")), item)
            results.append(result)
            print(f"[{section.get('id')}] {result.get('name')}: {result.get('status')}")
            if args.fail_fast and result.get("status") in {"FAIL", "BLOCKED"} and result.get("required", True):
                break
        status = section_status(section, results)
        section_results.append({
            "id": section.get("id"),
            "name": section.get("name"),
            "critical": bool(section.get("critical")),
            "runtime_required": bool(section.get("runtime_required")),
            "status": status,
            "owner_paths": section.get("owner_paths", []),
            "dependencies": section.get("dependencies", []),
            "consumers": section.get("consumers", []),
            "metrics": section.get("metrics", []),
            "acceptance": section.get("acceptance", {}),
            "notes": section.get("notes"),
            "commands": results,
        })

    report = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_root": str(project_root),
        "registry": str(registry_path),
        "overall_status": overall_status(section_results),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "sections": section_results,
    }

    report_dir = (project_root / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    scope = args.section or "all"
    json_path = report_dir / f"evaluation-{scope}-{timestamp}.json"
    md_path = report_dir / f"evaluation-{scope}-{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")

    latest_json = report_dir / f"evaluation-{scope}-latest.json"
    latest_md = report_dir / f"evaluation-{scope}-latest.md"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Overall: {report['overall_status']}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")

    return 0 if report["overall_status"] in {"PASS", "PASS_WITH_WARNINGS", "NOT_APPLICABLE"} else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
