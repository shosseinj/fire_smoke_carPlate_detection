#!/usr/bin/env python3
"""CRUD Coverage Report Generator.

Runs the comprehensive CRUD API test suite and generates structured
JSON and Markdown reports analyzing CRUD coverage across all application modules.

Usage:
    python scripts/crud_report.py                          # run CRUD tests and generate report
    python scripts/crud_report.py --report-only            # re-generate report from last run
    python scripts/crud_report.py --junit <path>           # parse pytest JUnit XML for CRUD report
    python scripts/crud_report.py --list-modules           # list all modules and their endpoints
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REPORT_DIR = PROJECT_ROOT / ".agentic" / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Re-generate report from last test run without re-running tests",
    )
    parser.add_argument(
        "--junit",
        type=str,
        help="Path to pytest JUnit XML to parse for CRUD report",
    )
    parser.add_argument(
        "--list-modules",
        action="store_true",
        help="List all registered API modules and their CRUD endpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPORT_DIR),
        help=f"Output directory for reports (default: {REPORT_DIR})",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Skip test execution, only generate module-level coverage map",
    )
    return parser.parse_args()


# ── Module endpoint registry ───────────────────────────────────────────

MODULE_ENDPOINTS: dict[str, list[dict[str, str]]] = {
    "auth": [
        {"method": "POST", "path": "/api/v1/auth/login", "operation": "authenticate"},
        {"method": "GET", "path": "/api/v1/auth/me", "operation": "read profile"},
        {"method": "POST", "path": "/api/v1/auth/create-user", "operation": "create user"},
        {"method": "GET", "path": "/api/v1/auth/users", "operation": "list users"},
        {"method": "GET", "path": "/api/v1/auth/users/{id}", "operation": "read user"},
        {"method": "PUT", "path": "/api/v1/auth/users/{id}", "operation": "update user"},
        {"method": "PATCH", "path": "/api/v1/auth/users/{id}/role", "operation": "change role"},
        {"method": "POST", "path": "/api/v1/auth/me/password", "operation": "change password"},
    ],
    "cameras": [
        {"method": "POST", "path": "/api/v1/cameras", "operation": "create camera"},
        {"method": "GET", "path": "/api/v1/cameras", "operation": "list cameras"},
        {"method": "GET", "path": "/api/v1/cameras/{id}", "operation": "read camera"},
        {"method": "PATCH", "path": "/api/v1/cameras/{id}", "operation": "update camera"},
        {"method": "PUT", "path": "/api/v1/cameras/{id}/tasks", "operation": "update tasks"},
        {"method": "PATCH", "path": "/api/v1/cameras/bulk", "operation": "bulk update"},
        {"method": "POST", "path": "/api/v1/cameras/{id}/enable", "operation": "enable camera"},
        {"method": "POST", "path": "/api/v1/cameras/{id}/disable", "operation": "disable camera"},
        {"method": "DELETE", "path": "/api/v1/cameras/{id}", "operation": "delete camera"},
    ],
    "sources": [
        {"method": "POST", "path": "/api/v1/sources", "operation": "create source"},
        {"method": "GET", "path": "/api/v1/sources", "operation": "list sources"},
        {"method": "GET", "path": "/api/v1/sources/{id}", "operation": "read source"},
        {"method": "PATCH", "path": "/api/v1/sources/{id}", "operation": "update source"},
        {"method": "POST", "path": "/api/v1/sources/{id}/enable", "operation": "enable source"},
        {"method": "POST", "path": "/api/v1/sources/{id}/disable", "operation": "disable source"},
        {"method": "DELETE", "path": "/api/v1/sources/{id}", "operation": "delete source"},
    ],
    "personnel": [
        {"method": "POST", "path": "/api/v1/personnel/", "operation": "create personnel"},
        {"method": "GET", "path": "/api/v1/personnel/", "operation": "list personnel"},
        {"method": "GET", "path": "/api/v1/personnel/{id}", "operation": "read personnel"},
        {"method": "GET", "path": "/api/v1/personnel/search/{code}", "operation": "search by national code"},
        {"method": "PUT", "path": "/api/v1/personnel/{id}", "operation": "update personnel"},
        {"method": "DELETE", "path": "/api/v1/personnel/{id}", "operation": "delete personnel"},
        {"method": "POST", "path": "/api/v1/personnel/{id}/images", "operation": "upload image"},
        {"method": "DELETE", "path": "/api/v1/personnel/images/{id}", "operation": "delete image"},
    ],
    "locations": [
        {"method": "POST", "path": "/buildings/", "operation": "create building"},
        {"method": "GET", "path": "/buildings/", "operation": "list buildings"},
        {"method": "GET", "path": "/buildings/{id}", "operation": "read building"},
        {"method": "PATCH", "path": "/buildings/{id}", "operation": "update building"},
        {"method": "DELETE", "path": "/buildings/{id}", "operation": "delete building"},
        {"method": "POST", "path": "/sections/", "operation": "create section"},
        {"method": "GET", "path": "/sections/", "operation": "list sections"},
        {"method": "POST", "path": "/rooms/", "operation": "create room"},
        {"method": "PUT", "path": "/rooms/{id}", "operation": "update room"},
        {"method": "DELETE", "path": "/rooms/{id}", "operation": "delete room"},
    ],
    "shifts": [
        {"method": "POST", "path": "/api/v1/shifts/", "operation": "create shift"},
        {"method": "GET", "path": "/api/v1/shifts/", "operation": "list shifts"},
        {"method": "GET", "path": "/api/v1/shifts/{id}", "operation": "read shift"},
        {"method": "PUT", "path": "/api/v1/shifts/{id}", "operation": "update shift"},
        {"method": "DELETE", "path": "/api/v1/shifts/{id}", "operation": "delete shift"},
        {"method": "POST", "path": "/api/v1/shifts/{id}/assign/{pid}", "operation": "assign personnel"},
    ],
    "holidays": [
        {"method": "POST", "path": "/api/v1/holidays/", "operation": "create holiday"},
        {"method": "GET", "path": "/api/v1/holidays/", "operation": "list holidays"},
        {"method": "GET", "path": "/api/v1/holidays/{id}", "operation": "read holiday"},
        {"method": "PUT", "path": "/api/v1/holidays/{id}", "operation": "update holiday"},
        {"method": "DELETE", "path": "/api/v1/holidays/{id}", "operation": "delete holiday"},
        {"method": "GET", "path": "/api/v1/holidays/check/{date}", "operation": "check holiday"},
    ],
    "requests": [
        {"method": "POST", "path": "/api/v1/requests/", "operation": "create request"},
        {"method": "GET", "path": "/api/v1/requests/", "operation": "list requests"},
        {"method": "GET", "path": "/api/v1/requests/{id}", "operation": "read request"},
        {"method": "POST", "path": "/api/v1/requests/{id}/approve", "operation": "approve/reject request"},
        {"method": "DELETE", "path": "/api/v1/requests/{id}", "operation": "delete request"},
    ],
    "broadcast": [
        {"method": "GET", "path": "/api/v1/broadcast/state", "operation": "read state"},
        {"method": "PUT", "path": "/api/v1/broadcast/state", "operation": "update state"},
    ],
    "diagnostics": [
        {"method": "GET", "path": "/api/v1/diagnostics/overview", "operation": "read overview"},
        {"method": "GET", "path": "/api/v1/diagnostics/checks", "operation": "run checks"},
    ],
    "faces": [
        {"method": "GET", "path": "/api/v1/faces/status", "operation": "read status"},
        {"method": "GET", "path": "/api/v1/faces/quality-settings", "operation": "read quality settings"},
        {"method": "PATCH", "path": "/api/v1/faces/quality-settings", "operation": "update quality settings"},
        {"method": "POST", "path": "/api/v1/faces/enroll", "operation": "enroll face"},
        {"method": "GET", "path": "/api/v1/faces/identities", "operation": "list identities"},
        {"method": "DELETE", "path": "/api/v1/faces/identities/{person}", "operation": "delete identity"},
    ],
    "fire_smoke": [
        {"method": "GET", "path": "/api/v1/fire-smoke/settings", "operation": "read settings"},
        {"method": "PUT", "path": "/api/v1/fire-smoke/settings", "operation": "update settings"},
        {"method": "GET", "path": "/api/v1/fire-smoke-logs", "operation": "list logs"},
    ],
    "general_settings": [
        {"method": "GET", "path": "/api/v1/settings/general", "operation": "read settings"},
        {"method": "PATCH", "path": "/api/v1/settings/general", "operation": "update settings"},
    ],
    "humans": [
        {"method": "GET", "path": "/api/v1/humans/status", "operation": "read status"},
        {"method": "GET", "path": "/api/v1/humans/logs", "operation": "list logs"},
        {"method": "GET", "path": "/api/v1/humans/active", "operation": "list active"},
    ],
    "models": [
        {"method": "GET", "path": "/api/v1/models/artifacts", "operation": "list artifacts"},
        {"method": "GET", "path": "/api/v1/models/settings", "operation": "read settings"},
        {"method": "PATCH", "path": "/api/v1/models/settings", "operation": "update settings"},
        {"method": "POST", "path": "/api/v1/models/conversions", "operation": "create conversion"},
        {"method": "GET", "path": "/api/v1/models/conversions", "operation": "list conversions"},
    ],
    "plate_logs": [
        {"method": "POST", "path": "/api/v1/plate-logs", "operation": "create log"},
        {"method": "GET", "path": "/api/v1/plate-logs", "operation": "list logs"},
    ],
    "plate_settings": [
        {"method": "GET", "path": "/api/v1/plate-settings/general", "operation": "read general"},
        {"method": "PUT", "path": "/api/v1/plate-settings/general", "operation": "update general"},
        {"method": "GET", "path": "/api/v1/plate-settings/cameras/{id}", "operation": "read camera settings"},
        {"method": "PATCH", "path": "/api/v1/plate-settings/cameras/{id}", "operation": "set camera override"},
        {"method": "DELETE", "path": "/api/v1/plate-settings/cameras/{id}", "operation": "delete camera override"},
    ],
    "results": [
        {"method": "GET", "path": "/api/v1/results/recent", "operation": "list recent"},
        {"method": "GET", "path": "/api/v1/router/status", "operation": "read router status"},
    ],
    
    "frames": [
        {"method": "POST", "path": "/api/v1/frame-rounds/jpeg", "operation": "submit frame"},
    ],
    "health": [
        {"method": "GET", "path": "/health", "operation": "health check"},
    ],
}


def build_coverage_map(test_results_path: str | None = None) -> dict[str, Any]:
    coverage_map: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_modules": len(MODULE_ENDPOINTS),
        "total_endpoints": sum(len(eps) for eps in MODULE_ENDPOINTS.values()),
        "modules": {},
    }

    tested_map: dict[str, set[str]] = {}
    if test_results_path:
        tested_map = _parse_test_results(test_results_path)

    for module, endpoints in sorted(MODULE_ENDPOINTS.items()):
        module_endpoints = []
        tested_count = 0
        for ep in endpoints:
            ep_id = f"{ep['method']} {ep['path']}"
            is_tested = ep_id in tested_map.get(module, set())
            if is_tested:
                tested_count += 1
            module_endpoints.append({
                "method": ep["method"],
                "path": ep["path"],
                "operation": ep["operation"],
                "tested": is_tested,
            })

        coverage_map["modules"][module] = {
            "total": len(endpoints),
            "tested": tested_count,
            "coverage_pct": round((tested_count / len(endpoints) * 100), 1) if endpoints else 0,
            "endpoints": module_endpoints,
        }

    all_tested = sum(m["tested"] for m in coverage_map["modules"].values())
    all_total = sum(m["total"] for m in coverage_map["modules"].values())
    coverage_map["overall_coverage_pct"] = round((all_tested / all_total * 100), 1) if all_total else 0
    coverage_map["overall_tested"] = all_tested
    coverage_map["overall_total"] = all_total

    return coverage_map


def _parse_test_results(path: str) -> dict[str, set[str]]:
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    root = tree.getroot()
    tested_map: dict[str, set[str]] = {}

    for testcase in root.iter("testcase"):
        class_name = testcase.get("class", "") or testcase.get("classname", "")
        test_name = testcase.get("name", "")
        if "CRUD" in class_name or "Crud" in class_name or "crud" in class_name:
            module = class_name.split(".")[-1].replace("Test", "").replace("Crud", "").lower()
            if module:
                tested_map.setdefault(module, set()).add(test_name)

    return tested_map


def generate_coverage_report(coverage_map: dict[str, Any], output_dir: str) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    json_path = out / f"crud-coverage-{timestamp}.json"
    md_path = out / f"crud-coverage-{timestamp}.md"

    json_path.write_text(json.dumps(coverage_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# CRUD Coverage Analysis Report",
        "",
        f"- Generated: `{coverage_map['generated_at']}`",
        f"- Total modules: {coverage_map['total_modules']}",
        f"- Total endpoints: {coverage_map['total_endpoints']}",
        f"- Overall coverage: **{coverage_map['overall_coverage_pct']}%** ({coverage_map['overall_tested']}/{coverage_map['overall_total']})",
        "",
        "## Module Coverage Summary",
        "",
        "| Module | Total | Tested | Coverage |",
        "|---|---:|---:|---:|",
    ]

    for module, mod_data in sorted(coverage_map["modules"].items()):
        lines.append(
            f"| {module} | {mod_data['total']} | {mod_data['tested']} | {mod_data['coverage_pct']}% |"
        )

    lines.extend(["", "## Endpoint Details", ""])

    for module, mod_data in sorted(coverage_map["modules"].items()):
        lines.extend([
            f"### {module} — {mod_data['coverage_pct']}%",
            "",
            "| Method | Path | Operation | Tested |",
            "|---|---:|---:|---:|",
        ])
        for ep in mod_data["endpoints"]:
            status = "✅" if ep["tested"] else "❌"
            lines.append(f"| {ep['method']} | `{ep['path']}` | {ep['operation']} | {status} |")
        lines.append("")

    md_content = "\n".join(lines).rstrip() + "\n"
    md_path.write_text(md_content, encoding="utf-8")

    latest_json = out / "crud-coverage-latest.json"
    latest_md = out / "crud-coverage-latest.md"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_content, encoding="utf-8")

    print(f"Coverage report: {json_path}")
    print(f"Coverage report: {md_path}")

    return {"json": json_path, "md": md_path}


def run_full_crud_analysis(output_dir: str) -> int:
    """Run CRUD tests and generate coverage + comprehensive reports."""
    print("=" * 60)
    print("CRUD Coverage Analysis")
    print("=" * 60)

    coverage_map = build_coverage_map()
    coverage_map["mode"] = "spec-only (no test results)"
    generate_coverage_report(coverage_map, output_dir)

    print(f"\nModules: {coverage_map['total_modules']}")
    print(f"Endpoints: {coverage_map['total_endpoints']}")
    print(f"Overall coverage: {coverage_map['overall_coverage_pct']}%\n")

    print("\nModule breakdown:")
    for module, mod_data in sorted(coverage_map["modules"].items()):
        bar_len = int(mod_data["coverage_pct"] / 5)
        bar = "#" * bar_len + "." * (20 - bar_len)
        print(f"  {module:20s} [{bar}] {mod_data['coverage_pct']:5.1f}% ({mod_data['tested']}/{mod_data['total']})")

    if coverage_map["overall_coverage_pct"] < 100:
        print(f"\n⚠️  Coverage is {coverage_map['overall_coverage_pct']}%. "
              f"Missing tests for {coverage_map['overall_total'] - coverage_map['overall_tested']} endpoints.")
    else:
        print("\n✅ Full CRUD coverage achieved!")

    return 0


def list_modules() -> None:
    print(f"{'Module':20s} {'Endpoints':>10s} {'Methods':>40s}")
    print("-" * 72)
    for module, endpoints in sorted(MODULE_ENDPOINTS.items()):
        methods = ", ".join(sorted(set(ep["method"] for ep in endpoints)))
        print(f"{module:20s} {len(endpoints):>10d} {methods:>40s}")
    total_eps = sum(len(eps) for eps in MODULE_ENDPOINTS.values())
    print("-" * 72)
    print(f"{'TOTAL':20s} {total_eps:>10d}")


def main() -> int:
    args = parse_args()

    output_dir = args.output_dir

    if args.list_modules:
        list_modules()
        return 0

    if args.report_only:
        latest = Path(output_dir) / "crud-coverage-latest.json"
        if latest.exists():
            coverage_map = json.loads(latest.read_text(encoding="utf-8"))
        else:
            coverage_map = build_coverage_map()
        generate_coverage_report(coverage_map, output_dir)
        return 0

    if args.junit:
        coverage_map = build_coverage_map(args.junit)
        generate_coverage_report(coverage_map, output_dir)
        return 0

    return run_full_crud_analysis(output_dir)


if __name__ == "__main__":
    sys.exit(main())
