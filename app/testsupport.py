"""Test support utilities for comprehensive CRUD API testing.

This module provides shared test infrastructure including runtime setup,
CRUD test execution, and report generation. It lives inside the app package
to avoid conflicts with the installed `tests` namespace package.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dataclasses import replace as dc_replace
from fastapi.testclient import TestClient

from app.config import settings as _BASE_SETTINGS
from app.runtime import build_runtime, Runtime


CRUD_REPORT_DIR = Path(".agentic/reports")
AUTH_USER = "admin"
AUTH_PASS = "admin123"
VALID_CODE_1 = "1234567891"
VALID_CODE_2 = "9876543210"
VALID_CODE_3 = "1234123411"


@dataclass
class CrudTestContext:
    runtime: Runtime
    old_runtime: Runtime
    old_auth_store: Any
    client: TestClient
    admin_token: str
    operator_token: str
    viewer_token: str


@dataclass
class CrudTestResult:
    module: str = ""
    endpoint: str = ""
    method: str = ""
    status: str = "NOT_TESTED"
    status_code: int | None = None
    detail: str = ""
    error: str = ""


@dataclass
class ModuleCrudReport:
    module: str = ""
    tests: list[CrudTestResult] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""

    @property
    def total(self) -> int:
        return len(self.tests)

    @property
    def passed(self) -> int:
        return sum(1 for t in self.tests if t.status == "PASS")

    @property
    def failed(self) -> int:
        return sum(1 for t in self.tests if t.status == "FAIL")

    @property
    def skipped(self) -> int:
        return sum(1 for t in self.tests if t.status == "SKIPPED")

    @property
    def coverage_pct(self) -> float:
        if not self.tests:
            return 0.0
        return round((self.passed / self.total) * 100, 1)


class CrudTestFailure(Exception):
    pass


def build_test_settings(tmp_path: Path) -> Any:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        raise RuntimeError("TEST_DATABASE_URL must point to a disposable PostgreSQL database")
    return dc_replace(
        _BASE_SETTINGS,
        processor_mode="mock",
        database_url=database_url,
        saved_media_path=tmp_path / "saved_media",
        video_ingestion_enabled=False,
        auth_default_admin_username=AUTH_USER,
        auth_default_admin_password=AUTH_PASS,
    )


def setup_crud_context(tmp_path: Path) -> CrudTestContext:
    import app.main as main_module
    from app.core import auth as auth_core
    from app.core.auth_store import AuthStore
    from app.database import get_database

    test_settings = build_test_settings(tmp_path)
    test_runtime = build_runtime(test_settings)
    old_runtime = main_module.runtime
    main_module.runtime = test_runtime

    old_store = auth_core._auth_store
    auth_core._auth_store = AuthStore(get_database(test_settings.database_url))
    auth_core._auth_store.seed_default_admin(AUTH_USER, AUTH_PASS)

    client = TestClient(main_module.app)

    admin_token = _get_token(client, AUTH_USER, AUTH_PASS)
    operator_token = _create_user_and_get_token(
        client, admin_token, "operator_crud", "Operator1!", "operator"
    )
    viewer_token = _create_user_and_get_token(
        client, admin_token, "viewer_crud", "Viewer123!", "viewer"
    )

    return CrudTestContext(
        runtime=test_runtime,
        old_runtime=old_runtime,
        old_auth_store=old_store,
        client=client,
        admin_token=admin_token,
        operator_token=operator_token,
        viewer_token=viewer_token,
    )


def teardown_crud_context(ctx: CrudTestContext) -> None:
    import app.main as main_module
    from app.core import auth as auth_core

    ctx.runtime.close()
    main_module.runtime = ctx.old_runtime
    auth_core._auth_store = ctx.old_auth_store


def _get_token(client: TestClient, username: str, password: str) -> str:
    resp = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    return resp.json()["access_token"]


def _create_user_and_get_token(
    client: TestClient, admin_token: str, username: str, password: str, role: str
) -> str:
    resp = client.post(
        "/api/v1/auth/create-user",
        json={
            "username": username,
            "password": password,
            "role": role,
            "email": f"{username}@test.local",
            "confirm_password": password,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    if resp.status_code not in (200, 201):
        return _get_token(client, username, password) if resp.status_code == 409 else ""
    return _get_token(client, username, password)


def request_json(
    client: TestClient,
    method: str,
    path: str,
    json_body: Any = None,
    token: str | None = None,
    expected_status: int | None = None,
) -> tuple[int, Any]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if json_body is not None:
        headers.setdefault("Content-Type", "application/json")

    resp = client.request(method, path, json=json_body, headers=headers)

    if expected_status is not None:
        assert resp.status_code == expected_status, (
            f"{method} {path}: expected {expected_status}, got {resp.status_code}: {resp.text}"
        )

    try:
        body = resp.json()
    except Exception:
        body = resp.text
    return resp.status_code, body


def run_crud_test(
    ctx: CrudTestContext,
    module: str,
    label: str,
    test_fn: Callable[[CrudTestContext], None],
    endpoint: str = "",
    method: str = "",
) -> CrudTestResult:
    result = CrudTestResult(module=module, endpoint=endpoint, method=method)
    try:
        test_fn(ctx)
        result.status = "PASS"
    except AssertionError as exc:
        result.status = "FAIL"
        result.detail = str(exc)[:500]
    except Exception as exc:
        result.status = "FAIL"
        result.error = f"{type(exc).__name__}: {str(exc)[:500]}"
    return result


def generate_crud_report(reports: list[ModuleCrudReport]) -> dict[str, Any]:
    total_tests = sum(r.total for r in reports)
    total_passed = sum(r.passed for r in reports)
    total_failed = sum(r.failed for r in reports)
    total_skipped = sum(r.skipped for r in reports)

    modules_data = []
    for report in reports:
        modules_data.append({
            "module": report.module,
            "total": report.total,
            "passed": report.passed,
            "failed": report.failed,
            "skipped": report.skipped,
            "coverage_pct": report.coverage_pct,
            "tests": [
                {
                    "endpoint": t.endpoint,
                    "method": t.method,
                    "status": t.status,
                    "status_code": t.status_code,
                    "detail": t.detail,
                    "error": t.error,
                }
                for t in report.tests
            ],
        })

    report_data: dict[str, Any] = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_type": "comprehensive_crud_analysis",
        "summary": {
            "modules": len(reports),
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_skipped": total_skipped,
            "overall_pass_rate": round((total_passed / total_tests * 100), 1) if total_tests else 0.0,
        },
        "modules": modules_data,
    }
    return report_data


def write_crud_report(report_data: dict[str, Any], prefix: str = "crud-analysis") -> dict[str, Path]:
    report_dir = Path(".agentic/reports").resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    json_path = report_dir / f"{prefix}-{timestamp}.json"
    md_path = report_dir / f"{prefix}-{timestamp}.md"

    json_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_content = _report_to_markdown(report_data)
    md_path.write_text(md_content, encoding="utf-8")

    latest_json = report_dir / f"{prefix}-latest.json"
    latest_md = report_dir / f"{prefix}-latest.md"
    latest_json.write_text(json.dumps(report_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    latest_md.write_text(md_content, encoding="utf-8")

    print(f"CRUD report: {json_path}")
    print(f"CRUD report: {md_path}")

    return {"json": json_path, "md": md_path}


def _report_to_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Comprehensive CRUD Analysis Report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Overall pass rate: **{summary['overall_pass_rate']}%**",
        f"- Modules tested: {summary['modules']}",
        f"- Total tests: {summary['total_tests']}",
        f"- Passed: {summary['total_passed']}",
        f"- Failed: {summary['total_failed']}",
        f"- Skipped: {summary['total_skipped']}",
        "",
        "## Module Summary",
        "",
        "| Module | Total | Passed | Failed | Skipped | Coverage % |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for mod in sorted(report["modules"], key=lambda m: m["module"]):
        lines.append(
            f"| {mod['module']} | {mod['total']} | {mod['passed']} | "
            f"{mod['failed']} | {mod['skipped']} | {mod['coverage_pct']}% |"
        )

    lines.extend(["", "## Detailed Results", ""])

    for mod in sorted(report["modules"], key=lambda m: m["module"]):
        lines.extend([
            f"### {mod['module']} — {mod['passed']}/{mod['total']} passed ({mod['coverage_pct']}%)",
            "",
            "| Endpoint | Method | Status | Detail |",
            "|---|---:|---:|---:|",
        ])
        for t in mod["tests"]:
            detail = ""
            if t.get("error"):
                detail = t["error"][:100]
            elif t.get("detail"):
                detail = t["detail"][:100]
            lines.append(
                f"| `{t['endpoint']}` | {t['method']} | **{t['status']}** | {detail} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
