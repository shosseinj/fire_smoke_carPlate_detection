#!/usr/bin/env python3
"""Import a simple previous-project option/endpoint catalog from an .xlsx file.

Uses only the Python standard library so the installed workflow has no extra
spreadsheet dependency. The workbook should contain a header row. Supported
columns include App Name/Subsystem, URL/Route/Endpoint, Option/Feature/Name,
Method, Description, Expected Behavior, Default, Allowed Values, and Notes.
"""
from __future__ import annotations

import json
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"

HEADER_ALIASES = {
    "app": {"app", "app name", "application", "area", "subsystem", "module", "category", "section"},
    "url": {"url", "route", "endpoint", "path", "api url", "api route"},
    "option": {"option", "option name", "feature", "feature name", "name", "setting", "setting name"},
    "method": {"method", "http method", "verb"},
    "description": {"description", "purpose", "summary"},
    "expected_behavior": {"expected behavior", "behavior", "legacy behavior", "requirements", "requirement"},
    "default_value": {"default", "default value"},
    "allowed_values": {"allowed values", "values", "choices", "enum"},
    "notes": {"notes", "note", "comments", "comment"},
    "priority": {"priority"},
}

PROGRESS_FIELDS = (
    "catalog_status",
    "selected_by_prompt",
    "analysis_status",
    "migration_status",
    "evaluation_status",
    "last_migration_run",
    "implementation_evidence",
    "assumptions",
    "notes",
)


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _slug(value: str) -> str:
    value = _norm(value)
    value = re.sub(r"[^a-z0-9/_{}.-]+", "-", value)
    return value.strip("-") or "unspecified"


def _col_index(reference: str) -> int:
    letters = "".join(ch for ch in reference if ch.isalpha()).upper()
    result = 0
    for ch in letters:
        result = result * 26 + (ord(ch) - 64)
    return max(result - 1, 0)


def _read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    name = "xl/sharedStrings.xml"
    if name not in zf.namelist():
        return []
    root = ET.fromstring(zf.read(name))
    strings: list[str] = []
    for si in root.findall(f"{{{NS_MAIN}}}si"):
        text = "".join((node.text or "") for node in si.iter(f"{{{NS_MAIN}}}t"))
        strings.append(text)
    return strings


def _sheet_path(zf: zipfile.ZipFile, requested_sheet: str | None) -> tuple[str, str]:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall(f"{{{NS_PKG_REL}}}Relationship")
    }
    sheets = workbook.find(f"{{{NS_MAIN}}}sheets")
    if sheets is None:
        raise ValueError("Workbook has no sheets")
    candidates: list[tuple[str, str]] = []
    for sheet in sheets.findall(f"{{{NS_MAIN}}}sheet"):
        name = sheet.attrib.get("name", "")
        rid = sheet.attrib.get(f"{{{NS_REL}}}id", "")
        target = rel_targets.get(rid)
        if target:
            path = target.lstrip("/")
            if not path.startswith("xl/"):
                path = "xl/" + path
            candidates.append((name, path))
    if not candidates:
        raise ValueError("Workbook has no readable worksheets")
    if requested_sheet:
        for name, path in candidates:
            if name.casefold() == requested_sheet.casefold():
                return name, path
        raise ValueError(f"Worksheet not found: {requested_sheet}")
    for name, path in candidates:
        if name.casefold() in {"apps and urls", "options", "endpoints", "catalog"}:
            return name, path
    return candidates[0]


def read_xlsx_rows(path: Path, requested_sheet: str | None = None) -> tuple[str, list[list[str]]]:
    with zipfile.ZipFile(path) as zf:
        shared = _read_shared_strings(zf)
        sheet_name, sheet_path = _sheet_path(zf, requested_sheet)
        root = ET.fromstring(zf.read(sheet_path))
        rows: list[list[str]] = []
        for row in root.findall(f".//{{{NS_MAIN}}}sheetData/{{{NS_MAIN}}}row"):
            values: dict[int, str] = {}
            max_col = -1
            for cell in row.findall(f"{{{NS_MAIN}}}c"):
                ref = cell.attrib.get("r", "A1")
                idx = _col_index(ref)
                max_col = max(max_col, idx)
                cell_type = cell.attrib.get("t")
                value = ""
                if cell_type == "inlineStr":
                    value = "".join((n.text or "") for n in cell.iter(f"{{{NS_MAIN}}}t"))
                else:
                    v = cell.find(f"{{{NS_MAIN}}}v")
                    raw = v.text if v is not None and v.text is not None else ""
                    if cell_type == "s" and raw:
                        try:
                            value = shared[int(raw)]
                        except (ValueError, IndexError):
                            value = raw
                    elif cell_type == "b":
                        value = "true" if raw == "1" else "false"
                    else:
                        value = raw
                values[idx] = value.strip()
            if max_col >= 0:
                rows.append([values.get(i, "") for i in range(max_col + 1)])
        return sheet_name, rows


def _canonical_headers(headers: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for idx, header in enumerate(headers):
        normalized = _norm(header)
        for canonical, aliases in HEADER_ALIASES.items():
            if normalized in aliases and canonical not in mapping:
                mapping[canonical] = idx
                break
    return mapping


def _get(row: list[str], headers: dict[str, int], key: str) -> str | None:
    idx = headers.get(key)
    if idx is None or idx >= len(row):
        return None
    value = row[idx].strip()
    return value or None


def _load_existing(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    items = data.get("items") or data.get("endpoints") or []
    return {str(item.get("id")): item for item in items if isinstance(item, dict) and item.get("id")}


def build_catalog(
    excel_path: Path,
    output_path: Path,
    source_copy_path: Path | None = None,
    requested_sheet: str | None = None,
    preserve_existing: bool = True,
) -> dict[str, Any]:
    sheet_name, rows = read_xlsx_rows(excel_path, requested_sheet)
    if not rows:
        raise ValueError("Excel workbook has no rows")
    headers = _canonical_headers(rows[0])
    if not any(key in headers for key in ("url", "option")):
        raise ValueError(
            "Excel needs a URL/Route/Endpoint column or an Option/Feature/Name column"
        )

    existing = _load_existing(output_path if preserve_existing else None)
    items: list[dict[str, Any]] = []
    seen: Counter[str] = Counter()
    apps: Counter[str] = Counter()

    for row_number, row in enumerate(rows[1:], start=2):
        app = _get(row, headers, "app") or "Uncategorized"
        url = _get(row, headers, "url")
        option = _get(row, headers, "option")
        if not url and not option:
            continue
        method = _get(row, headers, "method")
        identity = url or option or f"row-{row_number}"
        base_id = f"{_slug(app)}:{_slug(identity)}"
        if method:
            base_id += f":{_slug(method)}"
        seen[base_id] += 1
        item_id = base_id if seen[base_id] == 1 else f"{base_id}:{seen[base_id]}"
        item: dict[str, Any] = {
            "id": item_id,
            "app": app,
            "url": url,
            "option": option,
            "method": method.upper() if method else None,
            "description": _get(row, headers, "description"),
            "expected_behavior": _get(row, headers, "expected_behavior"),
            "default_value": _get(row, headers, "default_value"),
            "allowed_values": _get(row, headers, "allowed_values"),
            "priority": _get(row, headers, "priority"),
            "source_row": row_number,
            "specification_status": "PARTIAL_CATALOG_SPECIFICATION",
            "catalog_status": "PENDING",
            "selected_by_prompt": False,
            "analysis_status": "NOT_ANALYZED",
            "migration_status": "PENDING",
            "evaluation_status": "NOT_TESTED",
            "last_migration_run": None,
            "implementation_evidence": [],
            "assumptions": [],
            "notes": [],
        }
        source_notes = _get(row, headers, "notes")
        if source_notes:
            item["notes"] = [source_notes]
        previous = existing.get(item_id)
        if previous:
            for field in PROGRESS_FIELDS:
                if field in previous:
                    item[field] = previous[field]
            if previous.get("specification_status") == "PROMPT_ENRICHED":
                item["specification_status"] = "PROMPT_ENRICHED"
        items.append(item)
        apps[app] += 1

    if not items:
        raise ValueError("No option or endpoint rows were found in the Excel workbook")

    if source_copy_path:
        source_copy_path.parent.mkdir(parents=True, exist_ok=True)
        if excel_path.resolve() != source_copy_path.resolve():
            shutil.copy2(excel_path, source_copy_path)

    catalog = {
        "schema_version": 2,
        "source": {
            "type": "excel_option_catalog",
            "original_filename": excel_path.name,
            "installed_copy": str(source_copy_path) if source_copy_path else None,
            "sheet": sheet_name,
            "columns": rows[0],
            "item_count": len(items),
            "app_count": len(apps),
            "important_note": (
                "This Excel file is a catalog or partial specification, not previous source code. "
                "Use prompt details plus verified target-project conventions. Never claim exact previous behavior "
                "or behavioral parity unless the user supplied enough evidence."
            ),
        },
        "migration_mode": "PROMPT_SCOPED_INCREMENTAL_CATALOG_ONLY",
        "evidence_policy": {
            "priority": [
                "CURRENT_USER_PROMPT",
                "EXCEL_CATALOG_FIELDS",
                "VERIFIED_TARGET_PROJECT_PATTERN",
                "EXPLICITLY_RECORDED_SAFE_ASSUMPTION",
            ],
            "forbidden_claim": "Do not claim legacy parity without previous implementation evidence.",
            "underspecified_status": "NEEDS_DETAILS_OR_TARGET_INFERENCE",
        },
        "default_unselected_status": "PENDING",
        "apps": [
            {"name": name, "item_count": count}
            for name, count in sorted(apps.items(), key=lambda pair: pair[0].casefold())
        ],
        "items": items,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return catalog
