from __future__ import annotations
import json, re, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
required = [
    ".opencode/agents/safe-dev-orchestrator.md",
    ".opencode/agents/impact-checker.md",
    ".opencode/agents/scoped-implementer.md",
    ".opencode/commands/init-project-state.md",
    ".opencode/commands/develop-safely.md",
    ".opencode/commands/review-change.md",
    ".agentic/project-state/state.json",
]
for rel in required:
    if not (root / rel).is_file():
        print(f"ERROR missing {rel}")
        sys.exit(1)
for path in list((root / ".opencode/agents").glob("*.md")) + list((root / ".opencode/commands").glob("*.md")):
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        print(f"ERROR invalid frontmatter {path}")
        sys.exit(1)
    if re.search(r":\s*(null|~)\s*$", text.split("\n---\n",1)[0], re.M | re.I):
        print(f"ERROR null frontmatter {path}")
        sys.exit(1)
state = json.loads((root / ".agentic/project-state/state.json").read_text(encoding="utf-8"))
for key in ("accepted_features", "protected_options", "blockers"):
    if key not in state:
        print(f"ERROR missing state key {key}")
        sys.exit(1)
print("Minimal safe OpenCode workflow validation: PASS")
