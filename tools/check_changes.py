#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".workflow" / "phase.json"

def git(*args, check=True):
    p = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    if check and p.returncode != 0:
        print(p.stderr.strip(), file=sys.stderr)
        sys.exit(p.returncode)
    return p.stdout.strip(), p.returncode

def allowed(path, rules):
    p = path.replace("\\", "/").lstrip("./")
    for rule in rules:
        r = rule.replace("\\", "/").rstrip("/").lstrip("./")
        if p == r or p.startswith(r + "/"):
            return True
    return False

if not STATE.exists():
    print("ERROR: no phase state. Run tools/start_phase.py first.")
    sys.exit(1)

state = json.loads(STATE.read_text(encoding="utf-8"))
branch, _ = git("branch", "--show-current")
current_ai, ai_rc = git("rev-parse", "main", check=False)

errors = []
if branch != state["required_branch"]:
    errors.append(f"Current branch changed: expected {state['required_branch']}, got {branch}")
if ai_rc != 0:
    errors.append("main ref can no longer be resolved")
elif current_ai != state["ai_branch_commit"]:
    errors.append(
        "main CHANGED: "
        f"expected {state['ai_branch_commit']}, got {current_ai}"
    )

# Changes relative to the baseline commit, plus untracked files.
changed_text, _ = git("diff", "--name-only", state["baseline_commit"])
changed = {x.strip() for x in changed_text.splitlines() if x.strip()}
untracked_text, _ = git("ls-files", "--others", "--exclude-standard")
untracked = {x.strip() for x in untracked_text.splitlines() if x.strip()}
changed |= untracked

# Workflow control files are expected to exist after installation.
workflow_files = {
    "AGENTS.md", "PROJECT_STATE.md", "PROTECTED_COMPONENTS.md", "DECISIONS.md",
    "README.md", "opencode.json", ".workflow/phase.json", "tools/start_phase.py",
    "tools/check_changes.py"
}
def is_workflow_file(path):
    return path in workflow_files or path.startswith(".opencode/")

allowed_paths = state.get("allowed_paths", [])
unexpected = sorted(p for p in changed if not is_workflow_file(p) and not allowed(p, allowed_paths))
expected = sorted(p for p in changed if is_workflow_file(p) or allowed(p, allowed_paths))

print("\nGit protection report")
print("=" * 60)
print(f"Branch: {branch}")
print(f"main unchanged: {'YES' if ai_rc == 0 and current_ai == state['ai_branch_commit'] else 'NO'}")

print("\nAllowed/expected changed files:")
if expected:
    for p in expected:
        print(f"  OK   {p}")
else:
    print("  (none)")

print("\nUnexpected/protected changed files:")
if unexpected:
    for p in unexpected:
        print(f"  FAIL {p}")
else:
    print("  (none)")

if errors:
    print("\nBranch protection errors:")
    for e in errors:
        print(f"  FAIL {e}")

print("=" * 60)
if errors or unexpected:
    print("CHECK FAILED — report these changes before continuing.")
    sys.exit(1)

print("CHECK PASSED — no unexpected changes detected.")
