#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".workflow" / "phase.json"

def git(*args):
    p = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    if p.returncode != 0:
        print(p.stderr.strip(), file=sys.stderr)
        sys.exit(p.returncode)
    return p.stdout.strip()

parser = argparse.ArgumentParser(description="Start a protected live Redis/MinIO development phase")
parser.add_argument("--allow", nargs="*", default=[], help="Paths allowed to change in this phase")
args = parser.parse_args()

branch = git("branch", "--show-current")
if branch != "live":
    print(f"ERROR: current branch is '{branch}', expected 'live'.")
    sys.exit(1)

baseline = git("rev-parse", "HEAD")
try:
    ai_ref = git("rev-parse", "ai-branch")
except SystemExit:
    print("ERROR: local branch/ref 'ai-branch' was not found.")
    sys.exit(1)

status = git("status", "--porcelain")
if status:
    print("WARNING: repository already has uncommitted changes.")
    print("These changes are NOT discarded. Review them before continuing:\n")
    print(status)
    print("\nYou may continue, but check_changes.py will report them unless allowed.")

STATE.parent.mkdir(parents=True, exist_ok=True)
data = {
    "required_branch": "live",
    "baseline_commit": baseline,
    "ai_branch_commit": ai_ref,
    "allowed_paths": args.allow,
}
STATE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

print("Phase started.")
print(f"live baseline : {baseline}")
print(f"ai-branch ref : {ai_ref}")
print("Allowed paths:")
if args.allow:
    for path in args.allow:
        print(f"  - {path}")
else:
    print("  (none yet — diagnosis-only mode)")
print("\nRun: python tools/check_changes.py")
