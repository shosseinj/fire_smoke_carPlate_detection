from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    ".opencode/instructions/on-demand-gpu-broadcast.md",
    ".opencode/commands/implement-on-demand-gpu-broadcast.md",
    ".opencode/agents/broadcast-orchestrator.md",
    ".opencode/agents/pipeline-analyst.md",
    ".opencode/agents/broadcast-architect.md",
    ".opencode/agents/gpu-pipeline-implementer.md",
    ".opencode/agents/session-api-implementer.md",
    ".opencode/agents/frontend-stream-implementer.md",
    ".opencode/agents/broadcast-test-designer.md",
    ".opencode/agents/isolation-reviewer.md",
    ".opencode/agents/runtime-validator.md",
    "docs/on-demand-gpu-broadcast-design.md",
    "docs/on-demand-gpu-broadcast-implementation-plan.md",
    ".agentic/broadcast/state.json",
    ".agentic/broadcast/task-board.md",
]
missing = [p for p in REQUIRED if not (ROOT / p).is_file()]
if missing:
    print("Missing workflow files:")
    for item in missing:
        print(f" - {item}")
    sys.exit(1)
print(f"Workflow OK: {len(REQUIRED)} required files found.")
