#!/usr/bin/env bash
set -euo pipefail
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESTINATION="$(pwd)"
mkdir -p "$DESTINATION/.opencode" "$DESTINATION/docs" "$DESTINATION/scripts" "$DESTINATION/.agentic"
cp -a "$SOURCE/.opencode/." "$DESTINATION/.opencode/"
cp -a "$SOURCE/docs/." "$DESTINATION/docs/"
cp -a "$SOURCE/scripts/." "$DESTINATION/scripts/"
cp -a "$SOURCE/.agentic/broadcast" "$DESTINATION/.agentic/"
python3 "$DESTINATION/scripts/validate_broadcast_workflow.py"
echo "Run OpenCode and enter: /implement-on-demand-gpu-broadcast"
