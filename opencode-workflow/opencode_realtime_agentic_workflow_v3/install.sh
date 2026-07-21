#!/usr/bin/env sh
set -eu
TARGET="${1:-.}"
OPTIONS_EXCEL="${2:-}"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -n "$OPTIONS_EXCEL" ]; then
  python3 "$SCRIPT_DIR/installer/install.py" --target "$TARGET" --options-excel "$OPTIONS_EXCEL"
else
  python3 "$SCRIPT_DIR/installer/install.py" --target "$TARGET"
fi
