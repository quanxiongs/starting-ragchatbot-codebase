#!/bin/bash
# Development quality check script
# Usage:
#   ./format.sh          — format all Python files (auto-fix)
#   ./format.sh --check  — check formatting without modifying files (CI mode)

set -e

BACKEND_DIR="backend"

if [ "$1" == "--check" ]; then
    echo "Checking formatting (no files will be modified)..."
    uv run black --check "$BACKEND_DIR"
    echo "All files are correctly formatted."
else
    echo "Formatting Python files in $BACKEND_DIR/..."
    uv run black "$BACKEND_DIR"
    echo "Done."
fi
