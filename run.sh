#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting Qwen Dataset Manager..."

# Activate virtual environment
source .venv/bin/activate

# Run the application
python app.py
