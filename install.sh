#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Installing Qwen Dataset Manager..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "✅ Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ""
echo "✅ Installation complete!"
echo ""
echo "To run the application:"
echo "  ./run.sh"
echo ""
