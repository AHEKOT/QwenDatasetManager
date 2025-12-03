#!/bin/bash

echo "🚀 Installing Qwen Dataset Manager..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "✅ Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Installation complete!"
echo ""
echo "To run the application:"
echo "  ./run.sh"
echo ""
