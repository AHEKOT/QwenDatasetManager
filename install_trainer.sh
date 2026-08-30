#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINER_DIR="$SCRIPT_DIR/trainer"

if [ "$(uname -s)" = "Darwin" ]; then
  echo "CUDA trainer installation requires Linux or Windows with an NVIDIA GPU."
  exit 1
fi

echo "Installing Qwen Dataset Manager CUDA trainer..."
python3.12 -m venv "$TRAINER_DIR/.venv"
source "$TRAINER_DIR/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install --no-cache-dir -r "$TRAINER_DIR/ai_toolkit/requirements.txt"

python -c "import torch; assert torch.cuda.is_available(), 'PyTorch installed, but CUDA is not available'; print('CUDA trainer ready:', torch.cuda.get_device_name(0))"
echo "Trainer installation complete. Restart Qwen Dataset Manager."
