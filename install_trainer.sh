#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINER_DIR="$SCRIPT_DIR/trainer"
PIP_VERSION="26.2.1"
HATCHLING_VERSION="1.32.0"
PIP_WHEEL_URL="https://files.pythonhosted.org/packages/f3/6e/1736e5b4ae2b778ef2f81c47d797de9f891d4d8acb047a24ca37a60294dd/pip-26.2.1-py3-none-any.whl"
PIP_WHEEL_SHA256="71138adf1f4ca900cdb7d289c21b7494329f2332b6d85f0e1c42108c0384ed3e"

if [ "$(uname -s)" = "Darwin" ]; then
  echo "CUDA trainer installation requires Linux or Windows with an NVIDIA GPU." >&2
  exit 1
fi
if ! command -v python3.12 >/dev/null 2>&1; then
  echo "Python 3.12 and its venv module are required for the CUDA trainer." >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "Git is required because AI Toolkit pins a diffusers Git revision." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "An NVIDIA GPU and current NVIDIA driver are required." >&2
  exit 1
fi

bootstrap_pip() {
  local python_executable="$1"
  local installed_version
  installed_version="$($python_executable -c 'import pip; print(pip.__version__)')"
  if [ "$installed_version" = "$PIP_VERSION" ]; then
    echo "pip $PIP_VERSION is already installed."
    return
  fi

  local wheel_path
  local wheel_directory
  local actual_hash
  wheel_directory="$(mktemp -d)"
  wheel_path="$wheel_directory/pip-$PIP_VERSION-py3-none-any.whl"
  "$python_executable" -c \
    'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' \
    "$PIP_WHEEL_URL" "$wheel_path"
  actual_hash="$(sha256sum "$wheel_path" | awk '{print $1}')"
  if [ "$actual_hash" != "$PIP_WHEEL_SHA256" ]; then
    rm -rf -- "$wheel_directory"
    echo "pip wheel checksum mismatch." >&2
    return 1
  fi
  "$python_executable" -m pip install \
    --disable-pip-version-check \
    --no-deps \
    --force-reinstall \
    "$wheel_path"
  rm -rf -- "$wheel_directory"
}

echo "=== Installing Qwen Dataset Manager CUDA trainer ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if [ ! -x "$TRAINER_DIR/.venv/bin/python" ]; then
  if [ -d "$TRAINER_DIR/.venv" ]; then
    python3.12 -m venv --clear "$TRAINER_DIR/.venv"
  else
    python3.12 -m venv "$TRAINER_DIR/.venv"
  fi
fi

TRAINER_PYTHON="$TRAINER_DIR/.venv/bin/python"
TRAINER_MINOR="$($TRAINER_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ "$TRAINER_MINOR" != "3.12" ]; then
  echo "Recreating trainer environment with Python 3.12."
  python3.12 -m venv --clear "$TRAINER_DIR/.venv"
fi

bootstrap_pip "$TRAINER_PYTHON"
"$TRAINER_PYTHON" -m pip install \
  --disable-pip-version-check \
  --no-cache-dir \
  torch==2.13.0 \
  torchvision==0.28.0 \
  torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130

# Preinstalling hatchling and disabling build isolation avoids a reproducible
# hang while pip prepares the pinned diffusers revision.
"$TRAINER_PYTHON" -m pip install \
  --disable-pip-version-check \
  --no-cache-dir \
  "hatchling==$HATCHLING_VERSION"
"$TRAINER_PYTHON" -m pip install \
  --disable-pip-version-check \
  --no-cache-dir \
  --no-build-isolation \
  -r "$TRAINER_DIR/ai_toolkit/requirements.txt"

"$TRAINER_PYTHON" -m pip check
"$TRAINER_PYTHON" -c "import torch, diffusers, transformers, bitsandbytes, peft; assert torch.cuda.is_available(), 'PyTorch installed, but CUDA is not available'; print('CUDA trainer ready:', torch.__version__, torch.cuda.get_device_name(0))"
"$TRAINER_PYTHON" "$TRAINER_DIR/ai_toolkit/run.py" --help

echo "Trainer installation complete. Restart Qwen Dataset Manager."
