#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIP_VERSION="26.2.1"
PIP_WHEEL_URL="https://files.pythonhosted.org/packages/f3/6e/1736e5b4ae2b778ef2f81c47d797de9f891d4d8acb047a24ca37a60294dd/pip-26.2.1-py3-none-any.whl"
PIP_WHEEL_SHA256="71138adf1f4ca900cdb7d289c21b7494329f2332b6d85f0e1c42108c0384ed3e"
SKIP_TRAINER=0

for argument in "$@"; do
  case "$argument" in
    --skip-trainer) SKIP_TRAINER=1 ;;
    *)
      echo "Unknown option: $argument" >&2
      echo "Usage: ./install.sh [--skip-trainer]" >&2
      exit 2
      ;;
  esac
done

cd "$SCRIPT_DIR"

if command -v python3.12 >/dev/null 2>&1; then
  BOOTSTRAP_PYTHON="$(command -v python3.12)"
elif [ "$SKIP_TRAINER" -eq 1 ] && command -v python3 >/dev/null 2>&1; then
  BOOTSTRAP_PYTHON="$(command -v python3)"
else
  echo "Python 3.12 is required. Install Python 3.12 and its venv module, then rerun." >&2
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

echo "=== Installing Qwen Dataset Manager ==="
if [ ! -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  if [ -d "$SCRIPT_DIR/.venv" ]; then
    "$BOOTSTRAP_PYTHON" -m venv --clear "$SCRIPT_DIR/.venv"
  else
    "$BOOTSTRAP_PYTHON" -m venv "$SCRIPT_DIR/.venv"
  fi
fi

APP_PYTHON="$SCRIPT_DIR/.venv/bin/python"
bootstrap_pip "$APP_PYTHON"
if command -v nvidia-smi >/dev/null 2>&1; then
  LLAMA_CPP_WHEEL_INDEX="https://abetlen.github.io/llama-cpp-python/whl/cu130"
  echo "Installing the CUDA 13.0 llama.cpp wheel for local vision captioning."
else
  LLAMA_CPP_WHEEL_INDEX="https://abetlen.github.io/llama-cpp-python/whl/cpu"
  echo "NVIDIA GPU was not detected; installing the CPU llama.cpp wheel."
fi
"$APP_PYTHON" -m pip install \
  --disable-pip-version-check \
  --no-cache-dir \
  --extra-index-url "$LLAMA_CPP_WHEEL_INDEX" \
  -r "$SCRIPT_DIR/requirements.txt"
"$APP_PYTHON" -m pip check
"$APP_PYTHON" -c "import flask, PIL, llama_cpp, app; print('Application import check passed; llama.cpp:', llama_cpp.__version__)"

if [ "$SKIP_TRAINER" -eq 0 ]; then
  bash "$SCRIPT_DIR/install_trainer.sh"
fi

echo ""
echo "=== Installation complete ==="
echo "Run ./run.sh, then open http://127.0.0.1:5001"
