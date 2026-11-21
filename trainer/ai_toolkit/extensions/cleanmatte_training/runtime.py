"""Locate the self-contained package also distributed with the ComfyUI node."""
import sys
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parents[4] / "ComfyUI-QDM-ChromaKey"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from cleanmatte import CleanMatte, MODEL_ID, predict, recover_foreground  # noqa: E402
