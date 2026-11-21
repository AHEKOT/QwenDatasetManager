"""Independent alpha-first matting runtime. No legacy ChromaKey imports."""

from .model import CleanMatte, MODEL_ID
from .inference import load_model, predict, recover_foreground

__all__ = ["CleanMatte", "MODEL_ID", "load_model", "predict", "recover_foreground"]
