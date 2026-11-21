"""Shared neural architecture and inference helpers for QDM AnimeKeyMatte."""

from .model import (
    AnimeKeyMatte, AnimeKeyMatteV2, AnimeKeyMatteV3,
    ARCHITECTURE_ID, ARCHITECTURE_ID_V2, ARCHITECTURE_ID_V3,
)
from .inference import load_model, run_tiled
from .v4 import KeyMatteV4, ARCHITECTURE_ID_V4

__all__ = [
    "AnimeKeyMatte", "AnimeKeyMatteV2", "AnimeKeyMatteV3",
    "ARCHITECTURE_ID", "ARCHITECTURE_ID_V2", "ARCHITECTURE_ID_V3",
    "load_model", "run_tiled",
    "KeyMatteV4", "ARCHITECTURE_ID_V4",
]
