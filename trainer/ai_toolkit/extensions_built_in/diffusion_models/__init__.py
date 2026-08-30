"""Edit model registry intentionally limited to the models exposed by QDM."""

from .qwen_image import QwenImageEditPlusModel
from .flux2 import Flux2Klein4BModel, Flux2Klein9BModel


AI_TOOLKIT_MODELS = [
    QwenImageEditPlusModel,
    Flux2Klein4BModel,
    Flux2Klein9BModel,
]
