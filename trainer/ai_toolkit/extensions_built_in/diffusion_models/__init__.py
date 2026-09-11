"""Edit model registry intentionally limited to the models exposed by QDM."""

from .minimax_h3 import MinimaxH3Model, MinimaxH3Ref2VAModel
from .qwen_image import QwenImageEditPlusModel
from .flux2 import Flux2Klein4BModel, Flux2Klein9BModel


AI_TOOLKIT_MODELS = [
    MinimaxH3Model,
    MinimaxH3Ref2VAModel,
    QwenImageEditPlusModel,
    Flux2Klein4BModel,
    Flux2Klein9BModel,
]
