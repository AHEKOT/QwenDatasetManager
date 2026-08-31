from .qwen_image_edit_plus_rgba import QwenImageEditPlusRGBAModel
from .flux2_rgba import Flux2Klein4BRGBAModel, Flux2Klein9BRGBAModel
from toolkit.extension import Extension


class QwenRGBAVAETrainingExtension(Extension):
    uid = "qwen_rgba_vae_trainer"
    name = "Qwen RGBA VAE Compatibility Trainer"

    @classmethod
    def get_process(cls):
        from .qwen_rgba_vae_trainer import QwenRGBAVAETrainProcess

        return QwenRGBAVAETrainProcess


AI_TOOLKIT_MODELS = [
    QwenImageEditPlusRGBAModel,
    Flux2Klein4BRGBAModel,
    Flux2Klein9BRGBAModel,
]


AI_TOOLKIT_EXTENSIONS = [
    QwenRGBAVAETrainingExtension,
]
