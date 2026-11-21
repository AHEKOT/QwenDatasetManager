from .qwen_image_edit_plus_rgba import QwenImageEditPlusRGBAModel
from .flux2_rgba import Flux2Klein4BRGBAModel, Flux2Klein9BRGBAModel
from .h3_rgba import MinimaxH3RGBAModel, MinimaxH3Ref2VARGBAModel
from toolkit.extension import Extension


class QwenRGBAVAETrainingExtension(Extension):
    uid = "qwen_rgba_vae_trainer"
    name = "Qwen RGBA VAE Compatibility Trainer"

    @classmethod
    def get_process(cls):
        from .qwen_rgba_vae_trainer import QwenRGBAVAETrainProcess

        return QwenRGBAVAETrainProcess


class Flux2RGBAVAETrainingExtension(Extension):
    uid = "flux2_rgba_vae_trainer"
    name = "FLUX.2 Klein RGBA VAE Compatibility Trainer"

    @classmethod
    def get_process(cls):
        from .flux2_rgba_vae_trainer import Flux2RGBAVAETrainProcess

        return Flux2RGBAVAETrainProcess


class H3RGBAVAETrainingExtension(Extension):
    uid = 'h3_rgba_vae_trainer'
    name = 'MiniMax H3 RGBA VAE Compatibility Trainer'

    @classmethod
    def get_process(cls):
        from .h3_rgba_vae_trainer import H3RGBAVAETrainProcess
        return H3RGBAVAETrainProcess


AI_TOOLKIT_MODELS = [
    MinimaxH3RGBAModel,
    MinimaxH3Ref2VARGBAModel,
    QwenImageEditPlusRGBAModel,
    Flux2Klein4BRGBAModel,
    Flux2Klein9BRGBAModel,
]


AI_TOOLKIT_EXTENSIONS = [
    H3RGBAVAETrainingExtension,
    QwenRGBAVAETrainingExtension,
    Flux2RGBAVAETrainingExtension,
]
