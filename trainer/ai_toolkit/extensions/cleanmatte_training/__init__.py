"""New training entry point; independent of chromakey_training."""
from toolkit.extension import Extension


class CleanMatteExtension(Extension):
    uid = "qdm_cleanmatte_trainer"
    name = "CleanMatte: alpha-first chromakey"

    @classmethod
    def get_process(cls):
        from .trainer import CleanMatteProcess
        return CleanMatteProcess


AI_TOOLKIT_EXTENSIONS = [CleanMatteExtension]
