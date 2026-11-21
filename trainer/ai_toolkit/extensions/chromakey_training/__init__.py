"""QDM fully-neural anime chromakey training extension."""

from toolkit.extension import Extension


class QDMChromaKeyTrainingExtension(Extension):
    uid = "qdm_chromakey_trainer"
    name = "QDM KeyMatte V4 Trainer"

    @classmethod
    def get_process(cls):
        from .trainer import QDMChromaKeyTrainProcess

        return QDMChromaKeyTrainProcess


AI_TOOLKIT_EXTENSIONS = [QDMChromaKeyTrainingExtension]
