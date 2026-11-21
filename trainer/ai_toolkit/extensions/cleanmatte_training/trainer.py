from jobs.process.BaseExtensionProcess import BaseExtensionProcess
from .engine import Trainer


class CleanMatteProcess(BaseExtensionProcess):
    def run(self):
        super().run()
        Trainer(self.config, self.job.name).run()
