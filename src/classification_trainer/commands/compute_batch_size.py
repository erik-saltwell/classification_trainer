from dataclasses import dataclass

from classification_trainer.configuration import BaseModelInfo, DatasetInfo, TrainingInfo
from classification_trainer.protocols import CommmandProtocol, LoggingProtocol


@dataclass
class ComputeBatchSizeCommand(CommmandProtocol):
    dataset_info: DatasetInfo
    base_model_info: BaseModelInfo
    training_info: TrainingInfo

    def execute(self, logger: LoggingProtocol) -> None: ...
