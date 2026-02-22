from attr import dataclass

from classification_trainer.configuration import BaseModelInfo, DatasetInfo
from classification_trainer.protocols import CommmandProtocol, LoggingProtocol


@dataclass
class AnalyzeSequenceLengthCommand(CommmandProtocol):
    dataset_info: DatasetInfo
    base_model_info: BaseModelInfo

    def execute(self, logger: LoggingProtocol) -> None:
        return
