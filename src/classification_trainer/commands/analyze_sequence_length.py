from attr import dataclass

from classification_trainer.configuration import BaseModelInfo, DatasetInfo
from classification_trainer.protocols import CommmandProtocol, LoggingProtocol


@dataclass
class AnalyzeSequenceLengthCommand(CommmandProtocol):
    dataset_info: DatasetInfo
    base_model_info: BaseModelInfo
    merge_all_splits: bool

    def execute(self, logger: LoggingProtocol) -> None:
        # dataset_dict: DatasetDict = load_dataset_from_hf(self.dataset_info)
        # dataset: Dataset = dataset_dict[self.dataset_info.training_split_name]
        # if self.merge_all_splits:
        #     dataset = union_datasets(dataset_dict)

        return
