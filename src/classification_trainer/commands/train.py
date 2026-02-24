from attr import dataclass
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from classification_trainer.configuration import BaseModelInfo, DatasetInfo, TrainingInfo
from classification_trainer.helpers.dataset_helper import (
    load_dataset_from_hf,
    prep_classification_dataset_for_training,
    split_dataset,
)
from classification_trainer.helpers.tokenizer_helper import load_tokenizer_from_hf
from classification_trainer.protocols import CommmandProtocol, LoggingProtocol


@dataclass
class TrainCommand(CommmandProtocol):
    dataset_info: DatasetInfo
    base_model_info: BaseModelInfo
    training_info: TrainingInfo

    def execute(self, logger: LoggingProtocol) -> None:
        datasets: DatasetDict = load_dataset_from_hf(self.dataset_info)
        training_dataset: Dataset = datasets[self.dataset_info.training_split_name]

        if self.dataset_info.validation_split_name is None:
            datasets, self.dataset_info = split_dataset(self.dataset_info, training_dataset, 0.1, 0.1, 3414)
            training_dataset = datasets[self.dataset_info.training_split_name]
        test_dataset: Dataset = datasets[self.dataset_info.test_split_name]
        validation_dataset: Dataset = datasets[self.dataset_info.validation_split_name]
        tokenizer: PreTrainedTokenizerBase = load_tokenizer_from_hf(self.base_model_info)
        training_dataset = prep_classification_dataset_for_training(
            self.dataset_info,
            self.training_info,
            training_dataset,
            tokenizer,
            self.base_model_info.chat_template_info,
            True,
        )
        validation_dataset = prep_classification_dataset_for_training(
            self.dataset_info,
            self.training_info,
            validation_dataset,
            tokenizer,
            self.base_model_info.chat_template_info,
            True,
        )
        test_dataset = prep_classification_dataset_for_training(
            self.dataset_info,
            self.training_info,
            test_dataset,
            tokenizer,
            self.base_model_info.chat_template_info,
            True,
        )
