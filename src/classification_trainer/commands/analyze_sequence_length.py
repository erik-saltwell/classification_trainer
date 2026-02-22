from attr import dataclass
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from classification_trainer.configuration import BaseModelInfo, DatasetInfo
from classification_trainer.configuration.training_info import TrainingInfo
from classification_trainer.helpers.dataset_helper import (
    load_dataset_from_hf,
    prep_classification_dataset_for_training,
    union_datasets,
)
from classification_trainer.helpers.token_length_helper import (
    TokenLengthData,
    analyze_token_lengths,
    get_percent_samples_within_sequence_length,
)
from classification_trainer.helpers.tokenizer_helper import load_tokenizer_from_hf
from classification_trainer.protocols import CommmandProtocol, LoggingProtocol


@dataclass
class AnalyzeSequenceLengthCommand(CommmandProtocol):
    dataset_info: DatasetInfo
    base_model_info: BaseModelInfo
    training_info: TrainingInfo
    merge_all_splits: bool

    def execute(self, logger: LoggingProtocol) -> None:
        dataset_dict: DatasetDict = load_dataset_from_hf(self.dataset_info)
        dataset: Dataset = dataset_dict[self.dataset_info.training_split_name]
        if self.merge_all_splits:
            dataset = union_datasets(*dataset_dict.values())
        tokenizer: PreTrainedTokenizerBase = load_tokenizer_from_hf(self.base_model_info)
        dataset = prep_classification_dataset_for_training(self.dataset_info, self.training_info, dataset, tokenizer)

        result: TokenLengthData = analyze_token_lengths(self.dataset_info, dataset, tokenizer)

        logger.report_table_message(result._asdict())
        self.produce_coverage_report_from_target(dataset, 1024, tokenizer, logger)
        self.produce_coverage_report_from_target(dataset, 1536, tokenizer, logger)
        self.produce_coverage_report_from_target(dataset, 2048, tokenizer, logger)

    def produce_coverage_report_from_target(
        self, dataset: Dataset, target_sequence_len: int, tokenizer: PreTrainedTokenizerBase, logger: LoggingProtocol
    ) -> None:
        coverage_percent: float = get_percent_samples_within_sequence_length(
            self.dataset_info, dataset, tokenizer, target_sequence_len
        )
        coverage_loss_count = (1.0 - coverage_percent) * len(dataset)
        logger.report_message(
            f"Coverage for {target_sequence_len}: {coverage_percent}%. Loss: {coverage_loss_count} samples."
        )
