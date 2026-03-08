import typer
from attr import dataclass
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from classification_trainer.configuration import DatasetInfo
from classification_trainer.configuration.training_info import TrainingInfo
from classification_trainer.helpers.dataset_helper import (
    add_eval_column,
    label_distribution,
    load_dataset_from_hf,
    log_dataset,
    prep_classification_dataset_for_training,
    union_datasets,
)
from classification_trainer.helpers.token_length_helper import (
    TokenLengthData,
    analyze_token_lengths,
    get_percent_samples_within_sequence_length,
)
from classification_trainer.helpers.tokenizer_helper import load_tokenizer_from_hf
from classification_trainer.protocols import CommandProtocol, LoggingProtocol


@dataclass
class AnalyzeDatasetCommand(CommandProtocol):
    training_info: TrainingInfo
    dataset_info: DatasetInfo
    merge_all_splits: bool

    def execute(self, logger: LoggingProtocol) -> None:
        try:
            dataset_info = self.dataset_info
            dataset_dict: DatasetDict = load_dataset_from_hf(dataset_info)
            dataset: Dataset = dataset_dict[dataset_info.training_split_name]
            if self.merge_all_splits:
                dataset = union_datasets(*dataset_dict.values())
            tokenizer: PreTrainedTokenizerBase = load_tokenizer_from_hf(self.training_info.base_model_info)
            dataset = prep_classification_dataset_for_training(
                dataset_info,
                self.training_info,
                dataset,
                tokenizer,
                self.training_info.base_model_info.chat_template_info,
                filter_long_content=False,
            )
            dataset = add_eval_column(dataset_info, self.training_info, dataset, tokenizer)
            logger.report_message(f"Sample row for {dataset_info.huggingface_name}")
            log_dataset(dataset, logger)
            logger.add_break()

            label_distributions: dict[str, float] = label_distribution(dataset_info, dataset)
            logger.report_message(f"Label distributions for {dataset_info.huggingface_name}")
            logger.report_table_message(label_distributions)

            result: TokenLengthData = analyze_token_lengths(dataset_info, dataset, tokenizer)

            logger.report_message(f"Sequence lengths for {dataset_info.huggingface_name}")
            logger.report_table_message(result._asdict())
            for seq_len in dataset_info.potential_sequence_lengths:
                self.produce_coverage_report_from_target(dataset, seq_len, tokenizer, logger)
        except Exception as e:
            logger.report_exception("Error Analyzing Dataset", e)
            raise typer.Exit(code=1) from e

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
