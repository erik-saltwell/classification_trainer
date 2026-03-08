import typer
from attr import dataclass
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from classification_trainer.configuration import DatasetInfo
from classification_trainer.configuration.training_info import TrainingInfo
from classification_trainer.helpers.dataset_helper import label_distribution, log_dataset, union_datasets
from classification_trainer.helpers.token_length_helper import (
    TokenLengthData,
    analyze_token_lengths,
    get_percent_samples_within_sequence_length,
)
from classification_trainer.helpers.tokenizer_helper import load_tokenizer_from_hf
from classification_trainer.protocols import CommandProtocol, LoggingProtocol

from .training_runner import TrainingRunner


@dataclass
class AnalyzeDatasetCommand(CommandProtocol):
    training_info: TrainingInfo
    dataset_info: DatasetInfo
    merge_all_splits: bool

    def execute(self, logger: LoggingProtocol) -> None:
        try:
            logger.report_message(f"[blue]Analyzing Dataset: {self.dataset_info.huggingface_name}...[/blue]")
            runner: TrainingRunner = TrainingRunner(self.training_info, self.dataset_info)
            runner.prepare_data(logger)
            dataset = runner.training_split
            if self.merge_all_splits:
                dataset = union_datasets(runner.training_split, runner.test_split, runner.validation_split)

            logger.report_message("Sample row:")
            log_dataset(dataset, logger)
            logger.add_break()

            label_distributions: dict[str, float] = label_distribution(runner.dataset_info, dataset)
            logger.report_message("Label distributions:")
            logger.report_table_message(label_distributions)

            tokenizer: PreTrainedTokenizerBase = load_tokenizer_from_hf(self.training_info.base_model_info)
            result: TokenLengthData = analyze_token_lengths(runner.dataset_info, dataset, tokenizer)

            logger.report_message("Sequence lengths:")
            logger.report_table_message(result._asdict())
            for seq_len in runner.dataset_info.potential_sequence_lengths:
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
