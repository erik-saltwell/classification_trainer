from dataclasses import dataclass

import typer
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.utils import logging as hf_logging

from classification_trainer.commands.training_runner import TrainingRunner
from classification_trainer.configuration import DatasetInfo, TrainingInfo
from classification_trainer.helpers.dataset_helper import (
    make_stress_split,
    prep_classification_dataset_for_training,
)
from classification_trainer.protocols import CommandProtocol, LoggingProtocol


@dataclass
class ComputeBatchSizeCommand(CommandProtocol):
    training_info: TrainingInfo
    dataset_info: DatasetInfo
    stress_set_rowcount: int
    pretokenize: bool

    def minimize_hf_output(self) -> None:
        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()

    def execute(self, logger: LoggingProtocol) -> None:
        try:
            self.minimize_hf_output()
            runner: TrainingRunner = TrainingRunner(self.training_info, self.dataset_info, self.pretokenize)
            runner.prepare_data(logger)
            runner.load_model(logger)
            result: int = runner.compute_max_batch_size(self.stress_set_rowcount, logger)

            # P1 — Actionable final output
            if result > 0:
                logger.report_message(
                    f"[green]Recommended: set per_device_batch_size: {result} in your training YAML.[/green]"
                )
            else:
                logger.report_message(
                    "[red]Could not find a working batch size. Check GPU memory and model size.[/red]"
                )
        except Exception as e:
            logger.report_exception("Error Computing Batch Size", e)
            raise typer.Exit(code=1) from e

    def prep_for_stress_test(
        self, dataset_info: DatasetInfo, dataset: Dataset, tokenizer: PreTrainedTokenizerBase, logger: LoggingProtocol
    ) -> Dataset:
        return_dataset: Dataset = prep_classification_dataset_for_training(
            dataset_info,
            self.training_info,
            dataset,
            tokenizer,
            self.training_info.base_model_info.chat_template_info,
        )
        # A1 — pass return_dataset (post-prep) not the raw dataset
        return_dataset = make_stress_split(dataset_info, return_dataset, self.stress_set_rowcount, tokenizer)

        # P4 — warn when the dataset has fewer rows than requested
        actual_count = len(return_dataset)
        if actual_count < self.stress_set_rowcount:
            logger.report_message(
                f"Warning: requested {self.stress_set_rowcount} stress rows but dataset only has {actual_count} rows."
            )

        return return_dataset
