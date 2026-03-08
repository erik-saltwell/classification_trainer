from dataclasses import dataclass

import typer
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.utils import logging as hf_logging

from classification_trainer.commands.training_runner import TrainingRunner
from classification_trainer.configuration import DatasetInfo, TrainingInfo
from classification_trainer.helpers.dataset_helper import (
    make_stress_split,
    prep_dataset,
)
from classification_trainer.protocols import CommandProtocol, LoggingProtocol
from classification_trainer.utils import CommonPaths, flush_gpu_memory


@dataclass
class ComputeBatchSizeCommand(CommandProtocol):
    training_info: TrainingInfo
    dataset_info: DatasetInfo
    stress_set_rowcount: int

    def execute(self, logger: LoggingProtocol) -> None:
        CommonPaths.get().clear_cache_model_directories(self.training_info.model_name)
        try:
            hf_logging.disable_progress_bar()
            hf_logging.set_verbosity_error()
            runner: TrainingRunner = TrainingRunner(self.training_info, self.dataset_info, False)
            runner.prepare_stress_data(self.stress_set_rowcount, logger)
            runner.load_model(logger)

            # dataset_info = self.dataset_info
            # datasets: DatasetDict = load_dataset_from_hf(dataset_info)
            # tokenizer: PreTrainedTokenizerBase = load_tokenizer_from_hf(self.training_info.base_model_info)
            # training_dataset: Dataset = datasets[dataset_info.training_split_name]
            # training_dataset = self.prep_for_stress_test(dataset_info, training_dataset, tokenizer, logger)
            # evaluation_dataset = training_dataset
            # if dataset_info.validation_split_name is not None:
            #     evaluation_dataset = datasets[dataset_info.validation_split_name]
            #     evaluation_dataset = self.prep_for_stress_test(dataset_info, evaluation_dataset, tokenizer, logger)

            # result: int = find_max_batch_size(
            #     self.dataset_info,
            #     self.training_info.base_model_info,
            #     self.training_info,
            #     runner.training_split,
            #     runner.validation_split,
            #     logger,
            # )

            flush_gpu_memory()
            result: int = runner.find_max_batch_size(logger)
            del runner
            flush_gpu_memory()

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
        return_dataset: Dataset = prep_dataset(
            self.training_info, self.training_info.base_model_info, dataset_info, dataset, tokenizer, False
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
