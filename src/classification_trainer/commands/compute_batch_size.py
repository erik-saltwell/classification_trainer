from dataclasses import dataclass

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase
from transformers.utils import logging as hf_logging

from classification_trainer.configuration import BaseModelInfo, DatasetInfo, TrainingInfo
from classification_trainer.helpers.batch_size_helper import find_max_batch_size
from classification_trainer.helpers.dataset_helper import (
    load_dataset_from_hf,
    make_stress_split,
    prep_classification_dataset_for_training,
)
from classification_trainer.helpers.tokenizer_helper import load_tokenizer_from_hf
from classification_trainer.protocols import CommmandProtocol, LoggingProtocol
from classification_trainer.utils import flush_gpu_memory


@dataclass
class ComputeBatchSizeCommand(CommmandProtocol):
    dataset_info: DatasetInfo
    base_model_info: BaseModelInfo
    training_info: TrainingInfo
    stress_set_rowcount: int

    def execute(self, logger: LoggingProtocol) -> None:
        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
        datasets: DatasetDict = load_dataset_from_hf(self.dataset_info)
        tokenizer: PreTrainedTokenizerBase = load_tokenizer_from_hf(self.base_model_info)
        training_dataset: Dataset = datasets[self.dataset_info.training_split_name]
        training_dataset = self.prep_for_stress_test(training_dataset, tokenizer, logger)
        evaluation_dataset = training_dataset
        if self.dataset_info.validation_split_name is not None:
            evaluation_dataset = datasets[self.dataset_info.validation_split_name]
            evaluation_dataset = self.prep_for_stress_test(evaluation_dataset, tokenizer, logger)

        del tokenizer
        flush_gpu_memory()
        result: int = find_max_batch_size(
            self.dataset_info,
            self.base_model_info,
            self.training_info,
            training_dataset,
            evaluation_dataset,
            logger,
        )

        # P1 — Actionable final output
        if result > 0:
            logger.report_message(
                f"[green]Recommended: set per_device_batch_size: {result} in your training YAML.[/green]"
            )
        else:
            logger.report_message("[red]Could not find a working batch size. Check GPU memory and model size.[/red]")

    def prep_for_stress_test(
        self, dataset: Dataset, tokenizer: PreTrainedTokenizerBase, logger: LoggingProtocol
    ) -> Dataset:
        return_dataset: Dataset = prep_classification_dataset_for_training(
            self.dataset_info,
            self.training_info,
            dataset,
            tokenizer,
            self.base_model_info.chat_template_info,
        )
        # A1 — pass return_dataset (post-prep) not the raw dataset
        return_dataset = make_stress_split(self.dataset_info, return_dataset, self.stress_set_rowcount, tokenizer)

        # P4 — warn when the dataset has fewer rows than requested
        actual_count = len(return_dataset)
        if actual_count < self.stress_set_rowcount:
            logger.report_message(
                f"Warning: requested {self.stress_set_rowcount} stress rows but dataset only has {actual_count} rows."
            )

        return return_dataset
